"""Context-Conditioned & Unimodal Coordinate Gibbs Sequence Design (Section 2.6, Eq. 7).

Because LogiCA preserves each BioFM's native categorical token head, sequences can be
sampled directly from context-conditioned token distributions without training a separate
generative decoder:
  x_i^{(t+1)} ~ softmax( [ log pi_theta(. | x_{\\i}^{(t)}, y)
                          + beta * (log pi_theta(. | x_{\\i}^{(t)}, y) - log pi_theta(. | x_{\\i}^{(t)}, empty)) ] / tau )

Supports:
  1. De novo / pocket-targeted **Coordinate Pseudo-Gibbs & Metropolis-within-Gibbs** sequence
     sampling (:meth:`GibbsSampler.sample_sequences`) for Dual-BioFM or Single-BioFM models,
     including automatic zero-shot KL hotspot selection when ``design_positions=None``.
  2. Retrospective **DMS Landscape Gibbs Evaluation** (:func:`evaluate_landscape_gibbs`)
     implementing the three manuscript protocols (Within-Site Coordinate Gibbs, Global
     Boltzmann-Gibbs, and Multi-Step Coordinate MCMC Trajectories).
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Sequence

import numpy as np
import torch
import torch.nn.functional as F

from .backbones import CANONICAL_AAS
from .model import LogiCA


@dataclass
class GibbsDesignResult:
    """Structured output of :meth:`GibbsSampler.sample_sequences`."""

    seed_sequence: str
    context: str | None
    design_positions_0based: list[int]
    final_sequences: list[str]
    final_scores: np.ndarray  # shape (num_chains,)
    score_trajectories: np.ndarray  # shape (num_chains, num_steps + 1)
    site_probability_matrix: np.ndarray  # shape (len(design_positions), len(alphabet))
    alphabet: tuple[str, ...]
    effective_sample_size: float

    def top_designs(self, k: int = 5, deduplicate: bool = True) -> list[tuple[str, float]]:
        """Returns top-scoring sampled sequences ``[(sequence, score), ...]``."""
        order = np.argsort(-self.final_scores)
        seen: set[str] = set()
        out: list[tuple[str, float]] = []
        for idx in order:
            seq = self.final_sequences[int(idx)]
            if deduplicate and seq in seen:
                continue
            seen.add(seq)
            out.append((seq, float(self.final_scores[int(idx)])))
            if len(out) >= k:
                break
        return out


class GibbsSampler:
    """Decoder-free Coordinate Gibbs & Score-Guided MCMC sequence designer for :class:`LogiCA`."""

    def __init__(self, model: LogiCA) -> None:
        self.model = model

    @torch.no_grad()
    def site_conditional_distributions(
        self,
        sequence: str,
        positions_0based: Sequence[int],
        context: str | None = None,
        *,
        temperature: float = 0.3,
        guided_beta: float = 0.5,
        alphabet: Sequence[str] = CANONICAL_AAS,
        max_primary_len: int = 1022,
        max_partner_len: int = 256,
    ) -> np.ndarray:
        """Computes normalized ``(|positions|, |alphabet|)`` Coordinate Gibbs distributions."""
        if temperature <= 0.0:
            raise ValueError("temperature must be > 0")
        device = self.model.device
        p_enc = self.model.primary_backbone.tokenize(
            sequence, max_length=max_primary_len + 2, device=device
        )
        valid_tok_pos = torch.nonzero(
            (p_enc["attention_mask"][0] == 1) & (~p_enc["special_tokens_mask"][0].bool()),
            as_tuple=False,
        ).flatten().tolist()

        base_ids = p_enc["input_ids"][0]
        base_attn = p_enc["attention_mask"][0]
        mask_id = self.model.primary_backbone.mask_token_id
        vocab_ids = torch.tensor(
            [self.model.primary_backbone.token_to_id(a) for a in alphabet],
            dtype=torch.long,
            device=device,
        )

        batch_rows = []
        tok_targets = []
        for p0 in positions_0based:
            row = base_ids.clone()
            t_idx = valid_tok_pos[int(p0)]
            row[t_idx] = mask_id
            batch_rows.append(row)
            tok_targets.append(t_idx)

        b_ids = torch.stack(batch_rows, dim=0)
        b_attn = base_attn.unsqueeze(0).expand(b_ids.size(0), -1)
        p_last, p_qkv = self.model.encode_primary(b_ids, b_attn)

        if self.model.is_dual and context is not None and self.model.partner_backbone is not None:
            d_enc = self.model.partner_backbone.tokenize(
                context, max_length=max_partner_len, device=device
            )
            d_last = self.model.encode_partner(d_enc["input_ids"], d_enc["attention_mask"])
            out = self.model.forward_from_hidden(
                p_last,
                b_attn,
                d_last.expand(b_ids.size(0), -1, -1),
                d_enc["attention_mask"].expand(b_ids.size(0), -1),
                primary_adapter_input=p_qkv,
            )
        else:
            out = self.model.forward_from_hidden(p_last, b_attn, primary_adapter_input=p_qkv)

        row_idx = torch.arange(b_ids.size(0), device=device)
        t_pos_t = torch.tensor(tok_targets, dtype=torch.long, device=device)

        holo_sub = out.primary_holo_logits[row_idx, t_pos_t][:, vocab_ids].float()
        apo_sub = out.primary_apo_logits[row_idx, t_pos_t][:, vocab_ids].float()
        holo_lp = F.log_softmax(holo_sub, dim=-1)
        apo_lp = F.log_softmax(apo_sub, dim=-1)

        potentials = (holo_lp + guided_beta * (holo_lp - apo_lp)) / temperature
        probs = F.softmax(potentials, dim=-1)
        return probs.cpu().numpy().astype(np.float64)

    @torch.no_grad()
    def sample_sequences(
        self,
        seed_sequence: str,
        context: str | None = None,
        *,
        design_positions_0based: Sequence[int] | None = None,
        auto_kl_top_k: int = 6,
        num_chains: int = 16,
        num_steps: int = 12,
        temperature: float = 0.3,
        guided_beta: float = 0.5,
        metropolis_accept: bool = True,
        alphabet: Sequence[str] = CANONICAL_AAS,
        seed: int = 42,
        max_primary_len: int = 1022,
        max_partner_len: int = 256,
    ) -> GibbsDesignResult:
        """Runs ``num_chains`` parallel Coordinate Gibbs chains for ``num_steps`` steps (Eq. 7).

        If ``design_positions_0based`` is ``None`` and ``context`` is provided on a Dual-BioFM
        model, automatically discovers the top ``auto_kl_top_k`` partner-sensitive sites via
        zero-shot cross-modal KL divergence ``D_KL(pi(.|x,y) || pi(.|x))``!
        """
        if temperature <= 0.0:
            raise ValueError("temperature must be > 0")
        was_training = self.model.training
        self.model.eval()
        device = self.model.device
        rng = np.random.default_rng(seed)
        alpha_tuple = tuple(alphabet)
        aa_to_col = {a: i for i, a in enumerate(alpha_tuple)}

        if design_positions_0based is None:
            if self.model.is_dual and context is not None:
                kl_info = self.model.detect_kl_hotspots(
                    seed_sequence, context, top_k=auto_kl_top_k, alphabet=alpha_tuple
                )
                design_pos = sorted(int(i) for i in kl_info["top_indices_0based"])  # type: ignore[union-attr]
            else:
                design_pos = list(range(min(len(seed_sequence), auto_kl_top_k)))
        else:
            design_pos = [int(p) for p in design_positions_0based]

        if not design_pos:
            raise ValueError("design_positions_0based must contain at least one site.")

        # Initial site probability matrix on the seed sequence (for visualization & diagnostics)
        init_pwm = self.site_conditional_distributions(
            seed_sequence,
            design_pos,
            context=context,
            temperature=temperature,
            guided_beta=guided_beta,
            alphabet=alpha_tuple,
            max_primary_len=max_primary_len,
            max_partner_len=max_partner_len,
        )

        # Pre-encode partner context ONCE across all chains and all MCMC steps!
        use_partner = self.model.is_dual and context is not None and self.model.partner_backbone is not None
        if use_partner:
            assert self.model.partner_backbone is not None
            d_enc = self.model.partner_backbone.tokenize(
                context, max_length=max_partner_len, device=device
            )
            d_last_1 = self.model.encode_partner(d_enc["input_ids"], d_enc["attention_mask"])
            d_attn_1 = d_enc["attention_mask"]
        else:
            d_last_1 = None
            d_attn_1 = None

        vocab_ids = torch.tensor(
            [self.model.primary_backbone.token_to_id(a) for a in alpha_tuple],
            dtype=torch.long,
            device=device,
        )
        mask_id = self.model.primary_backbone.mask_token_id

        # Tokenize seed sequence once to establish token coordinates
        wt_enc = self.model.primary_backbone.tokenize(
            seed_sequence, max_length=max_primary_len + 2, device=device
        )
        valid_tok_pos = torch.nonzero(
            (wt_enc["attention_mask"][0] == 1) & (~wt_enc["special_tokens_mask"][0].bool()),
            as_tuple=False,
        ).flatten().tolist()
        wt_ids = wt_enc["input_ids"][0]
        base_attn = wt_enc["attention_mask"]

        # Precompute WT site reference log-probs for fast Eq. 3 score tracking
        wt_masked_rows = []
        for p0 in design_pos:
            r = wt_ids.clone()
            r[valid_tok_pos[p0]] = mask_id
            wt_masked_rows.append(r)
        wt_b_ids = torch.stack(wt_masked_rows, dim=0)
        wt_b_attn = base_attn.expand(wt_b_ids.size(0), -1)
        wt_plast, wt_pqkv = self.model.encode_primary(wt_b_ids, wt_b_attn)
        if use_partner and d_last_1 is not None and d_attn_1 is not None:
            wt_out = self.model.forward_from_hidden(
                wt_plast,
                wt_b_attn,
                d_last_1.expand(wt_b_ids.size(0), -1, -1),
                d_attn_1.expand(wt_b_ids.size(0), -1),
                primary_adapter_input=wt_pqkv,
            )
        else:
            wt_out = self.model.forward_from_hidden(wt_plast, wt_b_attn, primary_adapter_input=wt_pqkv)

        # Map each valid token index (excluding special tokens such as paired `|` separators)
        # back to its character index in `seed_sequence` so paired `VH|VL` / `CDR3b|CDR3a`
        # designs never shift across or overwrite the `|` delimiter.
        if len(seed_sequence) == len(valid_tok_pos):
            char_pos_by_valid = list(range(len(seed_sequence)))
        else:
            non_sep = [i for i, ch in enumerate(seed_sequence) if ch != "|"]
            char_pos_by_valid = non_sep if len(non_sep) == len(valid_tok_pos) else list(range(len(valid_tok_pos)))

        r_ar = torch.arange(len(design_pos), device=device)
        d_tok_t = torch.tensor([valid_tok_pos[p] for p in design_pos], dtype=torch.long, device=device)
        wt_holo_lp = F.log_softmax(wt_out.primary_holo_logits[r_ar, d_tok_t][:, vocab_ids].float(), dim=-1)
        wt_apo_lp = F.log_softmax(wt_out.primary_apo_logits[r_ar, d_tok_t][:, vocab_ids].float(), dim=-1)
        wt_pot = (wt_holo_lp + guided_beta * (wt_holo_lp - wt_apo_lp)).cpu().numpy()
        wt_ref_by_site = {
            p0: float(wt_pot[idx, aa_to_col.get(seed_sequence[char_pos_by_valid[p0]], 0)])
            for idx, p0 in enumerate(design_pos)
        }

        chains_chars = [list(seed_sequence) for _ in range(num_chains)]
        chain_ids = wt_ids.unsqueeze(0).repeat(num_chains, 1)
        chain_attn = base_attn.expand(num_chains, -1)

        trajectories = np.zeros((num_chains, num_steps + 1), dtype=np.float64)
        cur_scores = np.zeros(num_chains, dtype=np.float64)

        for step in range(num_steps):
            chosen_sites = [design_pos[int(i)] for i in rng.integers(0, len(design_pos), size=num_chains)]
            chosen_tok_pos = [valid_tok_pos[p0] for p0 in chosen_sites]

            masked_batch = chain_ids.clone()
            for c_i, t_pos in enumerate(chosen_tok_pos):
                masked_batch[c_i, t_pos] = mask_id

            p_last, p_qkv = self.model.encode_primary(masked_batch, chain_attn)
            if use_partner and d_last_1 is not None and d_attn_1 is not None:
                out = self.model.forward_from_hidden(
                    p_last,
                    chain_attn,
                    d_last_1.expand(num_chains, -1, -1),
                    d_attn_1.expand(num_chains, -1),
                    primary_adapter_input=p_qkv,
                )
            else:
                out = self.model.forward_from_hidden(p_last, chain_attn, primary_adapter_input=p_qkv)

            c_ar = torch.arange(num_chains, device=device)
            t_pos_tensor = torch.tensor(chosen_tok_pos, dtype=torch.long, device=device)
            h_lp = F.log_softmax(out.primary_holo_logits[c_ar, t_pos_tensor][:, vocab_ids].float(), dim=-1)
            a_lp = F.log_softmax(out.primary_apo_logits[c_ar, t_pos_tensor][:, vocab_ids].float(), dim=-1)
            pot = (h_lp + guided_beta * (h_lp - a_lp)).cpu().numpy()  # (num_chains, |V|)

            scaled = pot / temperature
            scaled -= scaled.max(axis=-1, keepdims=True)
            probs = np.exp(scaled)
            probs /= probs.sum(axis=-1, keepdims=True)

            for c_i in range(num_chains):
                p0 = chosen_sites[c_i]
                ch_pos = char_pos_by_valid[p0]
                t_pos = chosen_tok_pos[c_i]
                cur_aa = chains_chars[c_i][ch_pos]
                cur_col = aa_to_col.get(cur_aa, 0)
                prop_col = int(rng.choice(len(alpha_tuple), p=probs[c_i]))
                prop_aa = alpha_tuple[prop_col]

                delta_pot = float(pot[c_i, prop_col] - pot[c_i, cur_col])
                accept = True
                if metropolis_accept and delta_pot < 0.0:
                    accept = float(rng.random()) < float(np.exp(np.clip(delta_pot / temperature, -30.0, 0.0)))

                if accept and prop_aa != cur_aa:
                    old_contrib = float(pot[c_i, cur_col]) - wt_ref_by_site[p0]
                    new_contrib = float(pot[c_i, prop_col]) - wt_ref_by_site[p0]
                    cur_scores[c_i] += (new_contrib - old_contrib) / max(1, len(design_pos))
                    chains_chars[c_i][ch_pos] = prop_aa
                    chain_ids[c_i, t_pos] = vocab_ids[prop_col]

            trajectories[:, step + 1] = cur_scores

        final_seqs = ["".join(ch) for ch in chains_chars]
        # Inverse Simpson effective sample diversity N_eff = 1 / sum_u p_u^2
        counts: dict[str, int] = {}
        for s in final_seqs:
            counts[s] = counts.get(s, 0) + 1
        freqs = np.array([v / len(final_seqs) for v in counts.values()], dtype=np.float64)
        n_eff = float(1.0 / np.sum(freqs**2)) if freqs.size > 0 else 1.0

        if was_training:
            self.model.train()

        return GibbsDesignResult(
            seed_sequence=seed_sequence,
            context=context,
            design_positions_0based=design_pos,
            final_sequences=final_seqs,
            final_scores=cur_scores,
            score_trajectories=trajectories,
            site_probability_matrix=init_pwm,
            alphabet=alpha_tuple,
            effective_sample_size=n_eff,
        )


def evaluate_landscape_gibbs(
    experimental_fitness: Sequence[float] | np.ndarray,
    model_scores: Sequence[float] | np.ndarray,
    site_to_variant_indices: dict[int, Sequence[int] | np.ndarray] | None = None,
    *,
    site_tau: float = 0.3,
    boltzmann_tau: float = 0.25,
    num_chains: int = 200,
    num_steps: int = 12,
    seed: int = 42,
) -> dict[str, object]:
    """Evaluates retrospective Gibbs design across an experimental DMS landscape.

    Implements all three protocols from the LogiCA paper:
      1. ``within_site_gibbs``: Exact conditional amino-acid sampling within each mutated site;
      2. ``boltzmann_gibbs``: Global Boltzmann distribution ``p(x|y) propto exp(s(x,y)/tau)``;
      3. ``coordinate_mcmc_gibbs``: ``num_steps``-step Metropolis-within-Gibbs trajectories.
    """
    y = np.asarray(experimental_fitness, dtype=np.float64)
    s = np.asarray(model_scores, dtype=np.float64)
    y_std = float(np.std(y))
    y_z = (y - float(np.mean(y))) / (y_std if y_std > 1e-9 else 1.0)
    s_std = float(np.std(s))
    s_z = (s - float(np.mean(s))) / (s_std if s_std > 1e-9 else 1.0)
    is_top10 = (y >= float(np.quantile(y, 0.90))).astype(np.float64)
    base_top10 = max(float(np.mean(is_top10)), 1e-8)

    def _softmax(vec: np.ndarray, tau: float) -> np.ndarray:
        v = vec / max(tau, 1e-6)
        v = v - np.max(v)
        e = np.exp(v)
        return e / np.sum(e)

    # Protocol 1: Within-Site Coordinate Gibbs
    if site_to_variant_indices:
        site_z, site_enr = [], []
        for idx_list in site_to_variant_indices.values():
            arr = np.asarray(idx_list, dtype=np.int64)
            if arr.size < 2:
                continue
            p = _softmax(s_z[arr], site_tau)
            site_z.append(float(np.sum(p * y_z[arr])))
            site_enr.append(float(np.sum(p * is_top10[arr]) / base_top10))
        within_site_z = float(np.mean(site_z)) if site_z else 0.0
        within_site_top10 = float(np.mean(site_enr)) if site_enr else 1.0
    else:
        p_all = _softmax(s_z, site_tau)
        within_site_z = float(np.sum(p_all * y_z))
        within_site_top10 = float(np.sum(p_all * is_top10) / base_top10)

    # Protocol 2: Global Boltzmann-Gibbs
    p_boltz = _softmax(s_z, boltzmann_tau)
    boltz_z = float(np.sum(p_boltz * y_z))
    boltz_top10 = float(np.sum(p_boltz * is_top10) / base_top10)
    n_eff = float(1.0 / np.sum(p_boltz**2))

    # Protocol 3: Coordinate MCMC Gibbs Trajectories
    rng = np.random.default_rng(seed)
    n = len(y_z)
    cur = rng.integers(0, n, size=num_chains)
    traj_z = [float(np.mean(y_z[cur]))]
    traj_top10 = [float(np.mean(is_top10[cur]) / base_top10)]

    groups = (
        [np.asarray(v, dtype=np.int64) for v in site_to_variant_indices.values() if len(v) >= 2]
        if site_to_variant_indices
        else []
    )
    for _ in range(num_steps):
        if groups:
            g_choices = rng.integers(0, len(groups), size=num_chains)
            for c_i in range(num_chains):
                g = groups[int(g_choices[c_i])]
                p_g = _softmax(s_z[g], site_tau)
                prop = int(g[int(rng.choice(len(g), p=p_g))])
                delta = s_z[prop] - s_z[cur[c_i]]
                if delta >= 0.0 or float(rng.random()) < float(np.exp(np.clip(delta / site_tau, -25.0, 0.0))):
                    cur[c_i] = prop
        else:
            props = rng.choice(n, size=num_chains, p=p_boltz)
            deltas = s_z[props] - s_z[cur]
            acc = (deltas >= 0.0) | (rng.random(num_chains) < np.exp(np.clip(deltas / boltzmann_tau, -25.0, 0.0)))
            cur = np.where(acc, props, cur)

        traj_z.append(float(np.mean(y_z[cur])))
        traj_top10.append(float(np.mean(is_top10[cur]) / base_top10))

    return {
        "within_site_gibbs_z": within_site_z,
        "within_site_gibbs_top10_enrichment": within_site_top10,
        "boltzmann_gibbs_z": boltz_z,
        "boltzmann_gibbs_top10_enrichment": boltz_top10,
        "boltzmann_effective_diversity": n_eff,
        "mcmc_final_z": traj_z[-1],
        "mcmc_final_top10_enrichment": traj_top10[-1],
        "mcmc_trajectory_z": traj_z,
        "mcmc_trajectory_top10_enrichment": traj_top10,
    }
