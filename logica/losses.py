"""Logit-Space Contrastive Alignment & Preference Fine-Tuning Losses.

Implements the training objectives of LogiCA across both Dual-BioFM and Single-BioFM modes:
  - Eq. 1 & 4: :class:`InfoNCELogitLoss` — Bidirectional or single-anchor InfoNCE over
    site-averaged native conditional log-likelihoods (with amortized ``B x K`` cross-attention
    scoring and zero-WT-collision local mutation negative constructors).
  - Eq. 1 (|C|=2) & Eq. 3: :class:`BradleyTerryLogitLoss` and :class:`PlackettLuceLogitLoss` —
    Pairwise and listwise variant preference alignment in native logit space.
  - :class:`AncestralKLAnchorLoss` — Trust-region KL regularization anchoring adapted BioFM
    distributions to their frozen pretrained reference.
  - :class:`ThermodynamicHeadlessScorer` — Headless viability + context-shift parameterization
    for drug-resistance vs. agonist affinity adaptation without MLP heads.
"""

from __future__ import annotations

import random
from typing import Literal, Sequence

import torch
from torch import nn
import torch.nn.functional as F

from .backbones import CANONICAL_AAS
from .model import LogiCA

# Non-conservative amino-acid substitution groups (BLOSUM62 < 0)
NON_CONSERVATIVE_AA_MAP: dict[str, str] = {
    "A": "DEKRWFY",
    "C": "DEKRNPQ",
    "D": "AILMFWV",
    "E": "AILMFWV",
    "F": "DEKRNPG",
    "G": "FILMWVY",
    "H": "AILMVWF",
    "I": "DEKRNQP",
    "K": "AILMFWV",
    "L": "DEKRNQP",
    "M": "DEKRNP",
    "N": "FILMVW",
    "P": "FILMWVY",
    "Q": "FILMVW",
    "R": "AILMFWV",
    "S": "FILMWVY",
    "T": "FILMWVY",
    "V": "DEKRNQP",
    "W": "DEKRNPGS",
    "Y": "DEKRNPGA",
}


def make_masked_inputs(
    input_ids: torch.Tensor,
    attention_mask: torch.Tensor,
    special_tokens_mask: torch.Tensor,
    mask_token_id: int,
    mask_fraction: float = 0.15,
    generator: torch.Generator | None = None,
    forced_mask_positions: torch.Tensor | None = None,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Masks ``mask_fraction`` (guaranteed ``>= 1`` token) of non-special positions per row."""
    if not 0.0 < mask_fraction <= 1.0:
        raise ValueError(f"mask_fraction must lie in (0, 1]; got {mask_fraction}")
    masked = input_ids.clone()
    selected = torch.zeros_like(input_ids, dtype=torch.bool)

    for row in range(input_ids.shape[0]):
        valid = (attention_mask[row] == 1) & (~special_tokens_mask[row].bool())
        candidates = torch.nonzero(valid, as_tuple=False).flatten()
        if candidates.numel() == 0:
            raise ValueError(f"Row {row} has zero scoreable non-special tokens.")
        target_count = max(1, int(round(candidates.numel() * mask_fraction)))

        if forced_mask_positions is not None:
            forced_valid = torch.nonzero(
                valid & forced_mask_positions[row].bool(), as_tuple=False
            ).flatten()
        else:
            forced_valid = candidates.new_empty((0,))

        if forced_valid.numel() > 0:
            selected[row, forced_valid] = True
            masked[row, forced_valid] = int(mask_token_id)

        rem = max(0, target_count - int(forced_valid.numel()))
        if rem > 0:
            rem_cands = torch.nonzero(valid & (~selected[row]), as_tuple=False).flatten()
            if rem_cands.numel() > 0:
                perm = torch.randperm(
                    rem_cands.numel(), generator=generator, device=rem_cands.device
                )
                chosen = rem_cands[perm[:rem]]
                selected[row, chosen] = True
                masked[row, chosen] = int(mask_token_id)

    return masked, selected


def mean_selected_log_probability(
    logits: torch.Tensor,
    original_ids: torch.Tensor,
    selected_mask: torch.Tensor,
) -> torch.Tensor:
    """Computes ``(1 / |A|) * sum_{i in A} log pi(x_i | x_{\\A}, y)`` per batch row (Eq. 2)."""
    token_lp = (
        F.log_softmax(logits.float(), dim=-1)
        .gather(-1, original_ids.unsqueeze(-1).clamp(min=0))
        .squeeze(-1)
    )
    sel_f = selected_mask.to(token_lp.dtype)
    denom = sel_f.sum(dim=1)
    avg = (token_lp * sel_f).sum(dim=1) / denom.clamp_min(1.0)
    return torch.where(denom > 0, avg, torch.full_like(avg, -1e4))


def create_anchored_mutation_negatives(
    sequence: str,
    num_negatives: int = 5,
    *,
    num_mutations: int = 1,
    strategy: Literal["uniform", "non_conservative", "central_contact"] = "non_conservative",
    alphabet: Sequence[str] = CANONICAL_AAS,
    rng: random.Random | None = None,
) -> list[str]:
    """Generates ``num_negatives`` local sequence negatives with strictly ``>= 1`` substitution.

    Guarantees ``0%`` exact wild-type collisions even on short sequences (e.g. 9-mer epitopes).
    """
    gen = rng or random
    chars = list(sequence.strip().upper())
    valid_indices = [i for i, ch in enumerate(chars) if ch in alphabet and ch != "|"]
    if not valid_indices:
        raise ValueError(f"Sequence {sequence!r} has no mutable canonical residues.")

    if strategy == "central_contact" and len(valid_indices) >= 7:
        # Focus perturbations on central contact core (e.g. P4-P8 on 9-mers or CDR3 apex)
        lo = max(1, len(valid_indices) // 4)
        hi = max(lo + 1, len(valid_indices) - max(1, len(valid_indices) // 5))
        pool_indices = valid_indices[lo:hi]
    else:
        pool_indices = valid_indices

    k_eff = max(1, min(num_mutations, len(pool_indices)))
    out: list[str] = []
    for _ in range(num_negatives):
        mut_chars = list(chars)
        chosen_pos = gen.sample(pool_indices, k=k_eff)
        for pos in chosen_pos:
            orig = mut_chars[pos]
            if strategy in ("non_conservative", "central_contact") and orig in NON_CONSERVATIVE_AA_MAP:
                cand_aas = list(NON_CONSERVATIVE_AA_MAP[orig])
            else:
                cand_aas = [a for a in alphabet if a != orig]
            mut_chars[pos] = gen.choice(cand_aas)
        out.append("".join(mut_chars))
    return out


class InfoNCELogitLoss(nn.Module):
    """Bidirectional or single-anchor InfoNCE in native BioFM logit space (Eq. 1 & Eq. 4).

    Encodes each unique sequence in the batch ONCE through the heavy BioFM backbones, then
    evaluates all ``B x (1 + K)`` positive and negative pairs exclusively through the lightweight
    cross-attention adapter and native ``lm_head``.
    """

    def __init__(
        self,
        temperature: float = 0.07,
        alpha: float = 0.5,
        mask_fraction: float = 0.15,
        inbatch_negatives_per_anchor: int = 15,
        anchor_mode: Literal["bidirectional", "primary", "partner"] = "bidirectional",
        subtract_apo_baseline: bool = True,
        mlm_aux_weight: float = 0.0,
        adapter_chunk_size: int = 128,
    ) -> None:
        super().__init__()
        if temperature <= 0.0:
            raise ValueError("temperature must be > 0")
        self.temperature = temperature
        self.alpha = alpha
        self.mask_fraction = mask_fraction
        self.inbatch_negatives_per_anchor = inbatch_negatives_per_anchor
        self.anchor_mode = anchor_mode
        self.subtract_apo_baseline = subtract_apo_baseline
        self.mlm_aux_weight = mlm_aux_weight
        self.adapter_chunk_size = adapter_chunk_size

    def _score_indexed_pairs(
        self,
        model: LogiCA,
        p_last: torch.Tensor,
        p_qkv: torch.Tensor,
        p_attn: torch.Tensor,
        p_orig_ids: torch.Tensor,
        p_sel: torch.Tensor,
        d_last: torch.Tensor,
        d_attn: torch.Tensor,
        d_orig_ids: torch.Tensor,
        d_sel: torch.Tensor,
        p_idx: torch.Tensor,
        d_idx: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        with torch.no_grad():
            if self.subtract_apo_baseline:
                p_apo_ll = mean_selected_log_probability(
                    model.primary_backbone.logits_from_hidden(p_last), p_orig_ids, p_sel
                )
                assert model.partner_backbone is not None
                d_apo_ll = mean_selected_log_probability(
                    model.partner_backbone.logits_from_hidden(d_last), d_orig_ids, d_sel
                )
            else:
                p_apo_ll = torch.zeros(p_last.size(0), device=p_last.device)
                d_apo_ll = torch.zeros(d_last.size(0), device=d_last.device)

        scores: list[torch.Tensor] = []
        pos_mlm_terms: list[torch.Tensor] = []
        n_pairs = int(p_idx.numel())
        for st in range(0, n_pairs, self.adapter_chunk_size):
            en = min(st + self.adapter_chunk_size, n_pairs)
            pi = p_idx[st:en]
            di = d_idx[st:en]
            out = model.forward_from_hidden(
                p_last[pi],
                p_attn[pi],
                d_last[di],
                d_attn[di],
                primary_adapter_input=p_qkv[pi],
            )
            assert out.partner_holo_logits is not None
            p_ll = mean_selected_log_probability(out.primary_holo_logits, p_orig_ids[pi], p_sel[pi])
            d_ll = mean_selected_log_probability(out.partner_holo_logits, d_orig_ids[di], d_sel[di])
            pair_s = self.alpha * (p_ll - p_apo_ll[pi]) + (1.0 - self.alpha) * (d_ll - d_apo_ll[di])
            scores.append(pair_s)
            if self.mlm_aux_weight > 0.0:
                pos_mlm_terms.append(-0.5 * (p_ll + d_ll))

        all_scores = torch.cat(scores, dim=0)
        aux_mlm = torch.cat(pos_mlm_terms, dim=0) if pos_mlm_terms else all_scores.new_zeros((n_pairs,))
        return all_scores, aux_mlm

    def forward(
        self,
        model: LogiCA,
        primary_batch: dict[str, torch.Tensor],
        partner_batch: dict[str, torch.Tensor],
        *,
        hard_neg_partner_batch: dict[str, torch.Tensor] | None = None,
        hard_neg_primary_batch: dict[str, torch.Tensor] | None = None,
        false_negative_mask: torch.Tensor | None = None,
    ) -> dict[str, torch.Tensor]:
        if not model.is_dual or model.partner_backbone is None:
            raise ValueError("InfoNCELogitLoss requires a dual-BioFM LogiCA model.")

        p_ids = primary_batch["input_ids"]
        p_attn = primary_batch["attention_mask"]
        p_spec = primary_batch["special_tokens_mask"]
        d_ids = partner_batch["input_ids"]
        d_attn = partner_batch["attention_mask"]
        d_spec = partner_batch["special_tokens_mask"]
        bsz = p_ids.size(0)
        device = p_ids.device

        all_p_ids = (
            torch.cat([p_ids, hard_neg_primary_batch["input_ids"]], dim=0)
            if hard_neg_primary_batch is not None
            else p_ids
        )
        all_p_attn = (
            torch.cat([p_attn, hard_neg_primary_batch["attention_mask"]], dim=0)
            if hard_neg_primary_batch is not None
            else p_attn
        )
        all_p_spec = (
            torch.cat([p_spec, hard_neg_primary_batch["special_tokens_mask"]], dim=0)
            if hard_neg_primary_batch is not None
            else p_spec
        )

        all_d_ids = (
            torch.cat([d_ids, hard_neg_partner_batch["input_ids"]], dim=0)
            if hard_neg_partner_batch is not None
            else d_ids
        )
        all_d_attn = (
            torch.cat([d_attn, hard_neg_partner_batch["attention_mask"]], dim=0)
            if hard_neg_partner_batch is not None
            else d_attn
        )
        all_d_spec = (
            torch.cat([d_spec, hard_neg_partner_batch["special_tokens_mask"]], dim=0)
            if hard_neg_partner_batch is not None
            else d_spec
        )

        p_masked, p_sel = make_masked_inputs(
            all_p_ids, all_p_attn, all_p_spec, model.primary_backbone.mask_token_id, self.mask_fraction
        )
        d_masked, d_sel = make_masked_inputs(
            all_d_ids, all_d_attn, all_d_spec, model.partner_backbone.mask_token_id, self.mask_fraction
        )

        p_last, p_qkv = model.encode_primary(p_masked, all_p_attn)
        d_last = model.encode_partner(d_masked, all_d_attn)

        k_neg = min(self.inbatch_negatives_per_anchor, max(0, bsz - 1))
        has_hard_d = hard_neg_partner_batch is not None
        has_hard_p = hard_neg_primary_batch is not None
        if k_neg == 0 and not has_hard_d and not has_hard_p:
            raise ValueError("Batch size >= 2 or hard negatives required for InfoNCELogitLoss.")

        p2d_pi, p2d_di, d2p_pi, d2p_di = [], [], [], []
        p2d_fn, d2p_fn = [], []
        fn_grid = (
            false_negative_mask.detach().cpu().bool().tolist()
            if false_negative_mask is not None
            else None
        )

        for b in range(bsz):
            others = [j for j in range(bsz) if j != b]
            if len(others) > k_neg:
                perm = torch.randperm(len(others)).tolist()[:k_neg]
                neg_js = [others[i] for i in perm]
            else:
                neg_js = others

            cand_d = [b] + neg_js + ([bsz + b] if has_hard_d else [])
            p2d_pi.extend([b] * len(cand_d))
            p2d_di.extend(cand_d)

            cand_p = [b] + neg_js + ([bsz + b] if has_hard_p else [])
            d2p_pi.extend(cand_p)
            d2p_di.extend([b] * len(cand_p))

            if fn_grid is not None:
                p2d_fn.append([False] + [bool(fn_grid[b][j]) for j in neg_js] + ([False] if has_hard_d else []))
                d2p_fn.append([False] + [bool(fn_grid[j][b]) for j in neg_js] + ([False] if has_hard_p else []))

        n_cols_p2d = 1 + k_neg + (1 if has_hard_d else 0)
        n_cols_d2p = 1 + k_neg + (1 if has_hard_p else 0)
        targets = torch.zeros(bsz, dtype=torch.long, device=device)

        losses: list[torch.Tensor] = []
        accs: list[torch.Tensor] = []
        aux_terms: list[torch.Tensor] = []

        if self.anchor_mode in ("bidirectional", "primary") and n_cols_p2d > 1:
            sc_p2d, aux_p2d = self._score_indexed_pairs(
                model, p_last, p_qkv, all_p_attn, all_p_ids, p_sel,
                d_last, all_d_attn, all_d_ids, d_sel,
                torch.tensor(p2d_pi, dtype=torch.long, device=device),
                torch.tensor(p2d_di, dtype=torch.long, device=device),
            )
            lg_p2d = sc_p2d.view(bsz, n_cols_p2d) / self.temperature
            if fn_grid is not None:
                lg_p2d = lg_p2d.masked_fill(torch.tensor(p2d_fn, dtype=torch.bool, device=device), -1e4)
            losses.append(F.cross_entropy(lg_p2d, targets))
            accs.append((lg_p2d.argmax(dim=-1) == 0).float().mean())
            if self.mlm_aux_weight > 0.0:
                aux_terms.append(aux_p2d.view(bsz, n_cols_p2d)[:, 0].mean())

        if self.anchor_mode in ("bidirectional", "partner") and n_cols_d2p > 1:
            sc_d2p, aux_d2p = self._score_indexed_pairs(
                model, p_last, p_qkv, all_p_attn, all_p_ids, p_sel,
                d_last, all_d_attn, all_d_ids, d_sel,
                torch.tensor(d2p_pi, dtype=torch.long, device=device),
                torch.tensor(d2p_di, dtype=torch.long, device=device),
            )
            lg_d2p = sc_d2p.view(bsz, n_cols_d2p) / self.temperature
            if fn_grid is not None:
                lg_d2p = lg_d2p.masked_fill(torch.tensor(d2p_fn, dtype=torch.bool, device=device), -1e4)
            losses.append(F.cross_entropy(lg_d2p, targets))
            accs.append((lg_d2p.argmax(dim=-1) == 0).float().mean())
            if self.mlm_aux_weight > 0.0:
                aux_terms.append(aux_d2p.view(bsz, n_cols_d2p)[:, 0].mean())

        contrastive_loss = torch.stack(losses).mean()
        mlm_loss = torch.stack(aux_terms).mean() if aux_terms else contrastive_loss * 0.0
        total_loss = contrastive_loss + self.mlm_aux_weight * mlm_loss
        return {
            "loss": total_loss,
            "contrastive_loss": contrastive_loss.detach(),
            "mlm_loss": mlm_loss.detach(),
            "top1_accuracy": torch.stack(accs).mean().detach(),
        }


class BradleyTerryLogitLoss(nn.Module):
    """Pairwise Bradley–Terry preference loss over native mutation scores (Eq. 1 with |C|=2)."""

    def __init__(self, temperature: float = 1.0, min_target_gap: float = 1e-4) -> None:
        super().__init__()
        self.temperature = temperature
        self.min_target_gap = min_target_gap

    def forward(
        self,
        scores: torch.Tensor,
        targets: torch.Tensor,
        max_pairs: int = 512,
    ) -> torch.Tensor:
        n = scores.numel()
        if n < 2:
            return scores.sum() * 0.0
        diff_y = targets[:, None] - targets[None, :]
        pos_i, neg_j = torch.nonzero(diff_y > self.min_target_gap, as_tuple=True)
        if pos_i.numel() == 0:
            return scores.sum() * 0.0
        if pos_i.numel() > max_pairs:
            perm = torch.randperm(pos_i.numel(), device=scores.device)[:max_pairs]
            pos_i, neg_j = pos_i[perm], neg_j[perm]
        margins = (scores[pos_i] - scores[neg_j]) / self.temperature
        weights = diff_y[pos_i, neg_j].detach()
        weights = weights / weights.mean().clamp_min(1e-8)
        return (weights * F.softplus(-margins)).mean()


class PlackettLuceLogitLoss(nn.Module):
    """Listwise Plackett–Luce ranking loss emphasizing top-M functional variants."""

    def __init__(self, top_m: int = 32, temperature: float = 1.0) -> None:
        super().__init__()
        self.top_m = top_m
        self.temperature = temperature

    def forward(self, scores: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        order = torch.argsort(targets, descending=True)
        s_ord = scores[order] / self.temperature
        m = min(self.top_m, int(s_ord.numel()) - 1)
        if m <= 0:
            return scores.sum() * 0.0
        log_denom = torch.logcumsumexp(s_ord.flip(0), dim=0).flip(0)
        return torch.mean(log_denom[:m] - s_ord[:m])


class AncestralKLAnchorLoss(nn.Module):
    """KL trust-region anchor ``D_KL(pi_theta || pi_ref)`` preventing evolutionary prior drift."""

    def forward(
        self,
        adapted_log_probs: torch.Tensor,
        reference_log_probs: torch.Tensor,
        valid_mask: torch.Tensor | None = None,
    ) -> torch.Tensor:
        if valid_mask is not None:
            adapted_log_probs = adapted_log_probs[valid_mask]
            reference_log_probs = reference_log_probs[valid_mask]
        if adapted_log_probs.numel() == 0:
            return adapted_log_probs.sum() * 0.0
        return F.kl_div(
            adapted_log_probs,
            reference_log_probs.detach(),
            log_target=True,
            reduction="batchmean",
        )


class ThermodynamicHeadlessScorer(nn.Module):
    """Headless 3-scalar thermodynamic decomposition over native 20-AA log-probabilities.

    Separates folding/catalytic viability (``delta_ll_apo``) from pure partner-induced log-odds
    shift (``dd_ll_context``) so inhibitor-escape and agonist-binding assays share a physically
    consistent token-probability interface without MLP heads.
    """

    def __init__(
        self,
        init_w_viab: float = 1.0,
        init_w_context: float = 1.5,
        init_tau_fold: float = -2.5,
        signed_context: bool = False,
    ) -> None:
        super().__init__()
        self.signed_context = signed_context
        self.raw_w_viab = nn.Parameter(torch.tensor(float(torch.log(torch.expm1(torch.tensor(init_w_viab))))))
        if signed_context:
            self.raw_w_context = nn.Parameter(torch.tensor(float(init_w_context)))
        else:
            self.raw_w_context = nn.Parameter(
                torch.tensor(float(torch.log(torch.expm1(torch.tensor(init_w_context)))))
            )
        self.tau_fold = nn.Parameter(torch.tensor(float(init_tau_fold)))

    @property
    def w_viab(self) -> torch.Tensor:
        return F.softplus(self.raw_w_viab)

    @property
    def w_context(self) -> torch.Tensor:
        return self.raw_w_context if self.signed_context else F.softplus(self.raw_w_context)

    def forward(
        self,
        delta_ll_apo: torch.Tensor,
        dd_ll_context: torch.Tensor,
        assay_polarity: float | torch.Tensor = -1.0,
    ) -> torch.Tensor:
        """Scores variants; ``assay_polarity=-1.0`` rewards tighter binding, ``+1.0`` rewards drug escape."""
        viab = -self.w_viab * F.softplus(self.tau_fold - delta_ll_apo)
        ctx = -assay_polarity * self.w_context * dd_ll_context
        return viab + ctx
