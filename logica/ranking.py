"""Mutation-Local Variant Ranking in Native Logit Space (Section 2.2 & Proposition 1).

Implements Eq. 3 of the LogiCA paper:
  s_M(x, y; x^{wt}) = (1 / |M|) * sum_{i in M} log [ pi_theta(x_i | x_{\\i}, y)
                                                   / pi_theta(x_i^{wt} | x_{\\i}^{wt}, y) ]
across both Dual-BioFM (partner-conditioned) and Single-BioFM (``context=None``) models,
supporting:
  - Fast single-pass wild-type marginal scoring (``mode="wt_marginal"``),
  - Simultaneous multi-site masked scoring (``mode="masked_sites"``),
  - Stride-S masked marginal scoring (``mode="stride_masked"``),
  - Exact leave-one-out per-position masking (``mode="exact_loo"``),
  - Domain-centered sequence windowing for full-length receptors (>1022 AA),
  - Standard DMS evaluation metrics (Spearman rho, Pearson r, Top-k% Enrichment, Union Concordance).
"""

from __future__ import annotations

from dataclasses import dataclass
import re
from typing import Literal, Sequence

import numpy as np
from scipy import stats
import torch
import torch.nn.functional as F

from .backbones import CANONICAL_AAS
from .model import LogiCA

MUT_TOKEN_RE = re.compile(r"^([A-Za-z])(\d+)([A-Za-z])$")

ScoringMode = Literal["wt_marginal", "masked_sites", "stride_masked", "exact_loo"]
ReductionMode = Literal["mean", "sum"]
ScoreComponent = Literal["holo", "ddg_shift", "apo", "guided"]


@dataclass(frozen=True)
class MutationEdit:
    """Zero-based single-token substitution ``wt_token -> mut_token`` at ``pos_0based``."""

    pos_0based: int
    wt_token: str
    mut_token: str

    @property
    def label(self) -> str:
        return f"{self.wt_token}{self.pos_0based + 1}{self.mut_token}"


@dataclass
class VariantRankingResult:
    """Structured result of :meth:`VariantRanker.rank_variants`."""

    wild_type: str
    context: str | None
    variants: list[str]
    mutation_labels: list[str]
    scores: np.ndarray
    holo_scores: np.ndarray
    apo_scores: np.ndarray
    ddg_shift_scores: np.ndarray
    ranked_indices: np.ndarray

    def top_k(self, k: int = 10) -> list[tuple[int, str, float]]:
        """Returns ``[(rank_idx, mutation_label_or_seq, score), ...]`` for top ``k`` variants."""
        out: list[tuple[int, str, float]] = []
        for idx in self.ranked_indices[:k]:
            i = int(idx)
            lbl = self.mutation_labels[i] or self.variants[i]
            out.append((i, lbl, float(self.scores[i])))
        return out


def parse_mutation_string(mut_str: str) -> list[MutationEdit]:
    """Parses standard DMS mutation strings such as ``'T790M'``, ``'L858R_T790M'``, or ``'A42G:V80I'``."""
    s = str(mut_str).strip()
    if not s or s.upper() in ("WT", "WILDTYPE", "NAN"):
        return []
    edits: list[MutationEdit] = []
    for raw in re.split(r"[_,;/:\s]+", s):
        tok = raw.strip()
        if not tok or tok.upper() == "WT":
            continue
        m = MUT_TOKEN_RE.match(tok)
        if m is None:
            raise ValueError(f"Invalid mutation token {tok!r} in {mut_str!r}")
        w_aa, pos_1, m_aa = m.group(1).upper(), int(m.group(2)), m.group(3).upper()
        edits.append(MutationEdit(pos_0based=pos_1 - 1, wt_token=w_aa, mut_token=m_aa))
    return edits


def apply_mutations(wild_type: str, edits: Sequence[MutationEdit] | str) -> str:
    """Applies 1-based mutation string or :class:`MutationEdit` list to ``wild_type``."""
    parsed = parse_mutation_string(edits) if isinstance(edits, str) else list(edits)
    chars = list(wild_type)
    for ed in parsed:
        if not (0 <= ed.pos_0based < len(chars)):
            raise ValueError(f"Mutation {ed.label} out of range for sequence of length {len(chars)}")
        if chars[ed.pos_0based].upper() != ed.wt_token.upper():
            raise ValueError(
                f"WT mismatch at pos {ed.pos_0based + 1}: expected {ed.wt_token}, found {chars[ed.pos_0based]}"
            )
        chars[ed.pos_0based] = ed.mut_token
    return "".join(chars)


def diff_mutations(wild_type: str, variant: str) -> list[MutationEdit]:
    """Extracts substitution set ``M(x, x^{wt}) = {i : x_i != x_i^{wt}}`` without annotations."""
    if len(wild_type) != len(variant):
        raise ValueError(
            f"Length mismatch ({len(wild_type)} vs {len(variant)}); substitution scorer expects aligned lengths."
        )
    return [
        MutationEdit(pos_0based=i, wt_token=w, mut_token=m)
        for i, (w, m) in enumerate(zip(wild_type, variant))
        if w != m
    ]


def window_sequence_around_mutations(
    wild_type: str,
    edits_per_variant: Sequence[Sequence[MutationEdit]],
    max_len: int = 1022,
) -> tuple[str, list[list[MutationEdit]], int]:
    """Centers a ``<= max_len`` window around mutated positions and shifts coordinates."""
    if len(wild_type) <= max_len:
        return wild_type, [list(eds) for eds in edits_per_variant], 0

    all_pos = [ed.pos_0based for eds in edits_per_variant for ed in eds]
    center = int(round(0.5 * (min(all_pos) + max(all_pos)))) if all_pos else max_len // 2
    start = max(0, min(len(wild_type) - max_len, center - (max_len // 2)))
    wt_crop = wild_type[start : start + max_len]
    remapped: list[list[MutationEdit]] = []
    for eds in edits_per_variant:
        remapped.append(
            [
                MutationEdit(pos_0based=ed.pos_0based - start, wt_token=ed.wt_token, mut_token=ed.mut_token)
                for ed in eds
                if 0 <= (ed.pos_0based - start) < len(wt_crop)
            ]
        )
    return wt_crop, remapped, start


def top_k_enrichment(y_true: np.ndarray, y_score: np.ndarray, fraction: float = 0.01) -> float:
    """Computes Top-k% functional enrichment factor relative to random expectation (1.0x)."""
    n = len(y_true)
    if n == 0:
        return float("nan")
    k = max(1, int(round(fraction * n)))
    top_true = set(np.argsort(y_true)[-k:].tolist())
    top_pred = np.argsort(y_score)[-k:].tolist()
    hits = sum(1 for i in top_pred if i in top_true)
    return float((hits / k) / (k / n))


def evaluate_ranking_metrics(y_true: Sequence[float] | np.ndarray, y_score: Sequence[float] | np.ndarray) -> dict[str, float]:
    """Computes Spearman rho, Pearson r, and Top-1% / 5% / 10% enrichment factors."""
    yt = np.asarray(y_true, dtype=np.float64)
    ys = np.asarray(y_score, dtype=np.float64)
    mask = np.isfinite(yt) & np.isfinite(ys)
    yt, ys = yt[mask], ys[mask]
    if len(yt) < 3 or np.std(yt) < 1e-12 or np.std(ys) < 1e-12:
        return {
            "spearman_rho": 0.0,
            "pearson_r": 0.0,
            "prec_at_1pct": 1.0,
            "prec_at_5pct": 1.0,
            "prec_at_10pct": 1.0,
        }
    return {
        "spearman_rho": float(stats.spearmanr(yt, ys).statistic),
        "pearson_r": float(stats.pearsonr(yt, ys).statistic),
        "prec_at_1pct": top_k_enrichment(yt, ys, 0.01),
        "prec_at_5pct": top_k_enrichment(yt, ys, 0.05),
        "prec_at_10pct": top_k_enrichment(yt, ys, 0.10),
    }


class VariantRanker:
    """Model-agnostic mutation-local variant scorer & ranker on top of :class:`LogiCA`."""

    def __init__(self, model: LogiCA) -> None:
        self.model = model

    def _non_special_token_positions(self, wt_batch: dict[str, torch.Tensor]) -> list[int]:
        valid = (wt_batch["attention_mask"][0] == 1) & (~wt_batch["special_tokens_mask"][0].bool())
        return torch.nonzero(valid, as_tuple=False).flatten().tolist()

    def score_cohort_differentiable(
        self,
        wild_type: str,
        edits_list: Sequence[Sequence[MutationEdit]],
        context: str | None = None,
        *,
        alphabet: Sequence[str] = CANONICAL_AAS,
        reduction: ReductionMode = "mean",
        max_primary_len: int = 1022,
        max_partner_len: int = 256,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Differentiable 1-pass WT-marginal scorer for end-to-end LoRA / adapter training.

        Returns ``(holo_variant_scores, holo_site_log_probs_vocab, apo_site_log_probs_vocab)``.
        """
        device = self.model.device
        wt_win, mapped_edits, _ = window_sequence_around_mutations(
            wild_type, edits_list, max_len=max_primary_len
        )
        p_enc = self.model.primary_backbone.tokenize(wt_win, max_length=max_primary_len + 2, device=device)
        tok_positions = self._non_special_token_positions(p_enc)

        if self.model.is_dual and context is not None and self.model.partner_backbone is not None:
            d_enc = self.model.partner_backbone.tokenize(context, max_length=max_partner_len, device=device)
            out = self.model(
                p_enc["input_ids"],
                p_enc["attention_mask"],
                d_enc["input_ids"],
                d_enc["attention_mask"],
            )
        else:
            out = self.model(p_enc["input_ids"], p_enc["attention_mask"])

        vocab_ids = torch.tensor(
            [self.model.primary_backbone.token_to_id(a) for a in alphabet],
            dtype=torch.long,
            device=device,
        )
        tok_to_col = {a: idx for idx, a in enumerate(alphabet)}

        holo_lp = F.log_softmax(out.primary_holo_logits[0][:, vocab_ids].float(), dim=-1)
        apo_lp = F.log_softmax(out.primary_apo_logits[0][:, vocab_ids].float(), dim=-1)

        scores: list[torch.Tensor] = []
        for eds in mapped_edits:
            if not eds:
                scores.append(holo_lp.sum() * 0.0)
                continue
            deltas = [
                holo_lp[tok_positions[ed.pos_0based], tok_to_col[ed.mut_token]]
                - holo_lp[tok_positions[ed.pos_0based], tok_to_col[ed.wt_token]]
                for ed in eds
            ]
            stacked = torch.stack(deltas)
            scores.append(stacked.mean() if reduction == "mean" else stacked.sum())

        valid_tok_t = torch.tensor(tok_positions, dtype=torch.long, device=device)
        return torch.stack(scores), holo_lp[valid_tok_t], apo_lp[valid_tok_t]

    @torch.no_grad()
    def rank_variants(
        self,
        wild_type: str,
        variants: Sequence[str],
        context: str | None = None,
        *,
        variants_are_mutation_strings: bool = False,
        mode: ScoringMode = "stride_masked",
        stride: int = 16,
        reduction: ReductionMode = "mean",
        component: ScoreComponent = "holo",
        guided_beta: float = 0.5,
        alphabet: Sequence[str] = CANONICAL_AAS,
        max_primary_len: int = 1022,
        max_partner_len: int = 256,
        batch_size: int = 64,
    ) -> VariantRankingResult:
        """Scores and ranks mutant sequences (or mutation strings like ``'T790M'``) under ``context``.

        Works seamlessly in both Dual-BioFM mode (``context`` provided) and Single-BioFM mode
        (``context=None``).
        """
        was_training = self.model.training
        self.model.eval()
        device = self.model.device

        edits_list: list[list[MutationEdit]] = []
        full_sequences: list[str] = []
        mut_labels: list[str] = []

        for item in variants:
            if variants_are_mutation_strings or (
                len(item) < len(wild_type) and ("_" in item or any(ch.isdigit() for ch in item))
            ):
                eds = parse_mutation_string(item)
                seq = apply_mutations(wild_type, eds)
            else:
                seq = str(item).strip()
                eds = diff_mutations(wild_type, seq)
            edits_list.append(eds)
            full_sequences.append(seq)
            mut_labels.append("_".join(e.label for e in eds) if eds else "WT")

        wt_win, mapped_edits, _ = window_sequence_around_mutations(
            wild_type, edits_list, max_len=max_primary_len
        )
        p_enc = self.model.primary_backbone.tokenize(wt_win, max_length=max_primary_len + 2, device=device)
        base_ids = p_enc["input_ids"][0]
        base_attn = p_enc["attention_mask"][0]
        tok_positions = self._non_special_token_positions(p_enc)
        mask_id = self.model.primary_backbone.mask_token_id

        vocab_ids = torch.tensor(
            [self.model.primary_backbone.token_to_id(a) for a in alphabet],
            dtype=torch.long,
            device=device,
        )
        tok_to_col = {a: i for i, a in enumerate(alphabet)}

        use_partner = self.model.is_dual and context is not None and self.model.partner_backbone is not None
        if use_partner:
            assert self.model.partner_backbone is not None
            d_enc = self.model.partner_backbone.tokenize(context, max_length=max_partner_len, device=device)
            d_last_1 = self.model.encode_partner(d_enc["input_ids"], d_enc["attention_mask"])
            d_attn_1 = d_enc["attention_mask"]
        else:
            d_last_1 = None
            d_attn_1 = None

        def _eval_primary_batch(batch_ids: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
            holo_chunks, apo_chunks = [], []
            for st in range(0, batch_ids.size(0), batch_size):
                sub_ids = batch_ids[st : st + batch_size]
                sub_attn = base_attn.unsqueeze(0).expand(sub_ids.size(0), -1)
                p_last, p_qkv = self.model.encode_primary(sub_ids, sub_attn)
                if use_partner and d_last_1 is not None and d_attn_1 is not None:
                    out = self.model.forward_from_hidden(
                        p_last,
                        sub_attn,
                        d_last_1.expand(sub_ids.size(0), -1, -1),
                        d_attn_1.expand(sub_ids.size(0), -1),
                        primary_adapter_input=p_qkv,
                    )
                else:
                    out = self.model.forward_from_hidden(
                        p_last, sub_attn, primary_adapter_input=p_qkv
                    )
                h_lp = F.log_softmax(out.primary_holo_logits[:, :, vocab_ids].float(), dim=-1)
                a_lp = F.log_softmax(out.primary_apo_logits[:, :, vocab_ids].float(), dim=-1)
                holo_chunks.append(h_lp)
                apo_chunks.append(a_lp)
            return torch.cat(holo_chunks, dim=0), torch.cat(apo_chunks, dim=0)

        unique_mut_sites = sorted({ed.pos_0based for eds in mapped_edits for ed in eds})
        n_vars = len(full_sequences)
        holo_arr = np.zeros(n_vars, dtype=np.float64)
        apo_arr = np.zeros(n_vars, dtype=np.float64)

        if unique_mut_sites:
            if mode == "wt_marginal":
                h_lp, a_lp = _eval_primary_batch(base_ids.unsqueeze(0))
                pos_holo = {p: h_lp[0, tok_positions[p]] for p in unique_mut_sites}
                pos_apo = {p: a_lp[0, tok_positions[p]] for p in unique_mut_sites}
                for idx, eds in enumerate(mapped_edits):
                    if not eds:
                        continue
                    hv = [float((pos_holo[e.pos_0based][tok_to_col[e.mut_token]] - pos_holo[e.pos_0based][tok_to_col[e.wt_token]]).item()) for e in eds]
                    av = [float((pos_apo[e.pos_0based][tok_to_col[e.mut_token]] - pos_apo[e.pos_0based][tok_to_col[e.wt_token]]).item()) for e in eds]
                    holo_arr[idx] = float(np.mean(hv) if reduction == "mean" else np.sum(hv))
                    apo_arr[idx] = float(np.mean(av) if reduction == "mean" else np.sum(av))

            elif mode in ("stride_masked", "exact_loo"):
                eff_stride = len(wt_win) + 1 if mode == "exact_loo" else max(1, int(stride))
                groups: list[list[int]] = []
                batch_rows: list[torch.Tensor] = []
                for r in range(eff_stride):
                    pos_r = [p for p in unique_mut_sites if p % eff_stride == r]
                    if not pos_r:
                        continue
                    row_ids = base_ids.clone()
                    for p in pos_r:
                        row_ids[tok_positions[p]] = mask_id
                    groups.append(pos_r)
                    batch_rows.append(row_ids)
                h_lp, a_lp = _eval_primary_batch(torch.stack(batch_rows, dim=0))
                pos_holo = {}
                pos_apo = {}
                for g_i, pos_r in enumerate(groups):
                    for p in pos_r:
                        pos_holo[p] = h_lp[g_i, tok_positions[p]]
                        pos_apo[p] = a_lp[g_i, tok_positions[p]]
                for idx, eds in enumerate(mapped_edits):
                    if not eds:
                        continue
                    hv = [float((pos_holo[e.pos_0based][tok_to_col[e.mut_token]] - pos_holo[e.pos_0based][tok_to_col[e.wt_token]]).item()) for e in eds]
                    av = [float((pos_apo[e.pos_0based][tok_to_col[e.mut_token]] - pos_apo[e.pos_0based][tok_to_col[e.wt_token]]).item()) for e in eds]
                    holo_arr[idx] = float(np.mean(hv) if reduction == "mean" else np.sum(hv))
                    apo_arr[idx] = float(np.mean(av) if reduction == "mean" else np.sum(av))

            elif mode == "masked_sites":
                # Simultaneously mask each variant's exact mutation set M in 1 pass per unique M!
                sig_to_idx: dict[tuple[int, ...], int] = {}
                batch_rows = []
                for eds in mapped_edits:
                    sig = tuple(sorted({e.pos_0based for e in eds}))
                    if sig and sig not in sig_to_idx:
                        row_ids = base_ids.clone()
                        for p in sig:
                            row_ids[tok_positions[p]] = mask_id
                        sig_to_idx[sig] = len(batch_rows)
                        batch_rows.append(row_ids)
                h_lp, a_lp = _eval_primary_batch(torch.stack(batch_rows, dim=0))
                for idx, eds in enumerate(mapped_edits):
                    if not eds:
                        continue
                    sig = tuple(sorted({e.pos_0based for e in eds}))
                    b_i = sig_to_idx[sig]
                    hv = [float((h_lp[b_i, tok_positions[e.pos_0based], tok_to_col[e.mut_token]] - h_lp[b_i, tok_positions[e.pos_0based], tok_to_col[e.wt_token]]).item()) for e in eds]
                    av = [float((a_lp[b_i, tok_positions[e.pos_0based], tok_to_col[e.mut_token]] - a_lp[b_i, tok_positions[e.pos_0based], tok_to_col[e.wt_token]]).item()) for e in eds]
                    holo_arr[idx] = float(np.mean(hv) if reduction == "mean" else np.sum(hv))
                    apo_arr[idx] = float(np.mean(av) if reduction == "mean" else np.sum(av))
            else:
                raise ValueError(f"Unsupported mode: {mode!r}")

        ddg_shift = holo_arr - apo_arr
        if component == "holo":
            final_scores = holo_arr
        elif component == "ddg_shift":
            final_scores = ddg_shift
        elif component == "apo":
            final_scores = apo_arr
        elif component == "guided":
            final_scores = holo_arr + guided_beta * ddg_shift
        else:
            raise ValueError(f"Unknown score component: {component!r}")

        if was_training:
            self.model.train()

        ranked = np.argsort(-final_scores)
        return VariantRankingResult(
            wild_type=wild_type,
            context=context,
            variants=full_sequences,
            mutation_labels=mut_labels,
            scores=final_scores,
            holo_scores=holo_arr,
            apo_scores=apo_arr,
            ddg_shift_scores=ddg_shift,
            ranked_indices=ranked,
        )
