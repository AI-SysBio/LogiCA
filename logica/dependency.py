"""Cross-Modal & Intra-Sequence Dependency Maps and Zero-Shot KL Hotspot Detection.

Implements Section 2.6 (Eq. 6) of the LogiCA paper:
  - Perturbation-based Cross-Modal Dependency Maps ``D_{ij}^{y -> x}`` measuring how masking or
    substituting each partner token ``y_j`` shifts the native vocabulary distribution at each
    primary residue ``x_i`` (without any 3D contact supervision);
  - Annotation-free Zero-Shot KL Hotspot Self-Detection ``D_KL(pi(.|x,y) || pi(.|x))`` (single
    forward pass) to identify binding-pocket / interface sites when ``A`` is 100% unknown;
  - Intra-sequence residue–residue epistatic dependency maps when LogiCA is used in
    Single-BioFM mode (``partner=None``);
  - ``<5 A`` structural contact AUROC & AUPR evaluation helpers.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal, Sequence

import numpy as np
from sklearn.metrics import average_precision_score, roc_auc_score
import torch
import torch.nn.functional as F

from .backbones import CANONICAL_AAS
from .model import LogiCA

DivergenceMetric = Literal["sym_kl", "l2", "jsd"]


@dataclass
class DependencyMapResult:
    """Structured output of cross-modal or intra-sequence dependency mapping."""

    primary_tokens: list[str]
    partner_tokens: list[str]
    matrix: np.ndarray  # shape (L_primary, L_partner): coupling from partner token j to primary token i
    primary_profile: np.ndarray  # shape (L_primary,): aggregated sensitivity per primary residue
    partner_profile: np.ndarray  # shape (L_partner,): aggregated influence per partner token
    metric: str
    mode: str  # "cross_modal" or "intra_sequence"

    def top_primary_sites(self, k: int = 10) -> list[tuple[int, str, float]]:
        """Returns 1-based ``[(pos_1based, token, score), ...]`` for top ``k`` sensitive residues."""
        order = np.argsort(-self.primary_profile)[:k]
        return [(int(i) + 1, self.primary_tokens[int(i)], float(self.primary_profile[int(i)])) for i in order]


def _distribution_divergence(
    p_base: torch.Tensor,
    p_pert: torch.Tensor,
    metric: DivergenceMetric = "sym_kl",
) -> torch.Tensor:
    """Computes token-distribution shift between ``p_base`` ``(1, L, V)`` and ``p_pert`` ``(Q, L, V)``."""
    p = p_base.clamp_min(1e-8)
    q = p_pert.clamp_min(1e-8)
    if metric == "l2":
        return torch.linalg.vector_norm(p - q, ord=2, dim=-1)
    if metric == "sym_kl":
        kl_pq = (p * (p.log() - q.log())).sum(dim=-1)
        kl_qp = (q * (q.log() - p.log())).sum(dim=-1)
        return 0.5 * (kl_pq + kl_qp)
    if metric == "jsd":
        m = 0.5 * (p + q)
        kl_pm = (p * (p.log() - m.log())).sum(dim=-1)
        kl_qm = (q * (q.log() - m.log())).sum(dim=-1)
        return 0.5 * (kl_pm + kl_qm)
    raise ValueError(f"Unsupported divergence metric: {metric!r}")


def evaluate_contact_localization(
    contact_labels: Sequence[int] | np.ndarray,
    dependency_scores: Sequence[float] | np.ndarray,
) -> dict[str, float]:
    """Computes ``<5 A`` contact localization AUROC and AUPR."""
    y = np.asarray(contact_labels, dtype=np.int32)
    s = np.asarray(dependency_scores, dtype=np.float64)
    if len(y) < 3 or len(set(y.tolist())) < 2:
        return {"auroc": float("nan"), "aupr": float("nan")}
    return {
        "auroc": float(roc_auc_score(y, s)),
        "aupr": float(average_precision_score(y, s)),
    }


class DependencyMapper:
    """Computes cross-modal and intra-sequence native-logit dependency maps from :class:`LogiCA`."""

    def __init__(self, model: LogiCA) -> None:
        self.model = model

    @torch.no_grad()
    def detect_kl_hotspots(
        self,
        primary_sequence: str,
        partner_sequence: str,
        *,
        top_k: int = 8,
        alphabet: Sequence[str] = CANONICAL_AAS,
        max_primary_len: int = 1022,
        max_partner_len: int = 256,
    ) -> dict[str, object]:
        """Single-pass zero-shot interface hotspot detection ``D_KL(pi(.|x,y) || pi(.|x,empty))``.

        Recovers binding-pocket / interface sites with zero 3D structural annotation.
        """
        if not self.model.is_dual or self.model.partner_backbone is None:
            raise ValueError("`detect_kl_hotspots` requires a dual-BioFM LogiCA model.")

        was_training = self.model.training
        self.model.eval()
        device = self.model.device

        p_enc = self.model.primary_backbone.tokenize(
            primary_sequence, max_length=max_primary_len + 2, device=device
        )
        d_enc = self.model.partner_backbone.tokenize(
            partner_sequence, max_length=max_partner_len, device=device
        )
        out = self.model(
            p_enc["input_ids"],
            p_enc["attention_mask"],
            d_enc["input_ids"],
            d_enc["attention_mask"],
        )

        valid_pos = torch.nonzero(
            (p_enc["attention_mask"][0] == 1) & (~p_enc["special_tokens_mask"][0].bool()),
            as_tuple=False,
        ).flatten()
        vocab_ids = torch.tensor(
            [self.model.primary_backbone.token_to_id(a) for a in alphabet],
            dtype=torch.long,
            device=device,
        )

        p_holo = F.softmax(out.primary_holo_logits[0, valid_pos][:, vocab_ids].float(), dim=-1).clamp_min(1e-8)
        p_apo = F.softmax(out.primary_apo_logits[0, valid_pos][:, vocab_ids].float(), dim=-1).clamp_min(1e-8)
        kl_per_site = (p_holo * (p_holo.log() - p_apo.log())).sum(dim=-1).cpu().numpy()

        k_eff = min(top_k, len(kl_per_site))
        top_idx_0based = np.argsort(-kl_per_site)[:k_eff].tolist()

        if was_training:
            self.model.train()

        return {
            "kl_profile": kl_per_site,
            "top_indices_0based": [int(i) for i in top_idx_0based],
            "top_positions_1based": [int(i) + 1 for i in top_idx_0based],
        }

    @torch.no_grad()
    def compute_cross_dependency_map(
        self,
        primary_sequence: str,
        partner_sequence: str,
        *,
        perturbation: Literal["mask", "substitutions"] = "mask",
        substitution_tokens: Sequence[str] | None = None,
        metric: DivergenceMetric = "sym_kl",
        reduction: Literal["mean", "max"] = "mean",
        primary_vocab_subset: Sequence[str] | None = CANONICAL_AAS,
        max_primary_len: int = 1022,
        max_partner_len: int = 256,
        batch_size: int = 64,
    ) -> DependencyMapResult:
        """Computes the 2D cross-modal dependency map ``D_{ij}^{partner -> primary}`` (Eq. 6).

        Encodes ``primary_sequence`` ONCE, perturbs each non-special partner token ``j`` in
        ``partner_sequence`` (via masking or token substitutions), and measures how strongly
        each primary residue ``i``'s native token distribution changes.
        """
        if not self.model.is_dual or self.model.partner_backbone is None:
            raise ValueError("`compute_cross_dependency_map` requires a dual-BioFM LogiCA model.")

        was_training = self.model.training
        self.model.eval()
        device = self.model.device

        p_enc = self.model.primary_backbone.tokenize(
            primary_sequence, max_length=max_primary_len + 2, device=device
        )
        d_enc = self.model.partner_backbone.tokenize(
            partner_sequence, max_length=max_partner_len, device=device
        )

        p_valid = torch.nonzero(
            (p_enc["attention_mask"][0] == 1) & (~p_enc["special_tokens_mask"][0].bool()),
            as_tuple=False,
        ).flatten()
        d_valid = torch.nonzero(
            (d_enc["attention_mask"][0] == 1) & (~d_enc["special_tokens_mask"][0].bool()),
            as_tuple=False,
        ).flatten()

        p_tokens = [self.model.primary_backbone.id_to_token(int(p_enc["input_ids"][0, idx].item())) for idx in p_valid]
        d_tokens = [self.model.partner_backbone.id_to_token(int(d_enc["input_ids"][0, idx].item())) for idx in d_valid]

        if primary_vocab_subset is not None:
            try:
                vocab_slice = torch.tensor(
                    [self.model.primary_backbone.token_to_id(a) for a in primary_vocab_subset],
                    dtype=torch.long,
                    device=device,
                )
            except KeyError:
                vocab_slice = None
        else:
            vocab_slice = None

        # Encode primary sequence ONCE!
        p_last_1, p_qkv_1 = self.model.encode_primary(p_enc["input_ids"], p_enc["attention_mask"])

        # Build base + perturbed partner token batches
        d_base_ids = d_enc["input_ids"][0]
        d_base_attn = d_enc["attention_mask"][0]

        pert_rows: list[torch.Tensor] = [d_base_ids]
        col_slices: list[tuple[int, int]] = []

        if perturbation == "mask" or substitution_tokens is None:
            mask_d = self.model.partner_backbone.mask_token_id
            for d_pos in d_valid.tolist():
                st_row = len(pert_rows)
                row = d_base_ids.clone()
                row[int(d_pos)] = mask_d
                pert_rows.append(row)
                col_slices.append((st_row, st_row + 1))
        else:
            sub_ids = [self.model.partner_backbone.token_to_id(t) for t in substitution_tokens]
            for d_pos in d_valid.tolist():
                st_row = len(pert_rows)
                orig_tid = int(d_base_ids[int(d_pos)].item())
                for sid in sub_ids:
                    if sid == orig_tid:
                        continue
                    row = d_base_ids.clone()
                    row[int(d_pos)] = sid
                    pert_rows.append(row)
                col_slices.append((st_row, len(pert_rows)))

        all_d_ids = torch.stack(pert_rows, dim=0)
        all_d_attn = d_base_attn.unsqueeze(0).expand(all_d_ids.size(0), -1)

        probs_chunks: list[torch.Tensor] = []
        for st in range(0, all_d_ids.size(0), batch_size):
            sub_d_ids = all_d_ids[st : st + batch_size]
            sub_d_attn = all_d_attn[st : st + batch_size]
            b = sub_d_ids.size(0)
            d_last = self.model.encode_partner(sub_d_ids, sub_d_attn)
            out = self.model.forward_from_hidden(
                p_last_1.expand(b, -1, -1),
                p_enc["attention_mask"].expand(b, -1),
                d_last,
                sub_d_attn,
                primary_adapter_input=p_qkv_1.expand(b, -1, -1),
            )
            lg = out.primary_holo_logits[:, p_valid, :]
            if vocab_slice is not None:
                lg = lg[:, :, vocab_slice]
            probs_chunks.append(F.softmax(lg.float(), dim=-1))

        all_probs = torch.cat(probs_chunks, dim=0)  # (1 + N_pert, L_p, V)
        p_base = all_probs[0:1]

        cols: list[np.ndarray] = []
        for st_r, en_r in col_slices:
            if en_r <= st_r:
                cols.append(np.zeros(len(p_tokens), dtype=np.float64))
                continue
            div = _distribution_divergence(p_base, all_probs[st_r:en_r], metric=metric)  # (K_sub, L_p)
            cols.append(div.mean(dim=0).cpu().numpy().astype(np.float64))

        matrix = np.stack(cols, axis=1) if cols else np.zeros((len(p_tokens), 0), dtype=np.float64)
        if matrix.size > 0:
            p_prof = matrix.mean(axis=1) if reduction == "mean" else matrix.max(axis=1)
            d_prof = matrix.mean(axis=0) if reduction == "mean" else matrix.max(axis=0)
        else:
            p_prof = np.zeros(len(p_tokens), dtype=np.float64)
            d_prof = np.zeros(len(d_tokens), dtype=np.float64)

        if was_training:
            self.model.train()

        return DependencyMapResult(
            primary_tokens=p_tokens,
            partner_tokens=d_tokens,
            matrix=matrix,
            primary_profile=p_prof,
            partner_profile=d_prof,
            metric=metric,
            mode="cross_modal",
        )

    @torch.no_grad()
    def compute_intra_dependency_map(
        self,
        sequence: str,
        *,
        metric: DivergenceMetric = "sym_kl",
        primary_vocab_subset: Sequence[str] | None = CANONICAL_AAS,
        max_len: int = 512,
        batch_size: int = 64,
    ) -> DependencyMapResult:
        """Computes an intra-sequence residue–residue epistatic dependency map (Single-BioFM mode)."""
        was_training = self.model.training
        self.model.eval()
        device = self.model.device

        p_enc = self.model.primary_backbone.tokenize(sequence, max_length=max_len + 2, device=device)
        valid = torch.nonzero(
            (p_enc["attention_mask"][0] == 1) & (~p_enc["special_tokens_mask"][0].bool()),
            as_tuple=False,
        ).flatten()
        tokens = [self.model.primary_backbone.id_to_token(int(p_enc["input_ids"][0, idx].item())) for idx in valid]

        vocab_slice = None
        if primary_vocab_subset is not None:
            try:
                vocab_slice = torch.tensor(
                    [self.model.primary_backbone.token_to_id(a) for a in primary_vocab_subset],
                    dtype=torch.long,
                    device=device,
                )
            except KeyError:
                vocab_slice = None

        base_ids = p_enc["input_ids"][0]
        base_attn = p_enc["attention_mask"][0]
        mask_id = self.model.primary_backbone.mask_token_id

        rows = [base_ids]
        for pos in valid.tolist():
            r = base_ids.clone()
            r[int(pos)] = mask_id
            rows.append(r)
        all_ids = torch.stack(rows, dim=0)
        all_attn = base_attn.unsqueeze(0).expand(all_ids.size(0), -1)

        probs_list: list[torch.Tensor] = []
        for st in range(0, all_ids.size(0), batch_size):
            sub_ids = all_ids[st : st + batch_size]
            sub_attn = all_attn[st : st + batch_size]
            out = self.model(sub_ids, sub_attn)
            lg = out.primary_holo_logits[:, valid, :]
            if vocab_slice is not None:
                lg = lg[:, :, vocab_slice]
            probs_list.append(F.softmax(lg.float(), dim=-1))

        all_probs = torch.cat(probs_list, dim=0)
        div = _distribution_divergence(all_probs[0:1], all_probs[1:], metric=metric)  # (L, L)
        mat = div.T.cpu().numpy().astype(np.float64)
        np.fill_diagonal(mat, 0.0)

        if was_training:
            self.model.train()

        prof = mat.mean(axis=1) if mat.size > 0 else np.zeros(len(tokens), dtype=np.float64)
        return DependencyMapResult(
            primary_tokens=tokens,
            partner_tokens=tokens,
            matrix=mat,
            primary_profile=prof,
            partner_profile=prof,
            metric=metric,
            mode="intra_sequence",
        )
