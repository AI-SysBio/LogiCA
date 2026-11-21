#!/usr/bin/env python3
"""Step 2: Train `AbLang2-Paired` + `ESM-2` LogiCA on AbRank (`logica` Toolbox).

Couples `ablang2-paired` (`VH|VL` or single-domain `VHH|`) with an antigen protein language model
(`facebook/esm2_t12_35M_UR50D` / `150M` / `650M`, or `--single-biofm` for unimodal `ablang2-paired`)
using the model-agnostic `LogiCA` library (`from logica import LogiCA, ...`).

Two-Stage Native-Logit Training Protocol:
  - Stage 1 (Self-Supervised Partner Alignment):
      Optimizes bidirectional `InfoNCELogitLoss` over unique bound `(VH|VL, Ag)` training complexes
      with homologous antigen-cluster (`Ag_cluster`) false-negative masking.
  - Stage 2 (3-State Noise-Buffered Pairwise Logit Fine-Tuning):
      Trains directly on AbRank's 3-state pairs (`{-1: Depleted, 0: Neutral, +1: Improved}`),
      combining:
        1. Temperature-scaled Bradley–Terry directional loss on impactful pairs (`label in {-1, +1}`),
        2. Neutral deadzone hinge `max(0, |pred_diff| - eps)^2` on noise-buffered `label == 0` pairs,
        3. `AncestralKLAnchorLoss` trust-region anchoring on native token logits, and
        4. Late-layer LoRA + elevated cross-attention LR via `model.configure_finetuning_optimizer(...)`.

Strictly preserves both native BioFM vocabulary heads so the trained checkpoint supports:
  - AbRank evaluation (`03_evaluate_on_abrank.py`),
  - 2D Antigen -> Paratope dependency maps (`04_paratope_dependency_and_gibbs.py`), and
  - Antigen-conditioned CDR-H3 Coordinate Gibbs affinity maturation.

Usage:
  # Full training loading AbRank directly from Kaggle (`aurlienplissier/abrank`)
  python applications/antibodies-abrank/scripts/02_train_ablang_esm_logica.py \\
      --kaggle-dataset aurlienplissier/abrank --device cuda:0

  # Leave-One-Antigen-Cluster-Out (LOACO) fold training for `AbRank-Maturation`
  python applications/antibodies-abrank/scripts/02_train_ablang_esm_logica.py \\
      --kaggle-dataset aurlienplissier/abrank --holdout-cluster AgClust_001 --device cuda:0

  # Fast CPU smoke test
  python applications/antibodies-abrank/scripts/02_train_ablang_esm_logica.py --abrank-dir /path/to/benchmark --smoke-test
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import random
import sys
import time

import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset

SCRIPT_DIR = Path(__file__).resolve().parent
SUBFOLDER_ROOT = SCRIPT_DIR.parent
REPO_ROOT = SUBFOLDER_ROOT.parents[1]
for p in (str(REPO_ROOT), str(SCRIPT_DIR)):
    if p not in sys.path:
        sys.path.insert(0, p)

from abrank_loader import (  # noqa: E402
    DEFAULT_KAGGLE_DATASET,
    evaluate_predictions_with_abrank,
    load_complex_lookup,
    resolve_abrank_dir,
)
from logica import (  # noqa: E402
    AncestralKLAnchorLoss,
    InfoNCELogitLoss,
    LogiCA,
    mean_selected_log_probability,
)


class BoundComplexDataset(Dataset):
    def __init__(self, ab_seqs: list[str], ag_seqs: list[str], ag_clusters: list[str]) -> None:
        self.ab_seqs = ab_seqs
        self.ag_seqs = ag_seqs
        self.ag_clusters = ag_clusters

    def __len__(self) -> int:
        return len(self.ab_seqs)

    def __getitem__(self, idx: int) -> tuple[str, str, str]:
        return self.ab_seqs[idx], self.ag_seqs[idx], self.ag_clusters[idx]


class ThreeStatePairDataset(Dataset):
    def __init__(
        self,
        pairs_df: pd.DataFrame,
        lookup: dict[str, dict],
    ) -> None:
        rows = []
        for r in pairs_df.itertuples(index=False):
            c1 = lookup.get(str(r.AbAgID_1))
            c2 = lookup.get(str(r.AbAgID_2))
            if c1 is None or c2 is None:
                continue
            rows.append(
                (
                    c1["ab_seq"],
                    c1["ag_seq"],
                    c2["ab_seq"],
                    c2["ag_seq"],
                    int(r.label),
                    float(r.binding_diff),
                )
            )
        self.rows = rows

    def __len__(self) -> int:
        return len(self.rows)

    def __getitem__(self, idx: int) -> tuple[str, str, str, str, int, float]:
        return self.rows[idx]


def score_complex_batch_logits(
    model: LogiCA,
    ab_seqs: list[str],
    ag_seqs: list[str],
    *,
    max_ab_len: int,
    max_ag_len: int,
    device: torch.device,
    beta_holo_shift: float = 1.0,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """Compute native-logit complex score (viability + contextual partner shift) and KL anchor terms.

    Returns ``(complex_score, holo_logits_ab, apo_logits_ab, ab_valid_mask)`` while strictly
    preserving the native amino-acid vocabulary heads of both BioFMs.
    """
    p_enc = model.primary_backbone.tokenize(ab_seqs, max_length=max_ab_len, device=device)
    p_valid = (p_enc["attention_mask"] == 1) & (~p_enc["special_tokens_mask"].bool())

    if model.is_dual and model.partner_backbone is not None:
        d_enc = model.partner_backbone.tokenize(ag_seqs, max_length=max_ag_len, device=device)
        d_valid = (d_enc["attention_mask"] == 1) & (~d_enc["special_tokens_mask"].bool())
        out = model(
            p_enc["input_ids"],
            p_enc["attention_mask"],
            d_enc["input_ids"],
            d_enc["attention_mask"],
        )
        assert out.partner_holo_logits is not None and out.partner_apo_logits is not None
        ab_holo_ll = mean_selected_log_probability(out.primary_holo_logits, p_enc["input_ids"], p_valid)
        ab_apo_ll = mean_selected_log_probability(out.primary_apo_logits, p_enc["input_ids"], p_valid)
        ag_holo_ll = mean_selected_log_probability(out.partner_holo_logits, d_enc["input_ids"], d_valid)
        ag_apo_ll = mean_selected_log_probability(out.partner_apo_logits, d_enc["input_ids"], d_valid)
        # Headless thermodynamic decomposition: apo viability + contextual binding gain
        score = ab_apo_ll + beta_holo_shift * ((ab_holo_ll - ab_apo_ll) + 0.5 * (ag_holo_ll - ag_apo_ll))
        return score, out.primary_holo_logits, out.primary_apo_logits, p_valid

    out = model(p_enc["input_ids"], p_enc["attention_mask"])
    ab_holo_ll = mean_selected_log_probability(out.primary_holo_logits, p_enc["input_ids"], p_valid)
    return ab_holo_ll, out.primary_holo_logits, out.primary_apo_logits, p_valid


def three_state_noise_buffered_loss(
    pred_diff: torch.Tensor,
    labels: torch.Tensor,
    *,
    tau_bt: float = 0.5,
    eps_neutral: float = 0.15,
    lambda_neutral: float = 0.5,
) -> torch.Tensor:
    """AbRank 3-State Loss: Bradley-Terry on impactful (`+-1`) + deadzone hinge on neutral (`0`)."""
    imp_mask = (labels != 0).float()
    neu_mask = (labels == 0).float()
    y_sign = labels.float()

    loss_bt = (F.softplus(-y_sign * pred_diff / tau_bt) * imp_mask).sum() / imp_mask.sum().clamp_min(1.0)
    loss_neu = (F.relu(torch.abs(pred_diff) - eps_neutral).pow(2) * neu_mask).sum() / neu_mask.sum().clamp_min(1.0)
    return loss_bt + lambda_neutral * loss_neu


def load_training_partitions(
    bdir: Path,
    lookup: dict[str, dict],
    holdout_cluster: str | None = None,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Return `(bound_complexes_df, train_pairs_df, val_pairs_df)` strictly disjoint from test."""
    sp_dir = bdir / "splits" / "unseen_ag"
    if holdout_cluster is None:
        tr_pw = pd.read_csv(sp_dir / "train_pointwise.tsv", sep="\t", low_memory=False)
        tr_pairs = pd.read_csv(sp_dir / "train.tsv", sep="\t", low_memory=False)
        va_pairs = pd.read_csv(sp_dir / "val.tsv", sep="\t", low_memory=False)
        return tr_pw, tr_pairs, va_pairs

    # LOACO mode: hold out `holdout_cluster` + all antibody lineages present in that cluster
    kd = pd.read_csv(
        bdir / "assays_kd.tsv",
        sep="\t",
        usecols=["AbAgID", "Ag_cluster", "Ab_lineage"],
        low_memory=False,
    ).drop_duplicates("AbAgID")
    id2cl = kd.set_index("AbAgID")["Ag_cluster"].astype(str).to_dict()
    id2lin = kd.set_index("AbAgID")["Ab_lineage"].astype(str).to_dict()

    all_pairs = pd.concat(
        [pd.read_csv(sp_dir / f"{s}.tsv", sep="\t", low_memory=False) for s in ("train", "val", "test")],
        ignore_index=True,
    )
    all_pairs = all_pairs[all_pairs["Category"].isin(["L1", "L2", "L3"])].reset_index(drop=True)
    c1 = all_pairs["AbAgID_1"].map(id2cl)
    l1 = all_pairs["AbAgID_1"].map(id2lin)
    l2 = all_pairs["AbAgID_2"].map(id2lin)

    te_mask = c1 == holdout_cluster
    te_lins = set(l1[te_mask]).union(set(l2[te_mask]))
    pool_mask = (~te_mask) & (~l1.isin(te_lins)) & (~l2.isin(te_lins))
    pool = all_pairs[pool_mask].reset_index(drop=True)

    rem_clusters = sorted(pool["AbAgID_1"].map(id2cl).dropna().unique())
    val_clusters = set(rem_clusters[::6])
    va_mask = pool["AbAgID_1"].map(id2cl).isin(val_clusters)
    va_lins = set(pool.loc[va_mask, "AbAgID_1"].map(id2lin)).union(pool.loc[va_mask, "AbAgID_2"].map(id2lin))
    tr_mask = (~va_mask) & (~pool["AbAgID_1"].map(id2lin).isin(va_lins)) & (~pool["AbAgID_2"].map(id2lin).isin(va_lins))

    tr_pairs = pool[tr_mask].reset_index(drop=True)
    va_pairs = pool[va_mask].reset_index(drop=True)

    tr_abag_ids = set(tr_pairs["AbAgID_1"]).union(tr_pairs["AbAgID_2"])
    tr_pw_rows = [
        {"AbAgID": i, "Ag_cluster": lookup[i]["ag_cluster"]}
        for i in sorted(tr_abag_ids)
        if i in lookup
    ]
    return pd.DataFrame(tr_pw_rows), tr_pairs, va_pairs


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--abrank-dir", type=Path, default=None, help="Local AbRank `data/benchmark/` path")
    parser.add_argument("--kaggle-dataset", type=str, default=DEFAULT_KAGGLE_DATASET)
    parser.add_argument("--ab-model", type=str, default="ablang2-paired", help="Primary antibody BioFM (`ablang2-paired` or HF id)")
    parser.add_argument("--ag-model", type=str, default="facebook/esm2_t12_35M_UR50D", help="Partner antigen BioFM")
    parser.add_argument("--single-biofm", action="store_true", help="Train unimodal antibody BioFM (`partner=None`)")
    parser.add_argument("--holdout-cluster", type=str, default=None, help="Optional LOACO held-out `Ag_cluster` (e.g. `AgClust_001`)")
    parser.add_argument("--width", type=int, default=256)
    parser.add_argument("--layers", type=int, default=2)
    parser.add_argument("--max-ab-len", type=int, default=260)
    parser.add_argument("--max-ag-len", type=int, default=384)
    parser.add_argument("--stage1-epochs", type=int, default=3)
    parser.add_argument("--stage2-epochs", type=int, default=5)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--kl-weight", type=float, default=0.05)
    parser.add_argument("--max-stage1-complexes", type=int, default=4000)
    parser.add_argument("--max-stage2-pairs", type=int, default=6000)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--device", type=str, default="cuda:0" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--out-dir", type=Path, default=SUBFOLDER_ROOT / "outputs" / "ablang_esm_logica")
    parser.add_argument("--smoke-test", action="store_true", help="Run fast CPU verification on a small subset")
    args = parser.parse_args()

    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)

    bdir = resolve_abrank_dir(args.abrank_dir, kaggle_dataset=args.kaggle_dataset)
    lookup = load_complex_lookup(bdir, include_vhh=True)
    tr_pw, tr_pairs, va_pairs = load_training_partitions(bdir, lookup, holdout_cluster=args.holdout_cluster)

    if args.smoke_test:
        args.ag_model = "facebook/esm2_t6_8M_UR50D"
        args.width = 128
        args.layers = 1
        args.max_ab_len = 240
        args.max_ag_len = 160
        args.stage1_epochs = 1
        args.stage2_epochs = 1
        args.batch_size = 4
        args.max_stage1_complexes = 12
        args.max_stage2_pairs = 16
        va_pairs = va_pairs.head(12).reset_index(drop=True)

    # Prefer local mutational pairs (`L1-L3`) in Stage 2 while keeping balanced 3-state labels
    local_tr = tr_pairs[tr_pairs["Category"].isin(["L1", "L2", "L3"])]
    if len(local_tr) >= 16:
        tr_pairs = local_tr.reset_index(drop=True)
    if len(tr_pairs) > args.max_stage2_pairs:
        tr_pairs = tr_pairs.sample(n=args.max_stage2_pairs, random_state=args.seed).reset_index(drop=True)

    args.out_dir.mkdir(parents=True, exist_ok=True)
    device = torch.device(args.device)
    partner_spec = None if args.single_biofm else args.ag_model
    print(f"[AbRank LogiCA] Instantiating primary={args.ab_model}, partner={partner_spec} on {device}...")

    model = LogiCA.from_pretrained(
        args.ab_model,
        partner_spec,
        width=args.width,
        layers=args.layers,
        gate_mode="tanh_zero",
    ).to(device)

    param_groups = model.configure_finetuning_optimizer(
        base_lora_lr=args.lr,
        adapter_lr_multiplier=4.0,
        top_k_primary_lora=2,
        top_k_partner_lora=2 if model.is_dual else 0,
        lora_rank=8,
    )
    optimizer = torch.optim.AdamW(param_groups)
    kl_anchor = AncestralKLAnchorLoss()
    t_start = time.time()

    # ------------------------------------------------------------------
    # Stage 1: Self-Supervised Partner Alignment (`InfoNCELogitLoss`)
    # ------------------------------------------------------------------
    s1_history: list[dict[str, float]] = []
    if model.is_dual and args.stage1_epochs > 0:
        # Deduplicate bound (VH|VL, Ag) training complexes stratified across antigen clusters
        seen_pairs: set[tuple[str, str]] = set()
        ab_list, ag_list, cl_list = [], [], []
        pw_shuffled = tr_pw.sample(frac=1.0, random_state=args.seed)
        for abag_id in pw_shuffled["AbAgID"].astype(str):
            rec = lookup.get(abag_id)
            if rec is None:
                continue
            key = (rec["ab_seq"], rec["ag_seq"])
            if key in seen_pairs:
                continue
            seen_pairs.add(key)
            ab_list.append(rec["ab_seq"])
            ag_list.append(rec["ag_seq"])
            cl_list.append(rec["ag_cluster"])
            if len(ab_list) >= args.max_stage1_complexes:
                break

        s1_loader = DataLoader(
            BoundComplexDataset(ab_list, ag_list, cl_list),
            batch_size=args.batch_size,
            shuffle=True,
            drop_last=len(ab_list) >= args.batch_size,
        )
        infonce = InfoNCELogitLoss(
            temperature=0.10,
            alpha=0.5,
            mask_fraction=0.15,
            inbatch_negatives_per_anchor=min(7, max(1, args.batch_size - 1)),
            anchor_mode="bidirectional",
            subtract_apo_baseline=True,
            mlm_aux_weight=0.25,
        )
        print(f"[Stage 1] Contrastive native-logit alignment on {len(ab_list)} unique bound (Ab, Ag) complexes...")
        for ep in range(1, args.stage1_epochs + 1):
            model.train()
            ep_losses = []
            for ab_b, ag_b, cl_b in s1_loader:
                if len(ab_b) < 2:
                    continue
                p_enc = model.primary_backbone.tokenize(list(ab_b), max_length=args.max_ab_len, device=device)
                d_enc = model.partner_backbone.tokenize(list(ag_b), max_length=args.max_ag_len, device=device)  # type: ignore[union-attr]

                # Mask homologous training antigens in the same batch as false negatives
                bsz = len(cl_b)
                fn_mask = torch.zeros((bsz, bsz), dtype=torch.bool, device=device)
                for i in range(bsz):
                    for j in range(bsz):
                        if i != j and cl_b[i] == cl_b[j]:
                            fn_mask[i, j] = True
                # Only apply fn_mask when every row keeps at least 1 valid negative
                if bool((~fn_mask).sum(dim=1).min().item() < 2):
                    fn_mask = None

                out_s1 = infonce(model, p_enc, d_enc, false_negative_mask=fn_mask)
                optimizer.zero_grad()
                out_s1["loss"].backward()
                torch.nn.utils.clip_grad_norm_([p for p in model.parameters() if p.requires_grad], 1.5)
                optimizer.step()
                ep_losses.append(float(out_s1["loss"].item()))
            mean_l = float(np.mean(ep_losses)) if ep_losses else 0.0
            s1_history.append({"epoch": ep, "infonce_loss": round(mean_l, 4)})
            print(f"  [Stage 1] Epoch {ep}/{args.stage1_epochs} | infonce_loss={mean_l:.4f}")

    # ------------------------------------------------------------------
    # Stage 2: 3-State Noise-Buffered Pairwise Logit Fine-Tuning
    # ------------------------------------------------------------------
    tr_pair_ds = ThreeStatePairDataset(tr_pairs, lookup)
    va_pair_ds = ThreeStatePairDataset(va_pairs, lookup)
    tr_pair_loader = DataLoader(tr_pair_ds, batch_size=max(2, args.batch_size // 2), shuffle=True)
    va_pair_loader = DataLoader(va_pair_ds, batch_size=max(2, args.batch_size // 2), shuffle=False)

    print(f"[Stage 2] 3-state noise-buffered fine-tuning on {len(tr_pair_ds)} pairs (val={len(va_pair_ds)})...")
    s2_history: list[dict[str, float]] = []
    for ep in range(1, args.stage2_epochs + 1):
        model.train()
        ep_losses = []
        for ab1, ag1, ab2, ag2, lbl, _bdiff in tr_pair_loader:
            # Random 50/50 orientation flip to enforce antisymmetry during training
            ab1_l, ag1_l, ab2_l, ag2_l = list(ab1), list(ag1), list(ab2), list(ag2)
            lbl_arr = lbl.numpy().astype(np.int64)
            for i in range(len(lbl_arr)):
                if random.random() < 0.5:
                    ab1_l[i], ab2_l[i] = ab2_l[i], ab1_l[i]
                    ag1_l[i], ag2_l[i] = ag2_l[i], ag1_l[i]
                    lbl_arr[i] = -lbl_arr[i]

            s1, h1, a1, m1 = score_complex_batch_logits(
                model, ab1_l, ag1_l, max_ab_len=args.max_ab_len, max_ag_len=args.max_ag_len, device=device
            )
            s2, h2, a2, m2 = score_complex_batch_logits(
                model, ab2_l, ag2_l, max_ab_len=args.max_ab_len, max_ag_len=args.max_ag_len, device=device
            )
            pred_diff = s1 - s2
            lbl_t = torch.from_numpy(lbl_arr).to(device)
            loss_3s = three_state_noise_buffered_loss(pred_diff, lbl_t)
            loss_kl = 0.5 * (kl_anchor(h1, a1.detach(), m1) + kl_anchor(h2, a2.detach(), m2))
            loss = loss_3s + args.kl_weight * loss_kl

            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_([p for p in model.parameters() if p.requires_grad], 1.5)
            optimizer.step()
            ep_losses.append(float(loss.item()))

        # Validation check
        model.eval()
        val_diffs = []
        with torch.no_grad():
            for ab1, ag1, ab2, ag2, _lbl, _bdiff in va_pair_loader:
                s1, _, _, _ = score_complex_batch_logits(
                    model, list(ab1), list(ag1), max_ab_len=args.max_ab_len, max_ag_len=args.max_ag_len, device=device
                )
                s2, _, _, _ = score_complex_batch_logits(
                    model, list(ab2), list(ag2), max_ab_len=args.max_ab_len, max_ag_len=args.max_ag_len, device=device
                )
                val_diffs.extend((s1 - s2).cpu().numpy().tolist())

        val_sub = va_pairs.iloc[: len(val_diffs)].reset_index(drop=True)
        val_preds_df = pd.DataFrame({"pair_id": val_sub["pair_id"], "pred_diff": val_diffs})
        val_metrics = evaluate_predictions_with_abrank("val", val_preds_df, bdir, pairs_df=val_sub, n_bootstrap=0)
        mean_l = float(np.mean(ep_losses)) if ep_losses else 0.0
        rec = {
            "epoch": ep,
            "train_loss": round(mean_l, 4),
            "val_auc_dir": val_metrics.get("auc_dir"),
            "val_auc_ordinal": val_metrics.get("auc_ordinal"),
        }
        s2_history.append(rec)
        print(
            f"  [Stage 2] Epoch {ep}/{args.stage2_epochs} | loss={mean_l:.4f} | "
            f"val_dir={rec['val_auc_dir']} | val_ord={rec['val_auc_ordinal']}"
        )

    ckpt_path = args.out_dir / "ablang_esm_logica.pt"
    model.save_checkpoint(
        ckpt_path,
        extra_metadata={
            "ab_model": args.ab_model,
            "ag_model": partner_spec,
            "holdout_cluster": args.holdout_cluster,
            "stage1_history": s1_history,
            "stage2_history": s2_history,
        },
    )
    summary = {
        "checkpoint": str(ckpt_path),
        "ab_model": args.ab_model,
        "ag_model": partner_spec,
        "holdout_cluster": args.holdout_cluster,
        "elapsed_sec": round(time.time() - t_start, 2),
        "stage1_history": s1_history,
        "stage2_history": s2_history,
    }
    (args.out_dir / "train_summary.json").write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")
    print(f"\nSaved AbLang2+ESM-2 LogiCA checkpoint -> {ckpt_path}")


if __name__ == "__main__":
    main()
