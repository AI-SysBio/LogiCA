#!/usr/bin/env python3
"""Step 3: Evaluate `AbLang2-Paired` + `ESM-2` LogiCA on All 4 Official AbRank Suites.

Generates official AbRank submission files (`predictions_<suite>.tsv`) with:
  - Pairwise suites (`maturation`, `transfer`, `unseen_ag_test`):
      `pair_id`, `pred_diff` (exact antisymmetric `f(1, 2) = -f(2, 1)`),
      `pred_impact` (exact even-symmetric `q(1, 2) = +q(2, 1)` from partner-induced native-logit KL + |Δℓ|)
  - Pointwise suites (`discovery`, `generative`):
      `AbAgID`, `pred_score` (native-logit viability + antigen-conditioned log-likelihood gain)

The generated TSV can be evaluated directly with `AI-SysBio/AbRank`:
    python -m abrank.evaluate --suite maturation --track prospective --preds predictions_maturation.tsv
and is also scored automatically at the end of this script.

Usage:
  python antibodies-abrank/scripts/03_evaluate_on_abrank.py \\
      --suite maturation --checkpoint antibodies-abrank/outputs/ablang_esm_logica/ablang_esm_logica.pt \\
      --kaggle-dataset aurlienplissier/abrank

  # Zero-shot evaluation on AbRank-Generative
  python antibodies-abrank/scripts/03_evaluate_on_abrank.py --suite generative --kaggle-dataset aurlienplissier/abrank

  # Fast CPU smoke test
  python antibodies-abrank/scripts/03_evaluate_on_abrank.py --suite maturation --abrank-dir /path/to/benchmark --smoke-test
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys
import time

import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F

SCRIPT_DIR = Path(__file__).resolve().parent
SUBFOLDER_ROOT = SCRIPT_DIR.parent
REPO_ROOT = SUBFOLDER_ROOT.parent
for p in (str(REPO_ROOT), str(SCRIPT_DIR)):
    if p not in sys.path:
        sys.path.insert(0, p)

from abrank_loader import (  # noqa: E402
    DEFAULT_KAGGLE_DATASET,
    evaluate_predictions_with_abrank,
    load_complex_lookup,
    resolve_abrank_dir,
)
from logica import LogiCA, mean_selected_log_probability  # noqa: E402


@torch.no_grad()
def score_unique_complexes(
    model: LogiCA,
    abag_ids: list[str],
    lookup: dict[str, dict],
    *,
    batch_size: int,
    max_ab_len: int,
    max_ag_len: int,
    device: torch.device,
) -> tuple[dict[str, float], dict[str, float]]:
    """Score unique `(Ab, Ag)` complexes in native logit space, returning:
      - `score_map[AbAgID]`: scalar native-logit affinity score (`apo + contextual holo shift`)
      - `kl_map[AbAgID]`: mean partner-induced symmetric KL divergence (`D_KL(pi_holo || pi_apo)`)
    """
    model.eval()
    valid_ids = [i for i in dict.fromkeys(abag_ids) if i in lookup]
    score_map: dict[str, float] = {}
    kl_map: dict[str, float] = {}

    for st in range(0, len(valid_ids), batch_size):
        chunk = valid_ids[st : st + batch_size]
        ab_b = [lookup[i]["ab_seq"] for i in chunk]
        ag_b = [lookup[i]["ag_seq"] for i in chunk]

        p_enc = model.primary_backbone.tokenize(ab_b, max_length=max_ab_len, device=device)
        p_valid = (p_enc["attention_mask"] == 1) & (~p_enc["special_tokens_mask"].bool())

        if model.is_dual and model.partner_backbone is not None:
            d_enc = model.partner_backbone.tokenize(ag_b, max_length=max_ag_len, device=device)
            d_valid = (d_enc["attention_mask"] == 1) & (~d_enc["special_tokens_mask"].bool())
            out = model(
                p_enc["input_ids"],
                p_enc["attention_mask"],
                d_enc["input_ids"],
                d_enc["attention_mask"],
            )
            assert out.partner_holo_logits is not None and out.partner_apo_logits is not None
            ab_h = mean_selected_log_probability(out.primary_holo_logits, p_enc["input_ids"], p_valid)
            ab_a = mean_selected_log_probability(out.primary_apo_logits, p_enc["input_ids"], p_valid)
            ag_h = mean_selected_log_probability(out.partner_holo_logits, d_enc["input_ids"], d_valid)
            ag_a = mean_selected_log_probability(out.partner_apo_logits, d_enc["input_ids"], d_valid)
            sc = ab_a + (ab_h - ab_a) + 0.5 * (ag_h - ag_a)

            lp_h = F.log_softmax(out.primary_holo_logits.float(), dim=-1)
            lp_a = F.log_softmax(out.primary_apo_logits.float(), dim=-1)
            p_h = lp_h.exp()
            p_a = lp_a.exp()
            sym_kl_tok = 0.5 * ((p_h * (lp_h - lp_a)).sum(dim=-1) + (p_a * (lp_a - lp_h)).sum(dim=-1))
            vf = p_valid.float()
            kl_mean = (sym_kl_tok * vf).sum(dim=1) / vf.sum(dim=1).clamp_min(1.0)
        else:
            out = model(p_enc["input_ids"], p_enc["attention_mask"])
            sc = mean_selected_log_probability(out.primary_holo_logits, p_enc["input_ids"], p_valid)
            kl_mean = torch.zeros_like(sc)

        for abag_id, s_val, k_val in zip(chunk, sc.cpu().tolist(), kl_mean.cpu().tolist()):
            score_map[abag_id] = float(s_val)
            kl_map[abag_id] = float(k_val)

    return score_map, kl_map


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--suite",
        choices=["maturation", "discovery", "generative", "transfer", "unseen_ag_test"],
        default="maturation",
        help="Target AbRank suite to score.",
    )
    parser.add_argument("--checkpoint", type=Path, default=None, help="Trained LogiCA `.pt` checkpoint (optional; zero-shot if omitted)")
    parser.add_argument("--ab-model", type=str, default="ablang2-paired")
    parser.add_argument("--ag-model", type=str, default="facebook/esm2_t12_35M_UR50D")
    parser.add_argument("--abrank-dir", type=Path, default=None, help="Local AbRank `data/benchmark/` path")
    parser.add_argument("--kaggle-dataset", type=str, default=DEFAULT_KAGGLE_DATASET)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--max-ab-len", type=int, default=260)
    parser.add_argument("--max-ag-len", type=int, default=384)
    parser.add_argument("--bootstrap", type=int, default=200)
    parser.add_argument("--device", type=str, default="cuda:0" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--out-dir", type=Path, default=SUBFOLDER_ROOT / "outputs" / "abrank_eval")
    parser.add_argument("--smoke-test", action="store_true")
    args = parser.parse_args()

    t0 = time.time()
    bdir = resolve_abrank_dir(args.abrank_dir, kaggle_dataset=args.kaggle_dataset)
    include_transfer = args.suite == "transfer"
    lookup = load_complex_lookup(bdir, include_vhh=True, include_transfer=include_transfer)
    device = torch.device(args.device)

    if args.checkpoint is not None:
        print(f"[AbRank Eval] Loading checkpoint: {args.checkpoint}")
        model = LogiCA.from_checkpoint(args.checkpoint, device=device)
    else:
        ag_m = "facebook/esm2_t6_8M_UR50D" if args.smoke_test else args.ag_model
        print(f"[AbRank Eval] Zero-shot LogiCA ({args.ab_model} + {ag_m}) on {device}")
        model = LogiCA.from_pretrained(args.ab_model, ag_m, width=128 if args.smoke_test else 256, layers=1 if args.smoke_test else 2).to(device)

    args.out_dir.mkdir(parents=True, exist_ok=True)
    sp_dir = bdir / "splits"

    if args.suite in ("maturation", "unseen_ag_test", "transfer"):
        if args.suite == "maturation":
            parts = [pd.read_csv(sp_dir / "unseen_ag" / f"{s}.tsv", sep="\t", low_memory=False) for s in ("train", "val", "test")]
            pairs_df = pd.concat(parts, ignore_index=True)
            pairs_df = pairs_df[pairs_df["Category"].isin(["L1", "L2", "L3"])].reset_index(drop=True)
        elif args.suite == "unseen_ag_test":
            pairs_df = pd.read_csv(sp_dir / "unseen_ag" / "test.tsv", sep="\t", low_memory=False)
        else:
            pairs_df = pd.read_csv(sp_dir / "transfer" / "ic50_test.tsv", sep="\t", low_memory=False)

        if args.smoke_test:
            # Take a stratified 2-antigen subset so per-antigen macro metrics are well-defined
            top_ags = pairs_df["Ag_group_1"].value_counts().head(2).index.tolist()
            pairs_df = pd.concat(
                [pairs_df[pairs_df["Ag_group_1"] == ag].head(12) for ag in top_ags],
                ignore_index=True,
            )
            args.bootstrap = 20

        needed_ids = list(set(pairs_df["AbAgID_1"]).union(pairs_df["AbAgID_2"]))
        print(f"[AbRank Eval] Scoring {len(needed_ids)} unique complexes covering {len(pairs_df)} pairs...")
        score_map, kl_map = score_unique_complexes(
            model,
            needed_ids,
            lookup,
            batch_size=args.batch_size,
            max_ab_len=args.max_ab_len,
            max_ag_len=args.max_ag_len,
            device=device,
        )
        s1 = pairs_df["AbAgID_1"].map(score_map).to_numpy(dtype=np.float64)
        s2 = pairs_df["AbAgID_2"].map(score_map).to_numpy(dtype=np.float64)
        k1 = pairs_df["AbAgID_1"].map(kl_map).to_numpy(dtype=np.float64)
        k2 = pairs_df["AbAgID_2"].map(kl_map).to_numpy(dtype=np.float64)
        pred_diff = s1 - s2
        # Exact even-symmetric impact score: |pred_diff| + symmetric partner-induced KL sensitivity
        pred_impact = np.abs(pred_diff) + 0.5 * (k1 + k2)

        preds_df = pd.DataFrame(
            {
                "pair_id": pairs_df["pair_id"],
                "pred_diff": pred_diff,
                "pred_impact": pred_impact,
            }
        )
        pred_path = args.out_dir / f"predictions_{args.suite}.tsv"
        preds_df.to_csv(pred_path, sep="\t", index=False)

        metrics = evaluate_predictions_with_abrank(
            args.suite,
            preds_df,
            bdir,
            pairs_df=pairs_df,
            track="prospective",
            n_bootstrap=args.bootstrap,
        )
    else:
        # Pointwise suites: `discovery` or `generative`
        if args.suite == "discovery":
            pw_all = pd.concat(
                [pd.read_csv(sp_dir / "unseen_ag" / f"{s}_pointwise.tsv", sep="\t", low_memory=False) for s in ("train", "val", "test")],
                ignore_index=True,
            )
            panels = [g for (_, _), g in pw_all.groupby(["Ag_group", "Dataset"], sort=True) if 6 <= len(g) <= 200 and g["binding_score"].nunique() >= 3]
            pw_df = pd.concat(panels, ignore_index=True)
        else:
            pw_df = pd.read_csv(sp_dir / "generative" / "generative_panels.tsv", sep="\t", low_memory=False)

        if args.smoke_test:
            top_g = pw_df.groupby(["Ag_group", "Dataset"]).size().sort_values(ascending=False).head(2).index.tolist()
            pw_df = pd.concat(
                [pw_df[(pw_df["Ag_group"] == ag) & (pw_df["Dataset"] == ds)].head(10) for ag, ds in top_g],
                ignore_index=True,
            )
            args.bootstrap = 20

        needed_ids = pw_df["AbAgID"].astype(str).unique().tolist()
        print(f"[AbRank Eval] Scoring {len(needed_ids)} pointwise variants for `{args.suite}`...")
        score_map, _ = score_unique_complexes(
            model,
            needed_ids,
            lookup,
            batch_size=args.batch_size,
            max_ab_len=args.max_ab_len,
            max_ag_len=args.max_ag_len,
            device=device,
        )
        preds_df = pd.DataFrame(
            {
                "AbAgID": needed_ids,
                "pred_score": [score_map[i] for i in needed_ids],
            }
        )
        pred_path = args.out_dir / f"predictions_{args.suite}.tsv"
        preds_df.to_csv(pred_path, sep="\t", index=False)

        metrics = evaluate_predictions_with_abrank(
            args.suite,
            preds_df,
            bdir,
            track="prospective",
            n_bootstrap=args.bootstrap,
        )

    metrics["predictions_tsv"] = str(pred_path)
    metrics["elapsed_sec"] = round(time.time() - t0, 2)
    out_json = args.out_dir / f"metrics_{args.suite}.json"
    out_json.write_text(json.dumps(metrics, indent=2) + "\n", encoding="utf-8")

    print(f"\nWrote AbRank submission TSV -> {pred_path}")
    print(f"Wrote metrics JSON          -> {out_json}")
    print(json.dumps(metrics, indent=2))
    print(
        f"\nVerify independently with AI-SysBio/AbRank:\n"
        f"  python -m abrank.evaluate --suite {args.suite if args.suite != 'unseen_ag_test' else 'maturation'} --preds {pred_path}"
    )


if __name__ == "__main__":
    main()
