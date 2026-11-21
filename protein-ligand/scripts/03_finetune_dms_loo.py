#!/usr/bin/env python3
"""Step 3: Unseen-Cluster (`LOCO-30%`), Unseen-Ligand (`LOLO`) & `Oncology LOO` Fine-Tuning.

Uses the modular `logica` library (`LogiCA`, `VariantRanker`, `PlackettLuceLogitLoss`,
`AncestralKLAnchorLoss`) to fine-tune `ESM-2` + `SELFormer` from the released
`Yale-CompBio/logica` checkpoints (`8m`, `35m`, `150m`, `650m`) — AND supports single-BioFM
(`ESM-2` only, `partner=None`) and `LogiMLM` / `LogiCA-NoKL` controls!

Usage:
  # Full GPU LOCO-30% fine-tuning across all 32 `<30%` sequence-identity clusters (103 DMS cohorts):
  python protein-ligand/scripts/03_finetune_dms_loo.py --scale 150m --device cuda:0 --epochs 4

  # Single-BioFM native-logit fine-tuning control (ESM-2 without partner ligand):
  python protein-ligand/scripts/03_finetune_dms_loo.py --single-biofm --scale 150m --device cuda:0

  # Fast 1-fold smoke test:
  python protein-ligand/scripts/03_finetune_dms_loo.py --scale 8m --smoke-test
"""

from __future__ import annotations

import argparse
import copy
import json
from pathlib import Path
import re
import sys
import time
from typing import Any

import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F

SUBFOLDER_ROOT = Path(__file__).resolve().parents[1]
REPO_ROOT = SUBFOLDER_ROOT.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from logica import (  # noqa: E402
    CANONICAL_AAS,
    AncestralKLAnchorLoss,
    LogiCA,
    MutationEdit,
    PlackettLuceLogitLoss,
    VariantRanker,
    evaluate_ranking_metrics,
)
from logica.model import YALE_PROTEIN_LIGAND_SCALES  # noqa: E402

MUT_RE = re.compile(r"^([A-Z])(\d+)([A-Z])$")


def load_mutational_dms_cohorts(data_dir: Path) -> list[dict[str, Any]]:
    """Loads all 103 mutational DMS cohorts from `data/processed/` parquets."""
    cdf = pd.read_parquet(data_dir / "processed" / "cohorts.parquet")
    mut_cdf = cdf[cdf["is_mutational_dms"]].reset_index(drop=True)
    tasks_dir = data_dir / "processed" / "tasks"
    tables = {p.stem: pd.read_parquet(p) for p in sorted(tasks_dir.glob("*.parquet"))}
    aa_set = set(CANONICAL_AAS)

    cohorts: list[dict[str, Any]] = []
    for _, crow in mut_cdf.iterrows():
        cid = str(crow["cohort_id"])
        sub = tables[str(crow["task_name"])]
        sub = sub[sub["parent_strain"] == cid].reset_index(drop=True)

        wt_seq: str | None = None
        edits_list: list[list[MutationEdit]] = []
        fitness_list: list[float] = []

        for seq, m_str, y_val in zip(sub["sequence"], sub["aa_mutations"], sub["dms_score_raw"]):
            if not isinstance(seq, str) or not isinstance(m_str, str) or not np.isfinite(y_val):
                continue
            toks = [t.strip() for t in re.split(r"[_,;/:\s]+", m_str.strip()) if t.strip() and t.strip().upper() != "WT"]
            if not toks:
                continue
            eds: list[MutationEdit] = []
            ok = True
            for tok in toks:
                m = MUT_RE.match(tok)
                if m is None:
                    ok = False
                    break
                w_aa, p1, m_aa = m.group(1), int(m.group(2)), m.group(3)
                p0 = p1 - 1
                if p0 < 0 or p0 >= len(seq) or seq[p0] != m_aa or w_aa not in aa_set or m_aa not in aa_set:
                    ok = False
                    break
                eds.append(MutationEdit(pos_0based=p0, wt_token=w_aa, mut_token=m_aa))
            if not ok or not eds:
                continue
            if wt_seq is None:
                chars = list(seq)
                for ed in eds:
                    chars[ed.pos_0based] = ed.wt_token
                wt_seq = "".join(chars)
            edits_list.append(eds)
            fitness_list.append(float(y_val))

        if wt_seq is None or len(edits_list) < 5:
            continue

        cohorts.append(
            {
                "cohort_id": cid,
                "wt_seq": wt_seq,
                "ligand_selfies": str(crow["ligand_selfies"]).strip(),
                "loco30_cluster": str(crow["loco30_cluster"]),
                "lolo_group": str(crow["lolo_group"]).strip(),
                "is_oncology": bool(crow["is_oncology_lopo11"]),
                "edits_list": edits_list,
                "fitness": np.asarray(fitness_list, dtype=np.float64),
            }
        )
    return cohorts


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-dir", type=Path, default=SUBFOLDER_ROOT / "data")
    parser.add_argument("--scale", type=str, choices=["8m", "35m", "150m", "650m"], default="150m")
    parser.add_argument("--single-biofm", action="store_true", help="Fine-tune single ESM-2 BioFM without ligand partner")
    parser.add_argument("--objective", type=str, choices=["logica", "logica_nokl", "logimlm"], default="logica")
    parser.add_argument("--split", type=str, choices=["loco30", "lolo"], default="loco30")
    parser.add_argument("--epochs", type=int, default=4)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--lora-rank", type=int, default=8)
    parser.add_argument("--device", type=str, default="cuda:0" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--out-dir", type=Path, default=SUBFOLDER_ROOT / "outputs" / "stage2_loo")
    parser.add_argument("--smoke-test", action="store_true")
    args = parser.parse_args()

    args.out_dir.mkdir(parents=True, exist_ok=True)
    device = torch.device(args.device)

    if args.single_biofm:
        esm_id, _ = YALE_PROTEIN_LIGAND_SCALES[args.scale]
        print(f"[Stage-2] Loading Single-BioFM LogiCA ({esm_id}, partner=None) on {device}...")
        base_model = LogiCA.from_pretrained(esm_id, partner=None, width=256).to(device).eval()
    else:
        print(f"[Stage-2] Loading Dual-BioFM LogiCA checkpoint Yale-CompBio/logica ({args.scale}) on {device}...")
        base_model = LogiCA.from_checkpoint("Yale-CompBio/logica", scale=args.scale, device=device)

    cohorts = load_mutational_dms_cohorts(args.data_dir)
    split_key = "loco30_cluster" if args.split == "loco30" else "lolo_group"
    cohorts = [c for c in cohorts if c[split_key]]
    folds = sorted({c[split_key] for c in cohorts})
    if args.smoke_test:
        folds = folds[:1]
        args.epochs = 1

    pl_loss_fn = PlackettLuceLogitLoss(top_m=32)
    kl_loss_fn = AncestralKLAnchorLoss()
    kl_weight = 0.0 if args.objective == "logica_nokl" else 0.20

    rng = np.random.default_rng(42)
    per_cohort_records: list[dict[str, Any]] = []
    t0 = time.time()

    for held_out in folds:
        train_c = [c for c in cohorts if c[split_key] != held_out]
        test_c = [c for c in cohorts if c[split_key] == held_out]
        if args.smoke_test:
            train_c = train_c[:2]
            test_c = test_c[:1]

        model = copy.deepcopy(base_model)
        ranker = VariantRanker(model)
        ref_ranker = VariantRanker(base_model)

        opt_groups = model.configure_finetuning_optimizer(
            base_lora_lr=args.lr,
            adapter_lr_multiplier=5.0,
            top_k_primary_lora=4,
            top_k_partner_lora=0 if args.single_biofm else 2,
            lora_rank=args.lora_rank,
        )
        optimizer = torch.optim.AdamW(opt_groups)
        all_ligands = [c["ligand_selfies"] for c in train_c]

        model.train()
        for _ in range(args.epochs):
            for c_idx in rng.permutation(len(train_c)):
                tc = train_c[int(c_idx)]
                eds = tc["edits_list"]
                fit = tc["fitness"]
                if len(eds) > 256:
                    sel = rng.choice(len(eds), size=256, replace=False)
                    eds = [eds[int(i)] for i in sel]
                    fit = fit[sel]

                ctx = None if args.single_biofm else tc["ligand_selfies"]
                y_t = torch.tensor(fit, dtype=torch.float32, device=device)
                s_holo, lp_holo, _ = ranker.score_cohort_differentiable(tc["wt_seq"], eds, context=ctx)

                if args.objective == "logimlm":
                    top_m = y_t >= torch.quantile(y_t, 0.75)
                    loss_main = -s_holo[top_m].mean()
                else:
                    loss_main = pl_loss_fn(s_holo, y_t)

                if args.objective.startswith("logica") and (not args.single_biofm) and len(all_ligands) > 1:
                    neg_ctx = all_ligands[(int(c_idx) + 1) % len(all_ligands)]
                    s_neg, _, _ = ranker.score_cohort_differentiable(tc["wt_seq"], eds, context=neg_ctx)
                    top_m = y_t >= torch.quantile(y_t, 0.80)
                    loss_contrast = F.softplus(s_neg[top_m] - s_holo[top_m] + 0.25).mean()
                else:
                    loss_contrast = s_holo.sum() * 0.0

                if kl_weight > 0.0:
                    with torch.no_grad():
                        _, _, ref_apo_lp = ref_ranker.score_cohort_differentiable(tc["wt_seq"], eds[:1], context=None)
                    loss_kl = kl_loss_fn(lp_holo, ref_apo_lp)
                else:
                    loss_kl = s_holo.sum() * 0.0

                loss = loss_main + 0.25 * loss_contrast + kl_weight * loss_kl
                optimizer.zero_grad()
                loss.backward()
                torch.nn.utils.clip_grad_norm_([p for p in model.parameters() if p.requires_grad], 1.0)
                optimizer.step()

        model.eval()
        with torch.no_grad():
            for ec in test_c:
                ctx = None if args.single_biofm else ec["ligand_selfies"]
                s_pred, _, _ = ranker.score_cohort_differentiable(ec["wt_seq"], ec["edits_list"], context=ctx)
                m = evaluate_ranking_metrics(ec["fitness"], s_pred.detach().cpu().numpy())
                per_cohort_records.append(
                    {
                        "cohort_id": ec["cohort_id"],
                        "fold": held_out,
                        "is_oncology": ec["is_oncology"],
                        **m,
                    }
                )

    df_out = pd.DataFrame(per_cohort_records)
    summary = {
        "scale": args.scale,
        "single_biofm": args.single_biofm,
        "objective": args.objective,
        "split": args.split,
        "n_evaluated_cohorts": int(len(df_out)),
        "spearman_mean": round(float(df_out["spearman_rho"].mean()), 4),
        "prec_at_1pct_mean": round(float(df_out["prec_at_1pct"].mean()), 3),
        "prec_at_10pct_mean": round(float(df_out["prec_at_10pct"].mean()), 3),
        "elapsed_sec": round(time.time() - t0, 2),
    }
    tag = f"{args.split}_{args.objective}_{args.scale}" + ("_single" if args.single_biofm else "") + ("_smoke" if args.smoke_test else "")
    df_out.to_csv(args.out_dir / f"per_cohort_{tag}.csv", index=False)
    (args.out_dir / f"summary_{tag}.json").write_text(json.dumps(summary, indent=2))
    print(f"[Stage-2 LOO] {summary}")


if __name__ == "__main__":
    main()
