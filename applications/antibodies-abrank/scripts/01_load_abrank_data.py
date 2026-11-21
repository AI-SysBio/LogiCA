#!/usr/bin/env python3
"""Step 1: Fetch & Verify AbRank v2 Data from Kaggle (or a Local `AI-SysBio/AbRank` Checkout).

Verifies all master assay tables and the 4 Core AbRank Community Suites:
  - `AbRank-Maturation`: 22-Cluster LOACO local edits (`12,267` L1-L3 pairs)
  - `AbRank-Discovery` : 25 unfiltered continuous panels + VHH nanobody Kd (`813` pairs)
  - `AbRank-Generative`: 1,354 wet-lab generative AI binder designs (`5,069` pairs)
  - `AbRank-Transfer`  : Viral IC50, RBD Escape, and Cross-Modal readouts

Usage:
  # Auto-fetch public Kaggle dataset `aurlienplissier/abrank` (`pip install kagglehub`)
  python applications/antibodies-abrank/scripts/01_load_abrank_data.py --kaggle-dataset aurlienplissier/abrank

  # Or point directly to a local AI-SysBio/AbRank checkout / ABRANK_DATA_DIR
  python applications/antibodies-abrank/scripts/01_load_abrank_data.py --abrank-dir /path/to/AbRank/data/benchmark
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

import pandas as pd

SCRIPT_DIR = Path(__file__).resolve().parent
SUBFOLDER_ROOT = SCRIPT_DIR.parent
if str(SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR))

from abrank_loader import DEFAULT_KAGGLE_DATASET, resolve_abrank_dir  # noqa: E402


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--abrank-dir", type=Path, default=None, help="Local path to AbRank `data/benchmark/`")
    parser.add_argument(
        "--kaggle-dataset",
        type=str,
        default=DEFAULT_KAGGLE_DATASET,
        help=f"Kaggle dataset slug (default: {DEFAULT_KAGGLE_DATASET})",
    )
    parser.add_argument("--out-json", type=Path, default=None, help="Optional JSON inventory output path")
    args = parser.parse_args()

    bdir = resolve_abrank_dir(args.abrank_dir, kaggle_dataset=args.kaggle_dataset)
    print(f"[AbRank Loader] Resolved benchmark directory: {bdir}\n")

    inventory: dict[str, dict] = {"benchmark_dir": str(bdir), "tables": {}, "splits": {}}

    # 1. Master assay tables
    print("Master Assay Tables:")
    for fname in ("assays_kd.tsv", "assays_vhh.tsv", "assays_cross_modal.tsv", "assays_transfer.tsv"):
        p = bdir / fname
        if not p.exists():
            print(f"  [MISSING] {fname}")
            continue
        df = pd.read_csv(p, sep="\t", usecols=["AbAgID", "Ag_group", "Dataset"], low_memory=False)
        info = {
            "rows": int(len(df)),
            "unique_abag": int(df["AbAgID"].nunique()),
            "unique_ag_groups": int(df["Ag_group"].nunique()),
            "datasets": int(df["Dataset"].nunique()),
        }
        inventory["tables"][fname] = info
        print(
            f"  [OK] {fname:24s}  rows={info['rows']:>7,d}  "
            f"Ag_groups={info['unique_ag_groups']:>3d}  sources={info['datasets']:>2d}"
        )

    # 2. Core suite splits
    split_files = [
        ("unseen_ag/train", "splits/unseen_ag/train.tsv"),
        ("unseen_ag/val", "splits/unseen_ag/val.tsv"),
        ("unseen_ag/test", "splits/unseen_ag/test.tsv"),
        ("unseen_ag/test_candidates", "splits/unseen_ag/test_candidates.tsv"),
        ("generative/pairs", "splits/generative/generative_pairs.tsv"),
        ("generative/panels", "splits/generative/generative_panels.tsv"),
        ("transfer/ic50", "splits/transfer/ic50_test.tsv"),
        ("transfer/escape", "splits/transfer/escape_test.tsv"),
        ("cross_modal/modality", "splits/cross_modal/modality_test.tsv"),
        ("vhh/ranking", "splits/vhh/vhh_ranking_test.tsv"),
    ]
    print("\nCore Suite Splits:")
    for key, rel in split_files:
        p = bdir / rel
        if not p.exists():
            print(f"  [MISSING] {rel}")
            continue
        df = pd.read_csv(p, sep="\t", low_memory=False)
        info = {"rows": int(len(df))}
        if "label" in df.columns:
            vc = df["label"].value_counts().to_dict()
            info["labels"] = {str(k): int(v) for k, v in sorted(vc.items())}
            lbl_str = f"labels(-1/0/+1)={vc.get(-1,0)}/{vc.get(0,0)}/{vc.get(1,0)}"
        else:
            lbl_str = f"panels={df.groupby(['Ag_group','Dataset']).ngroups if {'Ag_group','Dataset'} <= set(df.columns) else 'N/A'}"
        inventory["splits"][key] = info
        print(f"  [OK] {key:26s}  rows={info['rows']:>6,d}  {lbl_str}")

    if args.out_json is not None:
        args.out_json.parent.mkdir(parents=True, exist_ok=True)
        args.out_json.write_text(json.dumps(inventory, indent=2) + "\n", encoding="utf-8")
        print(f"\nSaved inventory -> {args.out_json}")


if __name__ == "__main__":
    main()
