#!/usr/bin/env python3
"""Step 4: Held-Out BioLiP `<5 Å` Cross-Modal Dependency Map Benchmark (`logica.DependencyMapper`).

Evaluates zero-shot or fine-tuned `LogiCA` checkpoints across the 175 held-out, protein-disjoint
BioLiP small-molecule crystal structures (`44,337` residue observations across `data/biolip_175/`),
measuring `<5 Å` heavy-atom contact localization AUROC and AUPR without any structural supervision,
and exports an example 2D dependency map plot via `logica.plot_dependency_map`.

Usage:
  python protein-ligand/scripts/04_evaluate_dependency_maps_biolip.py --scale 35m --device cuda:0
  python protein-ligand/scripts/04_evaluate_dependency_maps_biolip.py --scale 8m --max-complexes 5
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys
import time

import matplotlib

matplotlib.use("Agg")
import numpy as np
import pandas as pd

SUBFOLDER_ROOT = Path(__file__).resolve().parents[1]
REPO_ROOT = SUBFOLDER_ROOT.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from logica import (  # noqa: E402
    DependencyMapper,
    LogiCA,
    evaluate_contact_localization,
    plot_dependency_map,
)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--biolip-dir", type=Path, default=SUBFOLDER_ROOT / "data" / "biolip_175")
    parser.add_argument("--scale", type=str, choices=["8m", "35m", "150m", "650m"], default="35m")
    parser.add_argument("--checkpoint", type=Path, default=None, help="Optional local fine-tuned checkpoint")
    parser.add_argument("--metric", type=str, choices=["sym_kl", "l2", "jsd"], default="sym_kl")
    parser.add_argument("--device", type=str, default="cpu")
    parser.add_argument("--max-complexes", type=int, default=None)
    parser.add_argument("--out-dir", type=Path, default=SUBFOLDER_ROOT / "outputs" / "biolip_dependency")
    args = parser.parse_args()

    args.out_dir.mkdir(parents=True, exist_ok=True)
    t0 = time.time()

    if args.checkpoint is not None:
        model = LogiCA.from_checkpoint(args.checkpoint, scale=args.scale, device=args.device)
    else:
        model = LogiCA.from_checkpoint("Yale-CompBio/logica", scale=args.scale, device=args.device)

    mapper = DependencyMapper(model)

    mi_df = pd.read_parquet(args.biolip_dir / "model_inputs.parquet")
    ew_df = pd.read_parquet(args.biolip_dir / "evaluation_windows.parquet")
    rl_df = pd.read_parquet(args.biolip_dir / "residue_labels.parquet")

    if args.max_complexes is not None:
        mi_df = mi_df.head(args.max_complexes).reset_index(drop=True)

    pred_by_complex: dict[str, dict[int, float]] = {}
    first_dep_result = None
    first_cid = None

    for row in mi_df.itertuples(index=False):
        cid = str(row.complex_id)
        seq = str(row.protein_aaseq)
        sf_str = str(row.selfies).strip()
        dep = mapper.compute_cross_dependency_map(seq, sf_str, metric=args.metric, reduction="mean")
        pred_by_complex[cid] = {i + 1: float(sc) for i, sc in enumerate(dep.primary_profile)}
        if first_dep_result is None:
            first_dep_result = dep
            first_cid = cid

    win_to_mi = dict(zip(ew_df["evaluation_id"].astype(str), ew_df["model_input_id"].astype(str)))
    win_to_inter = dict(zip(ew_df["evaluation_id"].astype(str), ew_df["interaction_id"].astype(str)))

    scored_rows = []
    first_contacts_1based: list[int] = []
    for r in rl_df.itertuples(index=False):
        ev_id = str(r.evaluation_id)
        cid = win_to_mi.get(ev_id, "")
        if cid not in pred_by_complex:
            continue
        p1 = int(r.protein_sequence_position)
        if p1 in pred_by_complex[cid]:
            scored_rows.append((win_to_inter[ev_id], int(r.contact), pred_by_complex[cid][p1]))
            if cid == first_cid and int(r.contact) == 1:
                first_contacts_1based.append(p1)

    aurocs, auprs = [], []
    if scored_rows:
        sdf = pd.DataFrame(scored_rows, columns=["interaction_id", "contact", "score"])
        for _, grp in sdf.groupby("interaction_id", sort=False):
            m = evaluate_contact_localization(grp["contact"].to_numpy(), grp["score"].to_numpy())
            if np.isfinite(m["auroc"]):
                aurocs.append(m["auroc"])
                auprs.append(m["aupr"])

    if first_dep_result is not None:
        plot_dependency_map(
            first_dep_result,
            contact_residues_1based=first_contacts_1based,
            title=f"BioLiP Complex {first_cid} ({args.scale.upper()})",
            save_path=args.out_dir / f"biolip_dependency_example_{args.scale}.png",
        )

    summary = {
        "scale": args.scale,
        "metric": args.metric,
        "n_evaluated_interactions": len(aurocs),
        "auroc_mean": round(float(np.mean(aurocs)), 4) if aurocs else float("nan"),
        "auroc_sd": round(float(np.std(aurocs, ddof=1)), 4) if len(aurocs) > 1 else 0.0,
        "aupr_mean": round(float(np.mean(auprs)), 4) if auprs else float("nan"),
        "elapsed_sec": round(time.time() - t0, 2),
    }
    out_json = args.out_dir / f"biolip_summary_{args.scale}.json"
    out_json.write_text(json.dumps(summary, indent=2))
    print(f"[BioLiP Dependency] {summary}")


if __name__ == "__main__":
    main()
