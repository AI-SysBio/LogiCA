#!/usr/bin/env python3
"""Step 5: Context-Conditioned Coordinate Gibbs Sampling & Zero-Shot KL Pocket Discovery.

Demonstrates both modes of `logica.gibbs`:
  1. Retrospective landscape Gibbs evaluation across all 103 mutational DMS cohorts
     (Within-Site Coordinate Gibbs, Global Boltzmann-Gibbs, and 12-Step Coordinate MCMC Gibbs);
  2. Live sequence-level Coordinate Gibbs pocket design + annotation-free zero-shot KL hotspot
     discovery (`model.detect_kl_hotspots` + `model.gibbs_sample`) on a pretrained
     `Yale-CompBio/logica` checkpoint (`--live-design`).

Usage:
  python applications/protein-ligand/scripts/05_run_gibbs_sampling.py
  python applications/protein-ligand/scripts/05_run_gibbs_sampling.py --live-design --scale 35m
"""

from __future__ import annotations

import argparse
from collections import defaultdict
import json
from pathlib import Path
import sys
import time

import matplotlib

matplotlib.use("Agg")
import numpy as np
import pandas as pd

SUBFOLDER_ROOT = Path(__file__).resolve().parents[1]
REPO_ROOT = SUBFOLDER_ROOT.parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from logica import (  # noqa: E402
    LogiCA,
    evaluate_landscape_gibbs,
    plot_gibbs_design,
)


def evaluate_all_dms_cohorts_gibbs(data_dir: Path) -> dict[str, object]:
    """Evaluates all 3 Gibbs protocols across the 103 mutational DMS cohorts."""
    cdf = pd.read_parquet(data_dir / "processed" / "cohorts.parquet")
    mut_cdf = cdf[cdf["is_mutational_dms"]].reset_index(drop=True)
    tasks_dir = data_dir / "processed" / "tasks"
    tables = {p.stem: pd.read_parquet(p) for p in sorted(tasks_dir.glob("*.parquet"))}

    metrics_by_model: dict[str, list[dict[str, object]]] = {"ESM2_650M": [], "LogiCA_650M": [], "LogiCA_Guided": []}

    for idx, crow in mut_cdf.iterrows():
        cid = str(crow["cohort_id"])
        sub = tables[str(crow["task_name"])]
        sub = sub[sub["parent_strain"] == cid].dropna(
            subset=["dms_score_raw", "esm2_650m_llr", "logica_ligand_llr", "aa_mutations"]
        ).reset_index(drop=True)
        if len(sub) < 10:
            continue

        site_groups: dict[int, list[int]] = defaultdict(list)
        for r_i, m_str in enumerate(sub["aa_mutations"].astype(str)):
            toks = [t for t in m_str.replace(",", "_").split("_") if len(t) >= 3 and t[1:-1].isdigit()]
            if len(toks) == 1:
                site_groups[int(toks[0][1:-1])].append(r_i)

        y = sub["dms_score_raw"].to_numpy(dtype=np.float64)
        e2 = sub["esm2_650m_llr"].to_numpy(dtype=np.float64)
        lg = sub["logica_ligand_llr"].to_numpy(dtype=np.float64)
        e2_z = (e2 - np.mean(e2)) / max(float(np.std(e2)), 1e-8)
        lg_z = (lg - np.mean(lg)) / max(float(np.std(lg)), 1e-8)
        guided = lg_z + 0.5 * (lg_z - e2_z)

        for name, vec in (("ESM2_650M", e2_z), ("LogiCA_650M", lg_z), ("LogiCA_Guided", guided)):
            gb = evaluate_landscape_gibbs(y, vec, site_groups, seed=42 + int(idx))
            metrics_by_model[name].append(gb)

    summary: dict[str, object] = {}
    for name, rows in metrics_by_model.items():
        summary[name] = {
            "n_cohorts": len(rows),
            "within_site_top10_enrichment": round(float(np.mean([r["within_site_gibbs_top10_enrichment"] for r in rows])), 3),
            "boltzmann_top10_enrichment": round(float(np.mean([r["boltzmann_gibbs_top10_enrichment"] for r in rows])), 3),
            "mcmc_12step_top10_enrichment": round(float(np.mean([r["mcmc_final_top10_enrichment"] for r in rows])), 3),
            "mcmc_12step_mean_z": round(float(np.mean([r["mcmc_final_z"] for r in rows])), 4),
        }
    return summary


def run_live_pocket_gibbs_showcase(scale: str, device: str, out_dir: Path) -> dict[str, object]:
    """Runs live zero-shot KL hotspot discovery + 12-step Coordinate Gibbs on TtgR + Phloretin."""
    phloretin_smiles = "O=C(CCc1ccc(O)cc1)c1c(O)cc(O)cc1O"
    df_all = pd.read_parquet(
        SUBFOLDER_ROOT / "data" / "processed" / "tasks" / "allosteric_antimicrobial_ligand_dms_zeroshot.parquet"
    )
    sub = df_all[df_all["parent_strain"] == "TtgR_Phloretin"].reset_index(drop=True)
    wt_chars = list(str(sub.loc[0, "sequence"]))
    for tok in str(sub.loc[0, "aa_mutations"]).split("_"):
        if len(tok) >= 3 and tok[1:-1].isdigit():
            wt_chars[int(tok[1:-1]) - 1] = tok[0]
    ttgr_wt = "".join(wt_chars)

    model = LogiCA.from_checkpoint("Yale-CompBio/logica", scale=scale, device=device)
    kl_hotspots = model.detect_kl_hotspots(ttgr_wt, phloretin_smiles, top_k=6)

    holo_design = model.gibbs_sample(
        ttgr_wt,
        context=phloretin_smiles,
        design_positions_0based=kl_hotspots["top_indices_0based"],  # type: ignore[arg-type]
        num_chains=16,
        num_steps=12,
        temperature=0.3,
        guided_beta=0.5,
    )
    apo_design = model.gibbs_sample(
        ttgr_wt,
        context=None,
        design_positions_0based=kl_hotspots["top_indices_0based"],  # type: ignore[arg-type]
        num_chains=16,
        num_steps=12,
        temperature=0.3,
        guided_beta=0.0,
    )
    fig_path = out_dir / f"ttgr_phloretin_gibbs_{scale}.png"
    plot_gibbs_design(
        holo_design,
        baseline_result=apo_design,
        title=f"TtgR + Phloretin Coordinate Gibbs ({scale.upper()})",
        save_path=fig_path,
    )
    return {
        "scale": scale,
        "kl_discovered_positions_1based": kl_hotspots["top_positions_1based"],
        "effective_sample_size": round(holo_design.effective_sample_size, 2),
        "top_3_designs": [
            {"sequence": s, "score": round(sc, 4)} for s, sc in holo_design.top_designs(k=3)
        ],
        "figure": str(fig_path),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-dir", type=Path, default=SUBFOLDER_ROOT / "data")
    parser.add_argument("--live-design", action="store_true")
    parser.add_argument("--scale", type=str, choices=["8m", "35m", "150m", "650m"], default="35m")
    parser.add_argument("--device", type=str, default="cpu")
    parser.add_argument("--out-dir", type=Path, default=SUBFOLDER_ROOT / "outputs" / "gibbs")
    args = parser.parse_args()

    args.out_dir.mkdir(parents=True, exist_ok=True)
    t0 = time.time()
    payload: dict[str, object] = {
        "landscape_benchmark_103_dms": evaluate_all_dms_cohorts_gibbs(args.data_dir),
    }
    if args.live_design:
        payload["live_ttgr_phloretin_design"] = run_live_pocket_gibbs_showcase(
            args.scale, args.device, args.out_dir
        )
    payload["elapsed_sec"] = round(time.time() - t0, 2)

    out_json = args.out_dir / "gibbs_summary.json"
    out_json.write_text(json.dumps(payload, indent=2))
    print(f"[Protein-Ligand Gibbs] Saved summary -> {out_json}\n{json.dumps(payload, indent=2)}")


if __name__ == "__main__":
    main()
