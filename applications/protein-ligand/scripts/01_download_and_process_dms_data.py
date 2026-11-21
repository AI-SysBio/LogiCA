#!/usr/bin/env python3
"""Step 1: Fetch Raw Protein–Ligand DMS Archives, Verify Splits & Score with LogiCA.

Provides three reproducible operations for the 137-cohort, 561,059-variant benchmark:
  1. Default verification: Audits `data/processed/cohorts.parquet`, `data/processed/tasks/*.parquet`,
     and `data/biolip_175/*.parquet` (all included in the repository for immediate execution) and
     writes `outputs/01_dataset_summary.json`.
  2. `--download-raw`: Downloads primary public raw DMS & enzyme kinetics archives (ProteinGym v1.3,
     TtgR Zenodo, TYK2 Zenodo, Fraser MET, Kim EGFR, Coelho Sanger, Landry ERG11/FKS/FCY1, CatPred,
     Goldman, UniKP) into `data/raw/`.
  3. `--score-logica`: Loads the official pretrained `Yale-CompBio/logica` (`8m`, `35m`, `150m`,
     or `650m`) checkpoint via the model-agnostic `logica` Python library (`LogiCA.from_checkpoint`)
     and computes stride-16 masked marginal mutation scores across cohorts!

Usage:
  python applications/protein-ligand/scripts/01_download_and_process_dms_data.py
  python applications/protein-ligand/scripts/01_download_and_process_dms_data.py --download-raw
  python applications/protein-ligand/scripts/01_download_and_process_dms_data.py --score-logica --scale 150m --max-cohorts 3
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys
import urllib.request

import numpy as np
import pandas as pd

SUBFOLDER_ROOT = Path(__file__).resolve().parents[1]
REPO_ROOT = SUBFOLDER_ROOT.parents[1]
SCRIPT_DIR = Path(__file__).resolve().parent
for _p in (str(REPO_ROOT), str(SCRIPT_DIR)):
    if _p not in sys.path:
        sys.path.insert(0, _p)

from logica import LogiCA, evaluate_ranking_metrics  # noqa: E402
from pl_data_loader import (  # noqa: E402
    DEFAULT_KAGGLE_DATASET,
    list_task_names,
    load_cohorts,
    load_task_df,
    resolve_biolip_dir,
    resolve_pl_data_dir,
)

PRIMARY_PUBLIC_SOURCES: dict[str, str] = {
    "proteingym_dms_v1_3.zip": (
        "https://marks.hms.harvard.edu/proteingym/ProteinGym_v1.3/DMS_ProteinGym_substitutions.zip"
    ),
    "proteingym_DMS_substitutions.csv": (
        "https://raw.githubusercontent.com/OATML-Markslab/ProteinGym/main/reference_files/DMS_substitutions.csv"
    ),
    "ttgr_raman_zenodo_10818566.json": "https://zenodo.org/api/records/10818566",
    "tyk2_elliott_zenodo_15794689.json": "https://zenodo.org/api/records/15794689",
    "met_fraser_dms.zip": (
        "https://codeload.github.com/fraser-lab/MET_kinase_Inhibitor_DMS/zip/refs/heads/main"
    ),
    "egfr_kim_sge.zip": "https://codeload.github.com/oreolic/SGE_EGFR/zip/refs/heads/main",
    "coelho_sanger_res1.zip": (
        "https://codeload.github.com/MatthewACoelho/Res1_analysis/zip/refs/heads/main"
    ),
    "landry_erg11_fks.zip": (
        "https://codeload.github.com/Landrylab/Durand_et_al_2026/zip/refs/heads/main"
    ),
    "landry_fcy1.zip": (
        "https://codeload.github.com/Landrylab/Aube_et_al_2025_FCY1promoter/zip/refs/heads/main"
    ),
    "goldman_enzyme_datasets.zip": (
        "https://codeload.github.com/samgoldman97/enzyme-datasets/zip/refs/heads/main"
    ),
    "luo_unikp.zip": "https://codeload.github.com/Luo-SynBioLab/UniKP/zip/refs/heads/main",
}


def download_raw_sources(raw_dir: Path) -> None:
    raw_dir.mkdir(parents=True, exist_ok=True)
    for filename, url in PRIMARY_PUBLIC_SOURCES.items():
        dest = raw_dir / filename
        if dest.is_file() and dest.stat().st_size > 0:
            print(f"[Skip] {dest.name} already downloaded ({dest.stat().st_size:,} bytes)")
            continue
        print(f"[Download] {url} -> {dest}")
        req = urllib.request.Request(url, headers={"User-Agent": "LogiCA-Benchmark/1.0"})
        with urllib.request.urlopen(req, timeout=180) as resp, open(dest, "wb") as out:
            out.write(resp.read())


def summarize_benchmark(data_dir: Path) -> dict[str, object]:
    cdf = load_cohorts(data_dir)
    mut_cdf = cdf[cdf["is_mutational_dms"]]

    task_counts: dict[str, dict[str, int]] = {}
    total_rows = 0
    for tname in list_task_names(data_dir):
        tdf = load_task_df(data_dir, tname, include_baselines=False)
        n_rows = int(len(tdf))
        total_rows += n_rows
        task_counts[tname] = {
            "cohorts": int(tdf["parent_strain"].nunique()),
            "variants": n_rows,
        }

    biolip_summary: dict[str, int] = {}
    try:
        biolip_dir = resolve_biolip_dir(data_dir)
        rl = pd.read_parquet(biolip_dir / "residue_labels.parquet")
        ew = pd.read_parquet(biolip_dir / "evaluation_windows.parquet")
        biolip_summary = {
            "interactions": int(ew["interaction_id"].nunique()),
            "evaluation_windows": int(len(ew)),
            "residue_observations": int(len(rl)),
            "positive_contacts_lt_5A": int(rl["contact"].sum()),
        }
    except FileNotFoundError:
        pass

    return {
        "total_cohorts": int(len(cdf)),
        "mutational_dms_cohorts": int(len(mut_cdf)),
        "wt_kinetics_cohorts": int(len(cdf) - len(mut_cdf)),
        "total_variants": total_rows,
        "loco30_mutational_clusters": int(mut_cdf["loco30_cluster"].nunique()),
        "unique_mutational_genes": int(mut_cdf["uniprot_id"].nunique()),
        "oncology_loo_cohorts": int(mut_cdf["is_oncology_lopo11"].sum()),
        "task_families": task_counts,
        "biolip_175": biolip_summary,
    }


def score_cohorts_with_logica(
    data_dir: Path,
    scale: str = "150m",
    device: str = "cpu",
    max_cohorts: int | None = None,
) -> list[dict[str, object]]:
    """Scores DMS cohorts live using `LogiCA.from_checkpoint('Yale-CompBio/logica', scale=...)`."""
    print(f"[LogiCA] Loading pretrained checkpoint Yale-CompBio/logica ({scale}) on {device}...")
    model = LogiCA.from_checkpoint("Yale-CompBio/logica", scale=scale, device=device)

    cdf = load_cohorts(data_dir)
    mut_cdf = cdf[cdf["is_mutational_dms"]].reset_index(drop=True)
    if max_cohorts is not None:
        mut_cdf = mut_cdf.head(max_cohorts)

    tasks_cache: dict[str, pd.DataFrame] = {}
    scored_summaries: list[dict[str, object]] = []

    for _, crow in mut_cdf.iterrows():
        tname = str(crow["task_name"])
        cid = str(crow["cohort_id"])
        if tname not in tasks_cache:
            tasks_cache[tname] = load_task_df(data_dir, tname, include_baselines=False)
        sub = tasks_cache[tname]
        sub = sub[sub["parent_strain"] == cid].dropna(subset=["sequence", "aa_mutations", "dms_score_raw"])
        sub = sub[sub["aa_mutations"].astype(str).str.upper() != "WT"].reset_index(drop=True)
        if len(sub) < 5:
            continue

        # Reconstruct wild-type sequence from first valid variant
        first_seq = list(str(sub.loc[0, "sequence"]))
        for tok in str(sub.loc[0, "aa_mutations"]).replace(",", "_").replace(":", "_").split("_"):
            tok = tok.strip()
            if len(tok) >= 3 and tok[1:-1].isdigit():
                first_seq[int(tok[1:-1]) - 1] = tok[0]
        wt_seq = "".join(first_seq)
        ligand_ctx = str(crow["ligand_selfies"] or crow["ligand_smiles"]).strip()

        ranking = model.rank_variants(
            wild_type=wt_seq,
            variants=sub["sequence"].astype(str).tolist(),
            context=ligand_ctx,
            mode="stride_masked",
            stride=16,
            component="holo",
        )
        y_true = sub["dms_score_raw"].to_numpy(dtype=np.float64)
        m_holo = evaluate_ranking_metrics(y_true, ranking.holo_scores)
        m_apo = evaluate_ranking_metrics(y_true, ranking.apo_scores)
        rec = {
            "cohort_id": cid,
            "n_variants": int(len(sub)),
            "logica_spearman": round(m_holo["spearman_rho"], 4),
            "esm2_apo_spearman": round(m_apo["spearman_rho"], 4),
            "logica_prec_at_1pct": round(m_holo["prec_at_1pct"], 3),
        }
        print(f"  {cid}: LogiCA rho={rec['logica_spearman']:+.4f} (P@1%={rec['logica_prec_at_1pct']:.2f}x) vs ESM-2 rho={rec['esm2_apo_spearman']:+.4f}")
        scored_summaries.append(rec)

    return scored_summaries


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-dir", type=Path, default=None, help="Local dataset dir (auto-fetches Kaggle if omitted)")
    parser.add_argument("--kaggle-dataset", type=str, default=DEFAULT_KAGGLE_DATASET)
    parser.add_argument("--out-dir", type=Path, default=SUBFOLDER_ROOT / "outputs")
    parser.add_argument("--download-raw", action="store_true")
    parser.add_argument("--score-logica", action="store_true")
    parser.add_argument("--scale", type=str, choices=["8m", "35m", "150m", "650m"], default="150m")
    parser.add_argument("--device", type=str, default="cpu")
    parser.add_argument("--max-cohorts", type=int, default=None)
    args = parser.parse_args()

    args.out_dir.mkdir(parents=True, exist_ok=True)
    if args.download_raw:
        download_raw_sources((args.data_dir or (SUBFOLDER_ROOT / "data")) / "raw")

    data_dir = resolve_pl_data_dir(args.data_dir, kaggle_dataset=args.kaggle_dataset)
    summary = summarize_benchmark(data_dir)
    if args.score_logica:
        summary["live_checkpoint_scoring"] = score_cohorts_with_logica(
            data_dir, scale=args.scale, device=args.device, max_cohorts=args.max_cohorts
        )

    out_json = args.out_dir / "01_dataset_summary.json"
    out_json.write_text(json.dumps(summary, indent=2))
    print(
        f"[Protein-Ligand Step 01] Verified {summary['total_cohorts']} cohorts "
        f"({summary['total_variants']:,} variants). Saved summary -> {out_json}"
    )


if __name__ == "__main__":
    main()
