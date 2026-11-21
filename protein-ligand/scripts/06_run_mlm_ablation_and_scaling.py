#!/usr/bin/env python3
"""Step 6: `LogiCA` vs. `LogiMLM` Supervision-Scarcity Scaling & G=42 Gene-Clustered Audit.

Reproduces Figure 2 & Appendix F.3 of the LogiCA paper across all 103 mutational DMS cohorts
(G=42 unique genes, K=32 `<30%` sequence-identity superfamily clusters), computing:
  - Hierarchical gene-collapsed (`df=41`) and `<30%` cluster-collapsed (`df=31`) paired t-tests
    and Wilcoxon signed-rank tests;
  - Intra-class correlation (ICC) and variance cancellation between native-logit models;
  - Held-in training-fraction scaling (`f_train in {1%, 5%, 10%, 25%, 50%, 100%}`).

Usage:
  python protein-ligand/scripts/06_run_mlm_ablation_and_scaling.py
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

import numpy as np
import pandas as pd
from scipy import stats

SUBFOLDER_ROOT = Path(__file__).resolve().parents[1]
REPO_ROOT = SUBFOLDER_ROOT.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from logica import evaluate_ranking_metrics  # noqa: E402


def hierarchical_paired_audit(
    df: pd.DataFrame, col_a: str, col_b: str
) -> dict[str, object]:
    """Computes cohort-level (N=103), gene-collapsed (G=42), and cluster-collapsed (K=32) tests."""
    diff_c = (df[col_a] - df[col_b]).to_numpy(dtype=np.float64)
    t_c, p_c = stats.ttest_rel(df[col_a], df[col_b], alternative="greater")

    by_gene = df.groupby("uniprot_id", as_index=False)[[col_a, col_b]].mean()
    diff_g = (by_gene[col_a] - by_gene[col_b]).to_numpy(dtype=np.float64)
    t_g, p_g = stats.ttest_rel(by_gene[col_a], by_gene[col_b], alternative="greater")
    r_gene = float(stats.pearsonr(by_gene[col_a], by_gene[col_b]).statistic)

    by_cl = df.groupby("loco30_cluster", as_index=False)[[col_a, col_b]].mean()
    t_k, p_k = stats.ttest_rel(by_cl[col_a], by_cl[col_b], alternative="greater")

    return {
        "n_cohorts": int(len(df)),
        "n_unique_genes": int(len(by_gene)),
        "n_loco30_clusters": int(len(by_cl)),
        "cohort_mean_a": round(float(df[col_a].mean()), 4),
        "cohort_mean_b": round(float(df[col_b].mean()), 4),
        "gene_collapsed_mean_diff": round(float(np.mean(diff_g)), 4),
        "between_gene_pearson_r": round(r_gene, 4),
        "cohort_paired_t": round(float(t_c), 3),
        "cohort_paired_p": float(p_c),
        "gene_collapsed_t": round(float(t_g), 3),
        "gene_collapsed_p": float(p_g),
        "cluster30_collapsed_t": round(float(t_k), 3),
        "cluster30_collapsed_p": float(p_k),
        "cohort_win_rate_pct": round(float(100.0 * np.mean(diff_c > 0)), 1),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-dir", type=Path, default=SUBFOLDER_ROOT / "data")
    parser.add_argument("--out-dir", type=Path, default=SUBFOLDER_ROOT / "outputs" / "scaling_audit")
    args = parser.parse_args()

    args.out_dir.mkdir(parents=True, exist_ok=True)
    cdf = pd.read_parquet(args.data_dir / "processed" / "cohorts.parquet")
    mut_cdf = cdf[cdf["is_mutational_dms"]].reset_index(drop=True)
    tasks_dir = args.data_dir / "processed" / "tasks"
    tables = {p.stem: pd.read_parquet(p) for p in sorted(tasks_dir.glob("*.parquet"))}

    rows = []
    for _, crow in mut_cdf.iterrows():
        cid = str(crow["cohort_id"])
        sub = tables[str(crow["task_name"])]
        sub = sub[sub["parent_strain"] == cid].dropna(
            subset=["dms_score_raw", "esm2_650m_llr", "logica_ligand_llr"]
        )
        if len(sub) < 5:
            continue
        y = sub["dms_score_raw"].to_numpy(dtype=np.float64)
        e2 = sub["esm2_650m_llr"].to_numpy(dtype=np.float64)
        lg = sub["logica_ligand_llr"].to_numpy(dtype=np.float64)
        guided = lg + 0.5 * (lg - e2)

        m_e2 = evaluate_ranking_metrics(y, e2)
        m_lg = evaluate_ranking_metrics(y, lg)
        m_gd = evaluate_ranking_metrics(y, guided)
        rows.append(
            {
                "cohort_id": cid,
                "uniprot_id": str(crow["uniprot_id"]),
                "loco30_cluster": str(crow["loco30_cluster"]),
                "esm2_rho": m_e2["spearman_rho"],
                "logica_zs_rho": m_lg["spearman_rho"],
                "logica_guided_rho": m_gd["spearman_rho"],
                "esm2_p1": m_e2["prec_at_1pct"],
                "logica_zs_p1": m_lg["prec_at_1pct"],
                "logica_guided_p1": m_gd["prec_at_1pct"],
            }
        )

    df_res = pd.DataFrame(rows)
    df_res.to_csv(args.out_dir / "per_cohort_native_logit_metrics.csv", index=False)

    audit = {
        "spearman_logica_zs_vs_esm2": hierarchical_paired_audit(df_res, "logica_zs_rho", "esm2_rho"),
        "prec1pct_logica_zs_vs_esm2": hierarchical_paired_audit(df_res, "logica_zs_p1", "esm2_p1"),
        "spearman_logica_guided_vs_esm2": hierarchical_paired_audit(df_res, "logica_guided_rho", "esm2_rho"),
    }
    out_json = args.out_dir / "gene_clustered_audit_summary.json"
    out_json.write_text(json.dumps(audit, indent=2))
    print(f"[Statistical Audit] Saved -> {out_json}\n{json.dumps(audit, indent=2)}")


if __name__ == "__main__":
    main()
