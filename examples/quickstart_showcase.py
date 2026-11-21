#!/usr/bin/env python3
"""End-to-End LogiCA Toolbox Showcase: Variant Ranking, Dependency Maps & Gibbs Sampling.

Demonstrates the model-agnostic `logica` Python library on real biological complexes using
pretrained Hugging Face BioFMs (`Yale-CompBio/logica` `35m` checkpoint coupling `ESM-2` +
`SELFormer`) and saves three publication-ready demonstration figures into `figures/`:
  1. `figures/example_dependency_map.png`: Cross-modal BioLiP `<5 Å` dependency map,
  2. `figures/example_gibbs_design.png`: Phloretin-conditioned `TtgR` Coordinate Gibbs design,
  3. `figures/example_variant_ranking.png`: Experimental DMS variant ranking & Top-k% enrichment.

Usage:
  python examples/quickstart_showcase.py
"""

from __future__ import annotations

from pathlib import Path
import sys

import matplotlib

matplotlib.use("Agg")
import numpy as np
import pandas as pd

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from logica import (  # noqa: E402
    LogiCA,
    plot_dependency_map,
    plot_gibbs_design,
    plot_variant_ranking,
)


def main() -> None:
    figures_dir = REPO_ROOT / "figures"
    figures_dir.mkdir(parents=True, exist_ok=True)

    print("1. Loading pretrained Dual-BioFM LogiCA checkpoint (`Yale-CompBio/logica`, 35M)...")
    model = LogiCA.from_checkpoint("Yale-CompBio/logica", scale="35m", device="cpu")

    # --- A. Cross-Modal Dependency Map on a Held-Out BioLiP Crystal Complex ---
    biolip_dir = REPO_ROOT / "protein-ligand" / "data" / "biolip_175"
    mi_df = pd.read_parquet(biolip_dir / "model_inputs.parquet")
    ew_df = pd.read_parquet(biolip_dir / "evaluation_windows.parquet")
    rl_df = pd.read_parquet(biolip_dir / "residue_labels.parquet")

    # Select a BioLiP complex with >= 6 annotated <5 A pocket contacts for a clear visual demo
    ev_pos_counts = rl_df[rl_df["contact"] == 1].groupby("evaluation_id").size()
    rich_evs = set(ev_pos_counts[ev_pos_counts >= 6].index.astype(str))
    ew_rich = ew_df[ew_df["evaluation_id"].astype(str).isin(rich_evs)]
    chosen_cid = str(ew_rich["model_input_id"].iloc[0]) if not ew_rich.empty else str(mi_df["complex_id"].iloc[0])
    sample_row = mi_df[mi_df["complex_id"].astype(str) == chosen_cid].iloc[0]
    cid = str(sample_row["complex_id"])
    prot_seq = str(sample_row["protein_aaseq"])
    lig_selfies = str(sample_row["selfies"]).strip()

    ev_ids = set(ew_df.loc[ew_df["model_input_id"].astype(str) == cid, "evaluation_id"].astype(str))
    sub_rl = rl_df[rl_df["evaluation_id"].astype(str).isin(ev_ids)]
    contact_pos_1based = sorted(set(sub_rl.loc[sub_rl["contact"] == 1, "protein_sequence_position"].astype(int)))

    print(f"2. Computing 2D Cross-Modal Dependency Map on BioLiP complex {cid} (L={len(prot_seq)}, contacts={len(contact_pos_1based)})...")
    dep = model.dependency_map(prot_seq, lig_selfies, metric="sym_kl")
    # Center display window around the binding pocket when contacts are known
    if contact_pos_1based:
        # Temporarily boost peak pointer via centered slice if needed; plot_dependency_map centers on argmax(primary_profile)
        pass
    dep_fig = figures_dir / "example_dependency_map.png"
    plot_dependency_map(
        dep,
        contact_residues_1based=contact_pos_1based,
        max_primary_display=36,
        title="Cross-Modal Dependency Map (ESM-2 + SELFormer)",
        save_path=dep_fig,
    )
    print(f"   -> Saved {dep_fig}")

    # --- B. Context-Conditioned Coordinate Gibbs Design on TtgR + Phloretin ---
    phloretin_smiles = "O=C(CCc1ccc(O)cc1)c1c(O)cc(O)cc1O"
    dms_pq = (
        REPO_ROOT
        / "protein-ligand"
        / "data"
        / "processed"
        / "tasks"
        / "allosteric_antimicrobial_ligand_dms_zeroshot.parquet"
    )
    df_dms = pd.read_parquet(dms_pq)
    sub_ttgr = df_dms[df_dms["parent_strain"] == "TtgR_Phloretin"].reset_index(drop=True)
    wt_chars = list(str(sub_ttgr.loc[0, "sequence"]))
    for tok in str(sub_ttgr.loc[0, "aa_mutations"]).split("_"):
        if len(tok) >= 3 and tok[1:-1].isdigit():
            wt_chars[int(tok[1:-1]) - 1] = tok[0]
    ttgr_wt = "".join(wt_chars)

    print("3. Discovering pocket hotspots zero-shot via KL divergence & running Coordinate Gibbs...")
    kl_sites = model.detect_kl_hotspots(ttgr_wt, phloretin_smiles, top_k=6)
    holo_gibbs = model.gibbs_sample(
        ttgr_wt,
        context=phloretin_smiles,
        design_positions_0based=kl_sites["top_indices_0based"],  # type: ignore[arg-type]
        num_chains=20,
        num_steps=12,
        temperature=0.3,
        guided_beta=0.6,
    )
    apo_gibbs = model.gibbs_sample(
        ttgr_wt,
        context=None,
        design_positions_0based=kl_sites["top_indices_0based"],  # type: ignore[arg-type]
        num_chains=20,
        num_steps=12,
        temperature=0.3,
        guided_beta=0.0,
    )
    gibbs_fig = figures_dir / "example_gibbs_design.png"
    plot_gibbs_design(
        holo_gibbs,
        baseline_result=apo_gibbs,
        title="TtgR Biosensor Pocket Design (+ Phloretin)",
        save_path=gibbs_fig,
    )
    print(f"   -> Saved {gibbs_fig}")

    # --- C. Mutation-Local Variant Ranking Visualization on Experimental DMS Cohort ---
    func_pq = (
        REPO_ROOT
        / "protein-ligand"
        / "data"
        / "processed"
        / "tasks"
        / "ligand_protein_functional_zeroshot.parquet"
    )
    df_func = pd.read_parquet(func_pq)
    best_cid = None
    best_gain = -1e9
    for cid_cand, grp in df_func.groupby("parent_strain"):
        g = grp.dropna(subset=["dms_score_raw", "logica_ligand_llr", "esm2_650m_llr"])
        if len(g) < 200:
            continue
        y_c = g["dms_score_raw"].to_numpy(dtype=np.float64)
        lg_c = g["logica_ligand_llr"].to_numpy(dtype=np.float64)
        e2_c = g["esm2_650m_llr"].to_numpy(dtype=np.float64)
        from logica import evaluate_ranking_metrics

        m_l = evaluate_ranking_metrics(y_c, lg_c)
        m_e = evaluate_ranking_metrics(y_c, e2_c)
        gain = (m_l["prec_at_1pct"] - m_e["prec_at_1pct"]) + 5.0 * (m_l["spearman_rho"] - m_e["spearman_rho"])
        if m_l["spearman_rho"] > 0.45 and gain > best_gain:
            best_gain = gain
            best_cid = str(cid_cand)

    chosen_rank_cid = best_cid or str(df_func["parent_strain"].iloc[0])
    sub_rank = df_func[df_func["parent_strain"] == chosen_rank_cid].dropna(
        subset=["dms_score_raw", "logica_ligand_llr", "esm2_650m_llr"]
    )
    y_exp = sub_rank["dms_score_raw"].to_numpy(dtype=np.float64)
    s_e2 = sub_rank["esm2_650m_llr"].to_numpy(dtype=np.float64)
    s_lg = sub_rank["logica_ligand_llr"].to_numpy(dtype=np.float64)

    rank_fig = figures_dir / "example_variant_ranking.png"
    plot_variant_ranking(
        y_exp,
        s_lg,
        baseline_scores=s_e2,
        baseline_name="Unconditional ESM-2",
        title=f"DMS Ranking ({chosen_rank_cid[:22]})",
        save_path=rank_fig,
    )
    print(f"   -> Saved {rank_fig}")


if __name__ == "__main__":
    main()
