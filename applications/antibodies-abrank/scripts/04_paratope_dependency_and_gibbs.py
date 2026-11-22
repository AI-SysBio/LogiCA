#!/usr/bin/env python3
"""Step 4: Antigen -> Paratope Dependency Map & CDR-H3 Coordinate Gibbs Affinity Maturation.

Demonstrates two head-free emergent capabilities of `AbLang2-Paired` + `ESM-2` `LogiCA` on a
real antibody–antigen complex from AbRank (or custom `--vh`, `--vl`, `--ag` sequences):
  1. Zero-Shot Cross-Modal Dependency Map (`model.dependency_map`) + KL Paratope Hotspot
     Discovery (`model.detect_kl_hotspots`) across the antibody variable domain (`VH|VL`) and
     antigen epitope window.
  2. Decoder-Free Antigen-Conditioned Coordinate Gibbs Affinity Maturation (`model.gibbs_sample`)
     optimizing the CDR-H3 paratope loop directly in `AbLang2`'s native 20-AA logit space.

Usage:
  python applications/antibodies-abrank/scripts/04_paratope_dependency_and_gibbs.py \\
      --checkpoint applications/antibodies-abrank/outputs/ablang_esm_logica/ablang_esm_logica.pt \\
      --kaggle-dataset aurlienplissier/abrank

  python applications/antibodies-abrank/scripts/04_paratope_dependency_and_gibbs.py --smoke-test
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

import pandas as pd
import torch

SCRIPT_DIR = Path(__file__).resolve().parent
SUBFOLDER_ROOT = SCRIPT_DIR.parent
REPO_ROOT = SUBFOLDER_ROOT.parents[1]
for p in (str(REPO_ROOT), str(SCRIPT_DIR)):
    if p not in sys.path:
        sys.path.insert(0, p)

from abrank_loader import (  # noqa: E402
    DEFAULT_KAGGLE_DATASET,
    detect_cdrh3_positions_0based,
    format_antibody_sequence,
    resolve_abrank_dir,
)
from logica import LogiCA, plot_dependency_map, plot_gibbs_design  # noqa: E402

# Default Trastuzumab (HER2) Fv + HER2 Domain IV epitope snippet if no AbRank path is provided
DEFAULT_VH = (
    "EVQLVESGGGLVQPGGSLRLSCAASGFNIKDTYIHWVRQAPGKGLEWVARIYPTNGYTRYADSVKGRFTISADTSKNTAYLQMNSLRAEDTAVYYCSRWGGDGFYAMDYWGQGTLVTVSS"
)
DEFAULT_VL = (
    "DIQMTQSPSSLSASVGDRVTITCRASQDVNTAVAWYQQKPGKAPKLLIYSASFLYSGVPSRFSGSRSGTDFTLTISSLQPEDFATYYCQQHYTTPPTFGQGTKVEIK"
)
DEFAULT_AG = (
    "CHPECQPQNGSVTCFGPEADQCVACAHYKDPPFCVARCPSGVKPDLSYMPIWKFPDEEGACQPCPINCTHSCVDLDDKGCPAEQRASPLT"
)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--checkpoint",
        type=str,
        default="Yale-CompBio/logica-abag",
        help="Trained LogiCA `.pt` path or Hugging Face repo (default: `Yale-CompBio/logica-abag`; pass `none` for untrained)",
    )
    parser.add_argument("--scale", type=str, default="esm35m", help="Checkpoint scale when loading from Hugging Face Hub")
    parser.add_argument("--ab-model", type=str, default="ablang2-paired")
    parser.add_argument("--ag-model", type=str, default="facebook/esm2_t12_35M_UR50D")
    parser.add_argument("--abrank-dir", type=Path, default=None, help="Optional local AbRank `data/benchmark/` path")
    parser.add_argument("--kaggle-dataset", type=str, default=None, help=f"Optional Kaggle slug (e.g. {DEFAULT_KAGGLE_DATASET})")
    parser.add_argument("--vh", type=str, default=None, help="Custom VH sequence")
    parser.add_argument("--vl", type=str, default=None, help="Custom VL sequence")
    parser.add_argument("--ag", type=str, default=None, help="Custom Antigen sequence")
    parser.add_argument("--num-chains", type=int, default=8)
    parser.add_argument("--num-steps", type=int, default=8)
    parser.add_argument("--temperature", type=float, default=0.45)
    parser.add_argument("--device", type=str, default="cuda:0" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--out-dir", type=Path, default=SUBFOLDER_ROOT / "outputs" / "paratope_analysis")
    parser.add_argument("--smoke-test", action="store_true")
    args = parser.parse_args()

    vh, vl, ag = args.vh, args.vl, args.ag
    complex_label = "Trastuzumab_HER2"

    if vh is None or ag is None:
        if args.abrank_dir is not None or args.kaggle_dataset is not None:
            bdir = resolve_abrank_dir(args.abrank_dir, kaggle_dataset=args.kaggle_dataset)
            kd = pd.read_csv(
                bdir / "assays_kd.tsv",
                sep="\t",
                usecols=["AbAgID", "Ag_group", "Ab_heavy_chain_seq", "Ab_light_chain_seq", "Ag_seq", "binding_score"],
                low_memory=False,
            )
            # Pick a high-affinity compact complex (shorter antigen for fast 2D dependency map)
            cand = kd[kd["Ag_seq"].astype(str).str.len().between(40, 140)].sort_values("binding_score", ascending=False)
            if len(cand) > 0:
                row = cand.iloc[0]
                vh = str(row["Ab_heavy_chain_seq"]).strip().upper()
                vl = str(row["Ab_light_chain_seq"]).strip().upper()
                ag = str(row["Ag_seq"]).strip().upper()
                ab_short = str(row["AbAgID"]).split("__")[0]
                ag_short = str(row["Ag_group"])[:20]
                complex_label = f"{ag_short} / {ab_short}"

    vh = vh or DEFAULT_VH
    vl = vl if vl is not None else DEFAULT_VL
    ag = ag or DEFAULT_AG
    ab_paired = format_antibody_sequence(vh, vl)
    cdrh3_pos = detect_cdrh3_positions_0based(vh)

    device = torch.device(args.device)
    if args.smoke_test and args.checkpoint == "Yale-CompBio/logica-abag":
        args.checkpoint = None
    if args.checkpoint is not None and str(args.checkpoint).lower() != "none":
        model = LogiCA.from_checkpoint(args.checkpoint, scale=args.scale, device=device)
    else:
        ag_m = "facebook/esm2_t6_8M_UR50D" if args.smoke_test else args.ag_model
        model = LogiCA.from_pretrained(args.ab_model, ag_m, width=128 if args.smoke_test else 256, layers=1 if args.smoke_test else 2).to(device)
    model.eval()

    if args.smoke_test:
        args.num_chains = 3
        args.num_steps = 3
        # Trim antigen window for sub-second 2D perturbation sweep in smoke test
        ag_dep = ag[:24]
        ab_dep = vh[max(0, cdrh3_pos[0] - 8) : min(len(vh), cdrh3_pos[-1] + 8)]
        design_pos = cdrh3_pos[:4]
    else:
        ag_dep = ag[:80]
        ab_dep = ab_paired
        design_pos = cdrh3_pos[:8]

    args.out_dir.mkdir(parents=True, exist_ok=True)
    print(f"[Paratope Analysis] Complex: {complex_label} | len(VH)={len(vh)} len(VL)={len(vl)} len(Ag)={len(ag)}")
    print(f"  Detected CDR-H3 positions (0-based in VH): {cdrh3_pos} -> designing {design_pos}")

    # 1. Single-pass zero-shot KL paratope hotspot detection
    hotspots = model.detect_kl_hotspots(ab_paired, ag, top_k=8)
    print("  Top-8 antigen-sensitive antibody residues (1-based):", hotspots["top_positions_1based"])

    # 2. 2D Antigen -> Antibody Dependency Map
    dep = model.dependency_map(ab_dep, ag_dep, metric="sym_kl")
    dep_png = args.out_dir / "ab_ag_dependency_map.png"
    plot_dependency_map(
        dep,
        title=f"AbLang2 + ESM-2 Cross-Modal Dependency Map ({complex_label})",
        save_path=dep_png,
    )

    # 3. Antigen-Conditioned CDR-H3 Coordinate Gibbs Affinity Maturation
    gibbs = model.gibbs_sample(
        seed_sequence=ab_paired,
        context=ag,
        design_positions_0based=design_pos,
        num_chains=args.num_chains,
        num_steps=args.num_steps,
        temperature=args.temperature,
        guided_beta=0.6,
    )
    gibbs_png = args.out_dir / "cdrh3_gibbs_maturation.png"
    plot_gibbs_design(
        gibbs,
        title=f"Antigen-Conditioned CDR-H3 Coordinate Gibbs Design ({complex_label})",
        save_path=gibbs_png,
    )

    top_designs = [
        {"designed_vh_vl": seq, "score": round(float(sc), 4)}
        for seq, sc in gibbs.top_designs(k=min(5, args.num_chains))
    ]
    summary = {
        "complex": complex_label,
        "cdrh3_positions_0based": cdrh3_pos,
        "designed_positions_0based": design_pos,
        "top_kl_hotspots_1based": hotspots["top_positions_1based"],
        "dependency_map_shape": list(dep.matrix.shape),
        "dependency_map_png": str(dep_png),
        "gibbs_design_png": str(gibbs_png),
        "effective_sample_size": round(float(gibbs.effective_sample_size), 2),
        "top_designs": top_designs,
    }
    (args.out_dir / "paratope_summary.json").write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")
    print(f"\nSaved dependency map -> {dep_png}")
    print(f"Saved Gibbs plot     -> {gibbs_png}")
    print("Top CDR-H3 Gibbs designs:")
    for item in top_designs[:3]:
        h_part = item["designed_vh_vl"].split("|")[0]
        cdr_seq = "".join(h_part[i] for i in cdrh3_pos if i < len(h_part))
        print(f"  score={item['score']:+.4f}  CDR-H3={cdr_seq}")


if __name__ == "__main__":
    main()
