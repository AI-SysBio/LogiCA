#!/usr/bin/env python3
"""Step 5: Epitope-Conditioned TCR CDR3 Gibbs Design & Cross-Modal Dependency Maps (`logica`).

Demonstrates `logica.GibbsSampler` and `logica.DependencyMapper` on `TCRLang` + `ESM-2`:
  1. Computes a 2D Peptide-to-CDR3 Cross-Modal Dependency Map on the benchmark `1G4` (`NY-ESO-1`
     `SLLMWITQC`) or `A6` (`Tax` `LLFGYPVYV`) TCR–pMHC complex, localizing central CDR3$\beta$ /
     CDR3$\alpha$ contact apex residues without 3D coordinates;
  2. Runs Epitope-Conditioned Coordinate Gibbs CDR3 Design (`model.gibbs_sample`) vs. unconditional
     `TCRLang`, saving publication-ready figures (`plot_dependency_map`, `plot_gibbs_design`).

Usage:
  python tcrlang-esm/scripts/05_run_tcr_gibbs_and_dependency.py --hf-subfolder dual-anchored-ratio5
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

import matplotlib

matplotlib.use("Agg")

SUBFOLDER_ROOT = Path(__file__).resolve().parents[1]
REPO_ROOT = SUBFOLDER_ROOT.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from logica import LogiCA, plot_dependency_map, plot_gibbs_design  # noqa: E402


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", type=Path, default=None)
    parser.add_argument("--hf-subfolder", type=str, default="dual-anchored-ratio5")
    parser.add_argument("--tcr", type=str, default="CASSYVGNTGELFF|CAVRPTSGGSYIPTF", help="1G4 CDR3beta|CDR3alpha")
    parser.add_argument("--epitope", type=str, default="SLLMWITQC", help="NY-ESO-1 157-165 epitope")
    parser.add_argument("--device", type=str, default="cpu")
    parser.add_argument("--out-dir", type=Path, default=SUBFOLDER_ROOT / "outputs" / "gibbs_and_dependency")
    args = parser.parse_args()

    args.out_dir.mkdir(parents=True, exist_ok=True)
    if args.checkpoint is not None:
        model = LogiCA.from_checkpoint(args.checkpoint, device=args.device)
    else:
        model = LogiCA.from_checkpoint("ydchen/logica-tcr", subfolder=args.hf_subfolder, device=args.device)

    # 1. Cross-modal Epitope -> Paired TCR CDR3 dependency map
    dep_result = model.dependency_map(args.tcr, args.epitope, metric="sym_kl")
    dep_fig = args.out_dir / "tcr_peptide_dependency_map.png"
    plot_dependency_map(
        dep_result,
        title="1G4 Paired CDR3 ← NY-ESO-1 (SLLMWITQC) Dependency",
        save_path=dep_fig,
    )

    # 2. Design CDR3β apex positions (IMGT central loop indices 4..8) conditioned on cognate epitope
    cdr3b_len = len(args.tcr.split("|")[0])
    apex_positions = list(range(4, min(9, cdr3b_len - 2)))
    holo_gibbs = model.gibbs_sample(
        args.tcr,
        context=args.epitope,
        design_positions_0based=apex_positions,
        num_chains=16,
        num_steps=12,
        temperature=0.35,
        guided_beta=0.5,
    )
    apo_gibbs = model.gibbs_sample(
        args.tcr,
        context=None,
        design_positions_0based=apex_positions,
        num_chains=16,
        num_steps=12,
        temperature=0.35,
        guided_beta=0.0,
    )
    gibbs_fig = args.out_dir / "tcr_cdr3_gibbs_design.png"
    plot_gibbs_design(
        holo_gibbs,
        baseline_result=apo_gibbs,
        title="Epitope-Conditioned 1G4 CDR3β Apex Gibbs Design",
        save_path=gibbs_fig,
    )

    summary = {
        "tcr": args.tcr,
        "epitope": args.epitope,
        "top_5_coupled_tcr_positions": [
            {"pos_1based": p, "aa": tok, "score": round(sc, 5)}
            for p, tok, sc in dep_result.top_primary_sites(5)
        ],
        "gibbs_effective_diversity": round(holo_gibbs.effective_sample_size, 2),
        "top_3_cdr3_designs": [
            {"tcr": seq, "score": round(sc, 4)} for seq, sc in holo_gibbs.top_designs(3)
        ],
        "dependency_figure": str(dep_fig),
        "gibbs_figure": str(gibbs_fig),
    }
    out_json = args.out_dir / "tcr_gibbs_and_dependency_summary.json"
    out_json.write_text(json.dumps(summary, indent=2))
    print(f"[TCR Gibbs & Dependency] {json.dumps(summary, indent=2)}")


if __name__ == "__main__":
    main()
