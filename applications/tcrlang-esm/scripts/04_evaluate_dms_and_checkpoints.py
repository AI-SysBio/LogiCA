#!/usr/bin/env python3
"""Step 4: Evaluate Partner-Conditioned TCR & Peptide DMS Variant Ranking with `logica`.

Loads any official Hugging Face (`ydchen/logica-tcr`) or local `LogiCA` checkpoint via
`LogiCA.from_checkpoint(...)` and evaluates mutation-local variant ranking (`VariantRanker` +
reciprocal partner shift) across:
  - **71 Peptide DMS Landscapes** (`26` `ePytope-Mutation`, `35` `BATCAVE`, `10` `ATLAS-Pep`),
  - **23 Expanded TCR CDR3 DMS Landscapes** (`703` variants across `ATLAS-CDR`, `TRAIT 2025`,
    `SKEMPI v2`, and `868-Z11` saturation scans).

Usage:
  python applications/tcrlang-esm/scripts/04_evaluate_dms_and_checkpoints.py --hf-subfolder tcr-anchored-neg1
  python applications/tcrlang-esm/scripts/04_evaluate_dms_and_checkpoints.py --hf-subfolder dual-anchored-ratio5
  python applications/tcrlang-esm/scripts/04_evaluate_dms_and_checkpoints.py --checkpoint applications/tcrlang-esm/outputs/stage1_pretrain/logica_dual.pt
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

SUBFOLDER_ROOT = Path(__file__).resolve().parents[1]
REPO_ROOT = SUBFOLDER_ROOT.parents[1]
SCRIPT_DIR = Path(__file__).resolve().parent
for _p in (str(REPO_ROOT), str(SCRIPT_DIR)):
    if _p not in sys.path:
        sys.path.insert(0, _p)

from logica import LogiCA, evaluate_ranking_metrics, mean_selected_log_probability  # noqa: E402
from tcr_data_loader import (  # noqa: E402
    DEFAULT_KAGGLE_DATASET,
    resolve_tcr_data_dir,
    resolve_tcr_paths,
)


@torch.no_grad()
def score_peptide_dms_group(model: LogiCA, grp: pd.DataFrame) -> np.ndarray:
    """Scores peptide variants against a cognate TCR combining peptide LLR + partner TCR shift."""
    device = model.device
    tcr_seq = str(grp["tcr"].iloc[0]).strip().upper()
    wt_pep = str(grp["index_peptide"].iloc[0]).strip().upper()
    mut_peps = [str(p).strip().upper() for p in grp["epitope"]]

    # 1. Self-chain mutation-local LLR via VariantRanker
    same_len = [p for p in mut_peps if len(p) == len(wt_pep)]
    if len(same_len) == len(mut_peps):
        rk = model.rank_variants(wt_pep, mut_peps, context=tcr_seq, mode="exact_loo", reduction="mean")
        pep_llr = rk.holo_scores
    else:
        pep_llr = np.zeros(len(mut_peps), dtype=np.float64)

    # 2. Partner TCR cross-attention log-likelihood shift induced by mutant vs WT peptide
    all_peps = [wt_pep] + mut_peps
    p_enc = model.primary_backbone.tokenize([tcr_seq], max_length=48, device=device)
    d_enc = model.partner_backbone.tokenize(all_peps, max_length=20, device=device)  # type: ignore[union-attr]
    p_last_1, p_qkv_1 = model.encode_primary(p_enc["input_ids"], p_enc["attention_mask"])
    d_last = model.encode_partner(d_enc["input_ids"], d_enc["attention_mask"])

    b = len(all_peps)
    out = model.forward_from_hidden(
        p_last_1.expand(b, -1, -1),
        p_enc["attention_mask"].expand(b, -1),
        d_last,
        d_enc["attention_mask"],
        primary_adapter_input=p_qkv_1.expand(b, -1, -1),
    )
    p_valid = (p_enc["attention_mask"] == 1) & (~p_enc["special_tokens_mask"].bool())
    tcr_ll = mean_selected_log_probability(
        out.primary_holo_logits, p_enc["input_ids"].expand(b, -1), p_valid.expand(b, -1)
    ).cpu().numpy()
    partner_shift = tcr_ll[1:] - tcr_ll[0]

    s1 = (pep_llr - np.mean(pep_llr)) / max(float(np.std(pep_llr)), 1e-8)
    s2 = (partner_shift - np.mean(partner_shift)) / max(float(np.std(partner_shift)), 1e-8)
    return 0.5 * (s1 + s2)


@torch.no_grad()
def score_tcr_dms_group(model: LogiCA, grp: pd.DataFrame) -> np.ndarray:
    """Scores TCR CDR3 variants against a fixed cognate peptide epitope."""
    epi_seq = str(grp["epitope"].iloc[0]).strip().upper()
    wt_tcr = str(grp["index_tcr"].iloc[0]).strip().upper()
    mut_tcrs = [str(t).strip().upper() for t in grp["tcr"]]

    scores = np.zeros(len(mut_tcrs), dtype=np.float64)
    same_idx = [i for i, t in enumerate(mut_tcrs) if len(t) == len(wt_tcr)]
    if same_idx:
        rk = model.rank_variants(
            wt_tcr,
            [mut_tcrs[i] for i in same_idx],
            context=epi_seq,
            mode="masked_sites",
            reduction="sum",
            component="guided",
            guided_beta=0.8,
            max_primary_len=64,
            max_partner_len=24,
        )
        for local_i, row_i in enumerate(same_idx):
            scores[row_i] = float(rk.scores[local_i])

    diff_idx = [i for i in range(len(mut_tcrs)) if i not in set(same_idx)]
    if diff_idx:
        device = model.device
        seqs = [wt_tcr] + [mut_tcrs[i] for i in diff_idx]
        p_enc = model.primary_backbone.tokenize(seqs, max_length=48, device=device)
        d_enc = model.partner_backbone.tokenize([epi_seq], max_length=20, device=device)  # type: ignore[union-attr]
        p_last, p_qkv = model.encode_primary(p_enc["input_ids"], p_enc["attention_mask"])
        d_last_1 = model.encode_partner(d_enc["input_ids"], d_enc["attention_mask"])
        out = model.forward_from_hidden(
            p_last,
            p_enc["attention_mask"],
            d_last_1.expand(len(seqs), -1, -1),
            d_enc["attention_mask"].expand(len(seqs), -1),
            primary_adapter_input=p_qkv,
        )
        valid = (p_enc["attention_mask"] == 1) & (~p_enc["special_tokens_mask"].bool())
        holo_ll = mean_selected_log_probability(out.primary_holo_logits, p_enc["input_ids"], valid).cpu().numpy()
        apo_ll = mean_selected_log_probability(out.primary_apo_logits, p_enc["input_ids"], valid).cpu().numpy()
        comb = 0.2 * (holo_ll[1:] - holo_ll[0]) + 0.8 * ((holo_ll[1:] - apo_ll[1:]) - (holo_ll[0] - apo_ll[0]))
        for local_i, row_i in enumerate(diff_idx):
            scores[row_i] = float(comb[local_i])
    return scores


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-dir", type=Path, default=None, help="Local dataset dir (auto-fetches Kaggle if omitted)")
    parser.add_argument("--kaggle-dataset", type=str, default=DEFAULT_KAGGLE_DATASET)
    parser.add_argument("--checkpoint", type=Path, default=None)
    parser.add_argument("--hf-subfolder", type=str, default="tcr-anchored-neg1")
    parser.add_argument("--device", type=str, default="cuda:0" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--max-groups", type=int, default=None)
    parser.add_argument("--out-dir", type=Path, default=SUBFOLDER_ROOT / "outputs" / "dms_eval")
    args = parser.parse_args()

    args.out_dir.mkdir(parents=True, exist_ok=True)
    t0 = time.time()
    data_dir = resolve_tcr_data_dir(args.data_dir, kaggle_dataset=args.kaggle_dataset)
    paths = resolve_tcr_paths(data_dir)

    if args.checkpoint is not None:
        model = LogiCA.from_checkpoint(args.checkpoint, device=args.device)
        model_tag = args.checkpoint.stem
    else:
        model = LogiCA.from_checkpoint("ydchen/logica-tcr", subfolder=args.hf_subfolder, device=args.device)
        model_tag = args.hf_subfolder

    # Swap primary=ESM-2, partner=TCRLang view for peptide-side VariantRanker convenience
    pep_view_model = LogiCA(
        primary_backbone=model.partner_backbone,  # type: ignore[arg-type]
        partner_backbone=model.primary_backbone,
        layers=model.num_adapter_layers,
        adapter_style=model.adapter_style,
        gate_mode=model.gate_mode,
    ).to(args.device).eval()
    if model.adapter_style == "shared_bottleneck" and model.cross_attn is not None:
        pep_view_model.prot_proj_in = model.drug_proj_in
        pep_view_model.prot_proj_out = model.drug_proj_out
        pep_view_model.drug_proj_in = model.prot_proj_in
        pep_view_model.drug_proj_out = model.prot_proj_out
        pep_view_model.prot_gate = model.drug_gate
        pep_view_model.drug_gate = model.prot_gate
        pep_view_model.cross_attn.p2d = model.cross_attn.d2p
        pep_view_model.cross_attn.d2p = model.cross_attn.p2d

    pep_sources = [
        ("ePytope-Mut", paths["epytope_mutation"], "peptide_activity", False),
        ("BATCAVE", paths["batcave"], "peptide_activity", False),
        ("ATLAS-Pep", paths["atlas_pep"], "KD_clean", True),
    ]

    pep_rhos, pep_p10s = [], []
    for ds_name, csv_path, target_col, negate_kd in pep_sources:
        df = pd.read_csv(csv_path)
        groups = list(df.groupby(["tcr", "index_peptide"], sort=False))
        if args.max_groups is not None:
            groups = groups[: args.max_groups]
        for _, grp in groups:
            if len(grp) < 4:
                continue
            y = pd.to_numeric(grp[target_col], errors="coerce").to_numpy(dtype=np.float64)
            if negate_kd:
                y = -np.log10(np.clip(y, 1e-6, None))
            if model.adapter_style == "shared_bottleneck":
                sc = score_peptide_dms_group(pep_view_model, grp.rename(columns={"tcr": "tmp_t", "epitope": "tmp_e"}).assign(tcr=grp["epitope"], epitope=grp["tcr"])) if False else None
            # Direct unified scorer on `model` (primary=TCR, partner=Peptide):
            sc = _score_peptide_with_tcr_primary(model, grp)
            m = evaluate_ranking_metrics(y, sc)
            pep_rhos.append(m["spearman_rho"])
            pep_p10s.append(m["prec_at_10pct"])
        print(f"  [{ds_name}] cumulative mean Spearman = {float(np.mean(pep_rhos)):+.4f}")

    tcr_df = pd.read_csv(paths["tcr_mutations_combined"])
    tcr_groups = list(tcr_df.groupby(["tcr_name", "epitope"], sort=False))
    if args.max_groups is not None:
        tcr_groups = tcr_groups[: args.max_groups]

    tcr_rhos, tcr_p10s = [], []
    for _, grp in tcr_groups:
        if len(grp) < 3:
            continue
        sc = score_tcr_dms_group(model, grp)
        y = grp["tcr_activity"].to_numpy(dtype=np.float64)
        m = evaluate_ranking_metrics(y, sc)
        tcr_rhos.append(m["spearman_rho"])
        tcr_p10s.append(m["prec_at_10pct"])

    summary = {
        "model": model_tag,
        "n_peptide_dms_groups": len(pep_rhos),
        "peptide_dms_spearman_mean": round(float(np.mean(pep_rhos)), 4) if pep_rhos else float("nan"),
        "peptide_dms_prec10_mean": round(float(np.mean(pep_p10s)), 3) if pep_p10s else float("nan"),
        "n_tcr_dms_groups": len(tcr_rhos),
        "tcr_dms_spearman_mean": round(float(np.mean(tcr_rhos)), 4) if tcr_rhos else float("nan"),
        "tcr_dms_prec10_mean": round(float(np.mean(tcr_p10s)), 3) if tcr_p10s else float("nan"),
        "elapsed_sec": round(time.time() - t0, 2),
    }
    out_json = args.out_dir / f"dms_eval_{model_tag}.json"
    out_json.write_text(json.dumps(summary, indent=2))
    print(f"[TCR DMS Eval] {summary}")


@torch.no_grad()
def _score_peptide_with_tcr_primary(model: LogiCA, grp: pd.DataFrame) -> np.ndarray:
    """Evaluates peptide substitutions when `primary_backbone=TCRLang` and `partner_backbone=ESM-2`."""
    device = model.device
    tcr_seq = str(grp["tcr"].iloc[0]).strip().upper()
    wt_pep = str(grp["index_peptide"].iloc[0]).strip().upper()
    mut_peps = [str(p).strip().upper() for p in grp["epitope"]]

    p_enc = model.primary_backbone.tokenize([tcr_seq], max_length=48, device=device)
    p_last_1, p_qkv_1 = model.encode_primary(p_enc["input_ids"], p_enc["attention_mask"])

    all_peps = [wt_pep] + mut_peps
    d_enc = model.partner_backbone.tokenize(all_peps, max_length=20, device=device)  # type: ignore[union-attr]
    d_last = model.encode_partner(d_enc["input_ids"], d_enc["attention_mask"])

    b = len(all_peps)
    out = model.forward_from_hidden(
        p_last_1.expand(b, -1, -1),
        p_enc["attention_mask"].expand(b, -1),
        d_last,
        d_enc["attention_mask"],
        primary_adapter_input=p_qkv_1.expand(b, -1, -1),
    )
    assert out.partner_holo_logits is not None
    p_valid = (p_enc["attention_mask"] == 1) & (~p_enc["special_tokens_mask"].bool())
    tcr_ll = mean_selected_log_probability(
        out.primary_holo_logits, p_enc["input_ids"].expand(b, -1), p_valid.expand(b, -1)
    ).cpu().numpy()
    partner_shift = tcr_ll[1:] - tcr_ll[0]

    # Masked peptide LLR at differing positions
    wt_d_ids = d_enc["input_ids"][0]
    wt_d_attn = d_enc["attention_mask"][0]
    d_valid_pos = torch.nonzero(
        (wt_d_attn == 1) & (~d_enc["special_tokens_mask"][0].bool()), as_tuple=False
    ).flatten().tolist()
    mask_d = model.partner_backbone.mask_token_id  # type: ignore[union-attr]

    pep_llr = np.zeros(len(mut_peps), dtype=np.float64)
    masked_rows = []
    pos_list = []
    for pos_i in range(min(len(wt_pep), len(d_valid_pos))):
        r = wt_d_ids.clone()
        r[d_valid_pos[pos_i]] = mask_d
        masked_rows.append(r)
        pos_list.append(d_valid_pos[pos_i])

    if masked_rows:
        mb = torch.stack(masked_rows, dim=0)
        ma = wt_d_attn.unsqueeze(0).expand(mb.size(0), -1)
        md_last = model.encode_partner(mb, ma)
        mout = model.forward_from_hidden(
            p_last_1.expand(mb.size(0), -1, -1),
            p_enc["attention_mask"].expand(mb.size(0), -1),
            md_last,
            ma,
            primary_adapter_input=p_qkv_1.expand(mb.size(0), -1, -1),
        )
        assert mout.partner_holo_logits is not None
        mlp = torch.log_softmax(mout.partner_holo_logits.float(), dim=-1)
        for idx, m_pep in enumerate(mut_peps):
            if len(m_pep) != len(wt_pep):
                continue
            diffs = [i for i in range(min(len(wt_pep), len(pos_list))) if m_pep[i] != wt_pep[i]]
            if diffs:
                vals = []
                for i in diffs:
                    tm = model.partner_backbone.token_to_id(m_pep[i])  # type: ignore[union-attr]
                    tw = model.partner_backbone.token_to_id(wt_pep[i])  # type: ignore[union-attr]
                    vals.append(float((mlp[i, pos_list[i], tm] - mlp[i, pos_list[i], tw]).item()))
                pep_llr[idx] = float(np.mean(vals))

    s1 = (pep_llr - np.mean(pep_llr)) / max(float(np.std(pep_llr)), 1e-8)
    s2 = (partner_shift - np.mean(partner_shift)) / max(float(np.std(partner_shift)), 1e-8)
    return 0.5 * (s1 + s2)


if __name__ == "__main__":
    main()
