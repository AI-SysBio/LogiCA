#!/usr/bin/env python3
"""Step 2: Stage-1 Pretraining of `TCRLang` + `ESM-2` Powered by the `logica` Library.

Couples paired `tcrlang-paired` (`CDR3beta|CDR3alpha`) with `ESM-2` (`facebook/esm2_t12_35M_UR50D`,
`150M`, or `650M`) via `LogiCA.from_pretrained` and trains native-logit contrastive alignment
(`InfoNCELogitLoss`) on the non-overlapping `EPACT + STAPLER + IEDB` repertoire corpus.

Supports all Table 3 & Section 3.2 training configurations:
  - `--mode logica_dual`: Bidirectional TCR + Peptide anchored native-logit InfoNCE
  - `--mode logica_tcr`:  TCR-anchored native-logit InfoNCE (`anchor_mode="primary"`)
  - `--mode logica_pep`:  Peptide-anchored native-logit InfoNCE (`anchor_mode="partner"`)
  - `--mode logimlm`:     Positive-only conditional MLM control
and all negative-construction strategies (`central_contact`, `non_conservative`, `uniform`, `inbatch`).

Usage:
  python tcrlang-esm/scripts/02_pretrain_tcrlang_esm.py --mode logica_dual --device cuda:0 --epochs 10
  python tcrlang-esm/scripts/02_pretrain_tcrlang_esm.py --smoke-test
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import random
import sys
import time

import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader, Dataset

SUBFOLDER_ROOT = Path(__file__).resolve().parents[1]
REPO_ROOT = SUBFOLDER_ROOT.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from logica import (  # noqa: E402
    InfoNCELogitLoss,
    LogiCA,
    create_anchored_mutation_negatives,
    make_masked_inputs,
    mean_selected_log_probability,
)


class TCRPeptideRepertoireDataset(Dataset):
    def __init__(self, tcrs: list[str], epitopes: list[str]) -> None:
        self.tcrs = tcrs
        self.epitopes = epitopes

    def __len__(self) -> int:
        return len(self.tcrs)

    def __getitem__(self, idx: int) -> tuple[str, str]:
        return self.tcrs[idx], self.epitopes[idx]


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--train-csv",
        type=Path,
        default=SUBFOLDER_ROOT / "data" / "pretrain" / "epact_stapler_iedb_train_non_overlap.csv",
    )
    parser.add_argument("--tcr-model", type=str, default="tcrlang-paired")
    parser.add_argument("--esm-model", type=str, default="facebook/esm2_t12_35M_UR50D")
    parser.add_argument(
        "--mode",
        type=str,
        choices=["logica_dual", "logica_tcr", "logica_pep", "logimlm"],
        default="logica_dual",
    )
    parser.add_argument(
        "--neg-strategy",
        type=str,
        choices=["central_contact", "non_conservative", "uniform", "inbatch"],
        default="central_contact",
    )
    parser.add_argument("--adapter-style", type=str, choices=["shared_bottleneck", "native_additional_layers"], default="shared_bottleneck")
    parser.add_argument("--width", type=int, default=384)
    parser.add_argument("--layers", type=int, default=2)
    parser.add_argument("--epochs", type=int, default=10)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--temperature", type=float, default=0.10)
    parser.add_argument("--device", type=str, default="cuda:0" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--out-dir", type=Path, default=SUBFOLDER_ROOT / "outputs" / "stage1_pretrain")
    parser.add_argument("--smoke-test", action="store_true")
    args = parser.parse_args()

    df = pd.read_csv(args.train_csv).dropna(subset=["tcr", "epitope"]).drop_duplicates(subset=["tcr", "epitope"])
    if args.smoke_test:
        df = df.head(24).reset_index(drop=True)
        args.esm_model = "facebook/esm2_t6_8M_UR50D"
        args.epochs = 1
        args.batch_size = 8
        args.width = 128

    args.out_dir.mkdir(parents=True, exist_ok=True)
    device = torch.device(args.device)
    rng = random.Random(42)

    ds = TCRPeptideRepertoireDataset(
        df["tcr"].astype(str).str.strip().str.upper().tolist(),
        df["epitope"].astype(str).str.strip().str.upper().tolist(),
    )
    loader = DataLoader(ds, batch_size=args.batch_size, shuffle=True, drop_last=len(ds) >= args.batch_size)

    print(f"[TCR Stage-1] Building LogiCA ({args.tcr_model} + {args.esm_model}, mode={args.mode}) on {device}...")
    model = LogiCA.from_pretrained(
        args.tcr_model,
        args.esm_model,
        width=args.width,
        layers=args.layers,
        adapter_style=args.adapter_style,  # type: ignore[arg-type]
    ).to(device)

    param_groups = model.configure_finetuning_optimizer(
        base_lora_lr=args.lr * 0.5,
        adapter_lr_multiplier=2.0,
        top_k_primary_lora=2,
        top_k_partner_lora=2,
        lora_rank=8,
    )
    optimizer = torch.optim.AdamW(param_groups)

    anchor_map = {
        "logica_dual": "bidirectional",
        "logica_tcr": "primary",
        "logica_pep": "partner",
        "logimlm": "bidirectional",
    }
    criterion = InfoNCELogitLoss(
        temperature=args.temperature,
        alpha=0.5,
        mask_fraction=0.15,
        inbatch_negatives_per_anchor=5,
        anchor_mode=anchor_map[args.mode],  # type: ignore[arg-type]
        subtract_apo_baseline=True,
        mlm_aux_weight=1.0,
    )

    history: list[dict[str, float]] = []
    for epoch in range(1, args.epochs + 1):
        t0 = time.time()
        model.train()
        losses, accs = [], []

        for tcrs_b, peps_b in loader:
            tcr_list = list(tcrs_b)
            pep_list = list(peps_b)
            if len(tcr_list) < 2:
                continue

            p_enc = model.primary_backbone.tokenize(tcr_list, max_length=48, device=device)
            d_enc = model.partner_backbone.tokenize(pep_list, max_length=20, device=device)  # type: ignore[union-attr]

            if args.mode == "logimlm":
                p_m, p_sel = make_masked_inputs(
                    p_enc["input_ids"], p_enc["attention_mask"], p_enc["special_tokens_mask"],
                    model.primary_backbone.mask_token_id, 0.15,
                )
                d_m, d_sel = make_masked_inputs(
                    d_enc["input_ids"], d_enc["attention_mask"], d_enc["special_tokens_mask"],
                    model.partner_backbone.mask_token_id, 0.15,  # type: ignore[union-attr]
                )
                out = model(p_m, p_enc["attention_mask"], d_m, d_enc["attention_mask"])
                assert out.partner_holo_logits is not None
                loss_a = -mean_selected_log_probability(out.primary_holo_logits, p_enc["input_ids"], p_sel).mean()
                loss_b = -mean_selected_log_probability(out.partner_holo_logits, d_enc["input_ids"], d_sel).mean()
                loss = 0.5 * (loss_a + loss_b)
                acc_val = 0.0
            else:
                hard_d_enc = None
                hard_p_enc = None
                if args.neg_strategy != "inbatch":
                    neg_peps = [
                        create_anchored_mutation_negatives(p, num_negatives=1, strategy=args.neg_strategy, rng=rng)[0]  # type: ignore[arg-type]
                        for p in pep_list
                    ]
                    neg_tcrs = [
                        create_anchored_mutation_negatives(t, num_negatives=1, strategy=args.neg_strategy, rng=rng)[0]  # type: ignore[arg-type]
                        for t in tcr_list
                    ]
                    hard_d_enc = model.partner_backbone.tokenize(neg_peps, max_length=20, device=device)  # type: ignore[union-attr]
                    hard_p_enc = model.primary_backbone.tokenize(neg_tcrs, max_length=48, device=device)

                out_stats = criterion(
                    model,
                    p_enc,
                    d_enc,
                    hard_neg_partner_batch=hard_d_enc,
                    hard_neg_primary_batch=hard_p_enc,
                )
                loss = out_stats["loss"]
                acc_val = float(out_stats["top1_accuracy"].item())

            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_([p for p in model.parameters() if p.requires_grad], 1.0)
            optimizer.step()

            losses.append(float(loss.item()))
            accs.append(acc_val)

        rec = {
            "epoch": epoch,
            "mean_loss": round(float(np.mean(losses)), 4),
            "mean_top1_acc": round(float(np.mean(accs)), 4),
            "elapsed_sec": round(time.time() - t0, 2),
        }
        history.append(rec)
        print(f"  Epoch {epoch}/{args.epochs} [{args.mode}]: loss={rec['mean_loss']:.4f}, acc={rec['mean_top1_acc']:.3f} ({rec['elapsed_sec']}s)")

    ckpt_path = args.out_dir / f"{args.mode}.pt"
    model.save_checkpoint(ckpt_path, extra_metadata={"mode": args.mode, "history": history})
    (args.out_dir / f"{args.mode}_history.json").write_text(json.dumps(history, indent=2))
    print(f"[TCR Stage-1] Saved checkpoint -> {ckpt_path}")


if __name__ == "__main__":
    main()
