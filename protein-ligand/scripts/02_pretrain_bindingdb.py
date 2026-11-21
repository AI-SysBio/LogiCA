#!/usr/bin/env python3
"""Step 2: Stage-1 BindingDB Pretraining of ESM-2 + SELFormer with the `logica` Library.

Trains a native-head-preserving Gated Cross-Attention adapter coupling any `ESM-2` backbone
(`facebook/esm2_t12_35M_UR50D`, `150M`, or `650M`) with `HUBioDataLab/SELFormer` via
bidirectional Masked-Gain InfoNCE (`logica.InfoNCELogitLoss`, Table 1 of the paper).

Usage:
  # Full GPU pretraining on curated Stage-0 BindingDB splits:
  python protein-ligand/scripts/02_pretrain_bindingdb.py \
      --bindingdb-dir protein-ligand/data/bindingdb \
      --protein-model facebook/esm2_t30_150M_UR50D \
      --ligand-model HUBioDataLab/SELFormer \
      --device cuda:0 --epochs 10 --batch-size 32

  # Fast smoke-test verification using cohort protein-ligand pairs:
  python protein-ligand/scripts/02_pretrain_bindingdb.py --smoke-test
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
from torch.utils.data import DataLoader, Dataset

SUBFOLDER_ROOT = Path(__file__).resolve().parents[1]
REPO_ROOT = SUBFOLDER_ROOT.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from logica import InfoNCELogitLoss, LogiCA  # noqa: E402


class ProteinLigandPairDataset(Dataset):
    """Positive `(protein_sequence, ligand_selfies_or_smiles)` dataset for InfoNCE pretraining."""

    def __init__(self, proteins: list[str], ligands: list[str]) -> None:
        self.proteins = proteins
        self.ligands = ligands

    def __len__(self) -> int:
        return len(self.proteins)

    def __getitem__(self, idx: int) -> tuple[str, str]:
        return self.proteins[idx], self.ligands[idx]


def load_positive_pairs(bindingdb_dir: Path, smoke_test: bool = False) -> tuple[list[str], list[str]]:
    train_pq = bindingdb_dir / "bindingdb_train.parquet"
    if train_pq.is_file():
        df = pd.read_parquet(train_pq)
        pos = df[df["label"] == 1].reset_index(drop=True)
        if smoke_test:
            pos = pos.head(16)
        return pos["protein_sequence"].astype(str).tolist(), pos["ligand_selfies"].astype(str).tolist()

    # Fallback for smoke-testing directly from tracked benchmark manifest without BindingDB download
    cohorts_pq = SUBFOLDER_ROOT / "data" / "biolip_175" / "model_inputs.parquet"
    cdf = pd.read_parquet(cohorts_pq).dropna(subset=["protein_aaseq", "selfies"])
    if smoke_test:
        cdf = cdf.head(12)
    return cdf["protein_aaseq"].astype(str).tolist(), cdf["selfies"].astype(str).tolist()


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--bindingdb-dir", type=Path, default=SUBFOLDER_ROOT / "data" / "bindingdb")
    parser.add_argument("--protein-model", type=str, default="facebook/esm2_t30_150M_UR50D")
    parser.add_argument("--ligand-model", type=str, default="HUBioDataLab/SELFormer")
    parser.add_argument("--width", type=int, default=384)
    parser.add_argument("--layers", type=int, default=2)
    parser.add_argument("--heads", type=int, default=4)
    parser.add_argument("--epochs", type=int, default=10)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--lr", type=float, default=2e-4)
    parser.add_argument("--temperature", type=float, default=0.07)
    parser.add_argument("--device", type=str, default="cuda:0" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--out-dir", type=Path, default=SUBFOLDER_ROOT / "outputs" / "stage1_pretrain")
    parser.add_argument("--smoke-test", action="store_true")
    args = parser.parse_args()

    if args.smoke_test:
        args.protein_model = "facebook/esm2_t6_8M_UR50D"
        args.epochs = 1
        args.batch_size = 4
        args.width = 128

    args.out_dir.mkdir(parents=True, exist_ok=True)
    device = torch.device(args.device)

    prots, ligs = load_positive_pairs(args.bindingdb_dir, smoke_test=args.smoke_test)
    ds = ProteinLigandPairDataset(prots, ligs)
    loader = DataLoader(ds, batch_size=args.batch_size, shuffle=True, drop_last=len(ds) >= args.batch_size)

    print(f"[Stage-1] Initializing LogiCA ({args.protein_model} + {args.ligand_model}) on {device}...")
    model = LogiCA.from_pretrained(
        args.protein_model,
        args.ligand_model,
        width=args.width,
        heads=args.heads,
        layers=args.layers,
        arch_version="v2",
    ).to(device)

    param_groups = model.configure_finetuning_optimizer(
        base_lora_lr=args.lr * 0.2,
        adapter_lr_multiplier=5.0,
        top_k_primary_lora=0,
        top_k_partner_lora=0,
    )
    optimizer = torch.optim.AdamW(param_groups)
    criterion = InfoNCELogitLoss(temperature=args.temperature, alpha=0.5, mask_fraction=0.15)

    history: list[dict[str, float]] = []
    for epoch in range(1, args.epochs + 1):
        t0 = time.time()
        model.train()
        losses, accs = [], []
        for batch_prots, batch_ligs in loader:
            if len(batch_prots) < 2:
                continue
            p_enc = model.primary_backbone.tokenize(list(batch_prots), max_length=1024, device=device)
            d_enc = model.partner_backbone.tokenize(list(batch_ligs), max_length=256, device=device)  # type: ignore[union-attr]
            stats_out = criterion(model, p_enc, d_enc)
            loss = stats_out["loss"]

            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_([p for p in model.parameters() if p.requires_grad], 1.0)
            optimizer.step()

            losses.append(float(loss.item()))
            accs.append(float(stats_out["top1_accuracy"].item()))

        rec = {
            "epoch": epoch,
            "mean_loss": round(float(np.mean(losses)), 4) if losses else float("nan"),
            "mean_top1_acc": round(float(np.mean(accs)), 4) if accs else float("nan"),
            "elapsed_sec": round(time.time() - t0, 2),
        }
        history.append(rec)
        print(f"  Epoch {epoch}/{args.epochs}: loss={rec['mean_loss']:.4f}, top1_acc={rec['mean_top1_acc']:.3f} ({rec['elapsed_sec']}s)")

    ckpt_path = args.out_dir / "best.pt"
    model.save_checkpoint(ckpt_path, extra_metadata={"history": history})
    (args.out_dir / "pretrain_metrics.json").write_text(json.dumps(history, indent=2))
    print(f"[Stage-1] Saved pretrained LogiCA checkpoint -> {ckpt_path}")


if __name__ == "__main__":
    main()
