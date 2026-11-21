#!/usr/bin/env python3
"""Step 3: Stage-2 Supervised TCR–Epitope Fine-Tuning in Native Logit Space (`logica`).

Fine-tunes a Stage-1 `LogiCA` (`tcrlang-paired` + `ESM-2`) checkpoint — or an official
`ydchen/logica-tcr` Hugging Face checkpoint — on labeled `(tcr, epitope, label)` pairs from
`EPACT` (`epact_train_epytope_binary_test`), optimizing bidirectional contextual log-likelihood
gain (`s_gain(x, y)`) via binary margin / logistic supervision while strictly preserving both
native token heads! Evaluates held-out `ePytope-Binary` AUROC and AUPR.

Usage:
  python tcrlang-esm/scripts/03_finetune_tcrlang_esm.py \
      --hf-subfolder dual-anchored-ratio5 --device cuda:0 --epochs 6

  python tcrlang-esm/scripts/03_finetune_tcrlang_esm.py --smoke-test
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys
import time

import numpy as np
import pandas as pd
from sklearn.metrics import average_precision_score, roc_auc_score
import torch
from torch import nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset

SUBFOLDER_ROOT = Path(__file__).resolve().parents[1]
REPO_ROOT = SUBFOLDER_ROOT.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from logica import LogiCA, mean_selected_log_probability  # noqa: E402


class BinaryTCRDataset(Dataset):
    def __init__(self, df: pd.DataFrame) -> None:
        self.tcrs = df["tcr"].astype(str).str.strip().str.upper().tolist()
        self.epis = df["epitope"].astype(str).str.strip().str.upper().tolist()
        self.labels = df["label"].astype(float).tolist()

    def __len__(self) -> int:
        return len(self.tcrs)

    def __getitem__(self, idx: int) -> tuple[str, str, float]:
        return self.tcrs[idx], self.epis[idx], self.labels[idx]


def compute_pair_contextual_gain(
    model: LogiCA,
    tcrs: list[str],
    epis: list[str],
    device: torch.device,
) -> torch.Tensor:
    """Computes bidirectional unmasked contextual log-likelihood gain `0.5 * (gain_tcr + gain_pep)`."""
    p_enc = model.primary_backbone.tokenize(tcrs, max_length=48, device=device)
    d_enc = model.partner_backbone.tokenize(epis, max_length=20, device=device)  # type: ignore[union-attr]
    out = model(p_enc["input_ids"], p_enc["attention_mask"], d_enc["input_ids"], d_enc["attention_mask"])
    assert out.partner_holo_logits is not None and out.partner_apo_logits is not None

    p_valid = (p_enc["attention_mask"] == 1) & (~p_enc["special_tokens_mask"].bool())
    d_valid = (d_enc["attention_mask"] == 1) & (~d_enc["special_tokens_mask"].bool())

    p_holo_ll = mean_selected_log_probability(out.primary_holo_logits, p_enc["input_ids"], p_valid)
    p_apo_ll = mean_selected_log_probability(out.primary_apo_logits, p_enc["input_ids"], p_valid)
    d_holo_ll = mean_selected_log_probability(out.partner_holo_logits, d_enc["input_ids"], d_valid)
    d_apo_ll = mean_selected_log_probability(out.partner_apo_logits, d_enc["input_ids"], d_valid)
    return 0.5 * ((p_holo_ll - p_apo_ll) + (d_holo_ll - d_apo_ll))


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--split-dir",
        type=Path,
        default=SUBFOLDER_ROOT / "data" / "finetune" / "epact_train_all_downstream_non_overlap" / "epact_train_epytope_binary_test",
    )
    parser.add_argument("--checkpoint", type=Path, default=None, help="Local Stage-1 `.pt` checkpoint")
    parser.add_argument("--hf-subfolder", type=str, default="dual-anchored-ratio5")
    parser.add_argument("--epochs", type=int, default=6)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--lr", type=float, default=5e-5)
    parser.add_argument("--device", type=str, default="cuda:0" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--out-dir", type=Path, default=SUBFOLDER_ROOT / "outputs" / "stage2_finetune")
    parser.add_argument("--smoke-test", action="store_true")
    args = parser.parse_args()

    args.out_dir.mkdir(parents=True, exist_ok=True)
    device = torch.device(args.device)

    df_train = pd.read_csv(args.split_dir / "train.csv").dropna(subset=["tcr", "epitope", "label"])
    df_test = pd.read_csv(args.split_dir / "test.csv").dropna(subset=["tcr", "epitope", "label"])

    # Exclude any pair overlapping expanded TCR DMS test set
    exp_tcr_csv = SUBFOLDER_ROOT / "data" / "expanded_tcr_mut" / "tcr_mutations_combined_test.csv"
    if exp_tcr_csv.is_file():
        tdf = pd.read_csv(exp_tcr_csv)
        held_out_pairs = set(zip(tdf["tcr"].str.strip().str.upper(), tdf["epitope"].str.strip().str.upper()))
        df_train = df_train[
            [(str(t).strip().upper(), str(e).strip().upper()) not in held_out_pairs for t, e in zip(df_train["tcr"], df_train["epitope"])]
        ].reset_index(drop=True)

    if args.smoke_test:
        df_train = df_train.head(32).reset_index(drop=True)
        df_test = df_test.head(32).reset_index(drop=True)
        args.epochs = 1
        args.batch_size = 16
        model = LogiCA.from_pretrained("tcrlang-paired", "facebook/esm2_t6_8M_UR50D", width=128).to(device)
    elif args.checkpoint is not None:
        model = LogiCA.from_checkpoint(args.checkpoint, device=device)
    else:
        model = LogiCA.from_checkpoint("ydchen/logica-tcr", subfolder=args.hf_subfolder, device=device)

    logit_scale = nn.Parameter(torch.tensor([4.0], device=device))
    logit_bias = nn.Parameter(torch.tensor([0.0], device=device))
    calib_head = nn.ParameterList([logit_scale, logit_bias])

    opt_groups = model.configure_finetuning_optimizer(
        base_lora_lr=args.lr,
        adapter_lr_multiplier=3.0,
        top_k_primary_lora=2,
        top_k_partner_lora=2,
        extra_modules=[calib_head],
    )
    optimizer = torch.optim.AdamW(opt_groups)

    train_loader = DataLoader(BinaryTCRDataset(df_train), batch_size=args.batch_size, shuffle=True)
    test_loader = DataLoader(BinaryTCRDataset(df_test), batch_size=args.batch_size, shuffle=False)

    t0 = time.time()
    for ep in range(1, args.epochs + 1):
        model.train()
        ep_losses = []
        for tcrs_b, epis_b, y_b in train_loader:
            y_t = y_b.to(device=device, dtype=torch.float32)
            gains = compute_pair_contextual_gain(model, list(tcrs_b), list(epis_b), device)
            logits = logit_scale * gains + logit_bias
            loss = F.binary_cross_entropy_with_logits(logits, y_t)
            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_([p for p in model.parameters() if p.requires_grad], 1.0)
            optimizer.step()
            ep_losses.append(float(loss.item()))
        print(f"  Epoch {ep}/{args.epochs}: bce_loss={float(np.mean(ep_losses)):.4f}")

    model.eval()
    all_y, all_s = [], []
    with torch.no_grad():
        for tcrs_b, epis_b, y_b in test_loader:
            gains = compute_pair_contextual_gain(model, list(tcrs_b), list(epis_b), device)
            probs = torch.sigmoid(logit_scale * gains + logit_bias).cpu().numpy().tolist()
            all_y.extend(y_b.numpy().tolist())
            all_s.extend(probs if isinstance(probs, list) else [probs])

    auroc = float(roc_auc_score(all_y, all_s)) if len(set(all_y)) == 2 else float("nan")
    aupr = float(average_precision_score(all_y, all_s)) if len(set(all_y)) == 2 else float("nan")

    ckpt_out = args.out_dir / "stage2_finetuned.pt"
    model.save_checkpoint(ckpt_out, extra_metadata={"epytope_binary_auroc": auroc, "epytope_binary_aupr": aupr})
    summary = {
        "epytope_binary_auroc": round(auroc, 4),
        "epytope_binary_aupr": round(aupr, 4),
        "elapsed_sec": round(time.time() - t0, 2),
        "checkpoint": str(ckpt_out),
    }
    (args.out_dir / "stage2_summary.json").write_text(json.dumps(summary, indent=2))
    print(f"[TCR Stage-2] {summary}")


if __name__ == "__main__":
    main()
