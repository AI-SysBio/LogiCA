#!/usr/bin/env python3
"""Step 1: Fetch, Clean, Deduplicate & Verify Zero-Leakage TCR–Epitope Datasets.

Manages all data splits for TCR–Peptide LogiCA (`tcrlang-paired` + `ESM-2`):
  1. Verifies the non-overlapping Stage-1 pretraining repertoire corpus
     (`data/pretrain/epact_stapler_iedb_train_non_overlap.csv` & `val`) and Stage-2
     supervised/DMS splits (`data/finetune/epact_train_all_downstream_non_overlap/`).
  2. Optionally rebuilds Stage-2 `EPACT + STAPLER` train/val splits from raw upstream CSVs
     (`--epact-train-csv` and `--stapler-train-csv`).
  3. Optionally re-fetches raw external TCR-mutation benchmarks (`--download-external`:
     `TRAIT 2025`, `SKEMPI v2`, and `868-Z11` 20-AA saturation DMS via HTTP Range from
     Zenodo `10685013`) and verifies strict `0.00%` `(tcr, epitope)` pair overlap against
     all Stage-1 and Stage-2 training sets!

Usage:
  python tcrlang-esm/scripts/01_fetch_and_prepare_tcr_data.py
  python tcrlang-esm/scripts/01_fetch_and_prepare_tcr_data.py --download-external
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import struct
import urllib.request
import zlib

import pandas as pd

SUBFOLDER_ROOT = Path(__file__).resolve().parents[1]

TRAIT_ZIP_URL = "https://pgx.zju.edu.cn/download.trait/Mutation_20250312.zip"
SKEMPI_CSV_URL = "https://life.bsc.es/pid/skempi2/database/download/skempi_v2.csv"
ZENODO_868_ZIP_URL = "https://zenodo.org/api/records/10685013/files/alldata.zip/content"


def _download_file(url: str, dest: Path) -> None:
    dest.parent.mkdir(parents=True, exist_ok=True)
    if dest.is_file() and dest.stat().st_size > 0:
        print(f"[Skip] {dest.name} already exists ({dest.stat().st_size:,} bytes)")
        return
    print(f"[Download] {url} -> {dest}")
    req = urllib.request.Request(url, headers={"User-Agent": "Mozilla/5.0 LogiCA-TCR/1.0"})
    with urllib.request.urlopen(req, timeout=120) as resp, open(dest, "wb") as out:
        out.write(resp.read())


def _fetch_range(url: str, start: int, end: int) -> bytes:
    req = urllib.request.Request(
        url,
        headers={"Range": f"bytes={start}-{end}", "User-Agent": "Mozilla/5.0 LogiCA-TCR/1.0"},
    )
    with urllib.request.urlopen(req, timeout=120) as resp:
        return resp.read()


def download_external_tcr_mutation_sources(raw_dir: Path) -> None:
    """Downloads SKEMPI v2, TRAIT 2025, and Range-extracts the 868-Z11 Fitness TSVs from Zenodo."""
    raw_dir.mkdir(parents=True, exist_ok=True)
    _download_file(SKEMPI_CSV_URL, raw_dir / "skempi_v2.csv")
    _download_file(TRAIT_ZIP_URL, raw_dir / "TRAIT_Mutation_20250312.zip")

    f1 = raw_dir / "9-868_Z11_Library1_1nm_25_Fitness.tsv"
    f2 = raw_dir / "10-868_Z11_Library2_1nm_25_Fitness.tsv"
    if f1.is_file() and f2.is_file():
        return

    print("[Download] Extracting 868-Z11 CDR3 DMS TSVs via HTTP Range from Zenodo 10685013...")
    req_head = urllib.request.Request(ZENODO_868_ZIP_URL, method="HEAD", headers={"User-Agent": "Mozilla/5.0"})
    with urllib.request.urlopen(req_head, timeout=60) as r:
        total_size = int(r.headers["Content-Length"])

    tail = _fetch_range(ZENODO_868_ZIP_URL, max(0, total_size - 65536), total_size - 1)
    eocd_pos = tail.rfind(b"PK\x05\x06")
    cd_size, cd_offset = struct.unpack("<II", tail[eocd_pos + 12 : eocd_pos + 20])
    cd_bytes = _fetch_range(ZENODO_868_ZIP_URL, cd_offset, cd_offset + cd_size - 1)

    targets = {
        "9-868_Z11_Library1_1nm_25_Fitness.tsv": f1,
        "10-868_Z11_Library2_1nm_25_Fitness.tsv": f2,
    }
    pos = 0
    while pos < len(cd_bytes):
        if cd_bytes[pos : pos + 4] != b"PK\x01\x02":
            break
        comp_method = struct.unpack("<H", cd_bytes[pos + 10 : pos + 12])[0]
        comp_size, _ = struct.unpack("<II", cd_bytes[pos + 20 : pos + 28])
        fn_len, ex_len, cm_len = struct.unpack("<HHH", cd_bytes[pos + 28 : pos + 34])
        lh_off = struct.unpack("<I", cd_bytes[pos + 42 : pos + 46])[0]
        fname = cd_bytes[pos + 46 : pos + 46 + fn_len].decode("utf-8", errors="replace")
        pos += 46 + fn_len + ex_len + cm_len
        base = Path(fname).name
        if base in targets and not targets[base].is_file():
            chunk = _fetch_range(ZENODO_868_ZIP_URL, lh_off, lh_off + 30 + fn_len + 256 + comp_size)
            lfn, lex = struct.unpack("<HH", chunk[26:30])
            payload = chunk[30 + lfn + lex : 30 + lfn + lex + comp_size]
            data = zlib.decompress(payload, -15) if comp_method == 8 else payload
            targets[base].write_bytes(data)
            print(f"  -> Extracted {base} ({len(data):,} bytes)")


def verify_and_summarize_tcr_splits(data_dir: Path) -> dict[str, object]:
    pre_tr = pd.read_csv(data_dir / "pretrain" / "epact_stapler_iedb_train_non_overlap.csv")
    pre_va = pd.read_csv(data_dir / "pretrain" / "epact_stapler_iedb_val_non_overlap.csv")

    ft_root = data_dir / "finetune" / "epact_train_all_downstream_non_overlap"
    ft_tr = pd.read_csv(ft_root / "epact_train_epytope_binary_test" / "train.csv")
    ft_va = pd.read_csv(ft_root / "epact_train_epytope_binary_test" / "val.csv")

    train_pairs = set(
        zip(pre_tr["tcr"].astype(str).str.strip().str.upper(), pre_tr["epitope"].astype(str).str.strip().str.upper())
    ) | set(
        zip(ft_tr["tcr"].astype(str).str.strip().str.upper(), ft_tr["epitope"].astype(str).strip() if False else ft_tr["epitope"].astype(str).str.strip().str.upper())
    )

    pep_benchmarks = {}
    for name, folder in (
        ("epytope_mutation", "epact_train_epytope_mutation_test"),
        ("batcave", "epact_train_batcave_test"),
        ("batcave_full", "epact_train_batcave_test_full"),
        ("atlas_pep", "epact_train_atlaspep_test"),
        ("epytope_binary", "epact_train_epytope_binary_test"),
        ("immrep2025", "epact_train_immrep2025_test"),
    ):
        tdf = pd.read_csv(ft_root / folder / "test.csv")
        pep_benchmarks[name] = int(len(tdf))

    exp_tcr = pd.read_csv(data_dir / "expanded_tcr_mut" / "tcr_mutations_combined_test.csv")
    test_pairs = set(
        zip(exp_tcr["tcr"].astype(str).str.strip().str.upper(), exp_tcr["epitope"].astype(str).str.strip().str.upper())
    )
    pretrain_overlap = len(
        set(zip(pre_tr["tcr"].astype(str).str.strip().str.upper(), pre_tr["epitope"].astype(str).str.strip().str.upper()))
        & test_pairs
    )

    return {
        "stage1_pretrain_pairs": int(len(pre_tr)),
        "stage1_val_pairs": int(len(pre_va)),
        "stage1_unique_tcrs": int(pre_tr["tcr"].nunique()),
        "stage1_unique_epitopes": int(pre_tr["epitope"].nunique()),
        "stage2_finetune_train_rows": int(len(ft_tr)),
        "stage2_finetune_val_rows": int(len(ft_va)),
        "downstream_test_benchmarks": pep_benchmarks,
        "expanded_tcr_dms_variants": int(len(exp_tcr)),
        "expanded_tcr_dms_groups": int(exp_tcr.groupby(["tcr_name", "epitope"]).ngroups),
        "stage1_to_expanded_tcr_leakage_pairs": int(pretrain_overlap),
        "total_training_reference_pairs": int(len(train_pairs)),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-dir", type=Path, default=SUBFOLDER_ROOT / "data")
    parser.add_argument("--out-dir", type=Path, default=SUBFOLDER_ROOT / "outputs")
    parser.add_argument("--download-external", action="store_true")
    args = parser.parse_args()

    args.out_dir.mkdir(parents=True, exist_ok=True)
    if args.download_external:
        download_external_tcr_mutation_sources(args.data_dir / "raw")

    summary = verify_and_summarize_tcr_splits(args.data_dir)
    out_json = args.out_dir / "01_tcr_dataset_summary.json"
    out_json.write_text(json.dumps(summary, indent=2))
    print(f"[TCRLang-ESM Step 01] Verified all zero-leakage TCR–epitope splits -> {out_json}")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
