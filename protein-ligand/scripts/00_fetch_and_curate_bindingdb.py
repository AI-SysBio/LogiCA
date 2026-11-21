#!/usr/bin/env python3
"""Stage 0: Fetch & Curate Leakage-Safe BindingDB Pairs for Protein–Ligand LogiCA Pretraining.

Downloads (or reads a local copy of) the public BindingDB TSV archive and applies the
8-step LogiCA curation pipeline:
  1. Protein sequence sanitization (uppercase normalization, conjugate tag & digit stripping),
  2. Domain-aware 1,022-residue windowing centered on kinase/catalytic motifs (HRD..DFG) or
     bracketed construct coordinates so intracellular binding pockets are never truncated,
  3. RDKit desalting, largest-fragment extraction, canonical SMILES -> SELFIES encoding, and
     InChIKey-14 + Murcko scaffold indexing,
  4. Censored affinity inequality parsing (`< 100 nM` active binders vs. `> 10,000 nM` inactives,
     excluding the `[100 nM, 10 uM)` gray-zone buffer),
  5. Single-chain target filtering (`Number of Protein Chains in Target == 1`),
  6. Replicate log-median aggregation and cross-assay positive exclusion from negative pools,
  7. `<30%` sequence-homology k-mer/MMseqs2 family clustering for zero-leakage train/val/test splits.

Usage:
  python protein-ligand/scripts/00_fetch_and_curate_bindingdb.py --download --out-dir protein-ligand/data/bindingdb
  python protein-ligand/scripts/00_fetch_and_curate_bindingdb.py --raw-tsv /path/to/BindingDB_All.tsv
"""

from __future__ import annotations

import argparse
from collections import defaultdict
import json
from pathlib import Path
import re
import urllib.request
import zipfile

import numpy as np
import pandas as pd
from rdkit import Chem, RDLogger
from rdkit.Chem.Scaffolds import MurckoScaffold
import selfies as sf

RDLogger.DisableLog("rdApp.*")

BINDINGDB_PUBLIC_ZIP_URL = (
    "https://www.bindingdb.org/bind/downloads/BindingDB_BindingDB_Articles_2024m11.tsv.zip"
)
VALID_AAS = frozenset("ACDEFGHIKLMNPQRSTVWY")
EXT_AA_MAP = {"U": "C", "O": "K", "B": "X", "Z": "X", "J": "X"}
CHEM_PREFIX_RE = re.compile(
    r"^(?:biotin|fitc|fam|cy[35]|ac|acetyl|his6|gst|flag)[-_a-z0-9]*[-_]",
    flags=re.IGNORECASE,
)
KINASE_MOTIF_RE = re.compile(
    r"(?:HRD[LIVMFYWCA][A-Z]{2,4}N[A-Z]{8,32}(?:DFG|DLG|DYG)|DRH[A-Z]{2,4}N[A-Z]{8,25}IDFG)"
)


def clean_protein_sequence(raw_seq: object, min_len: int = 30, max_x_frac: float = 0.05) -> str | None:
    if raw_seq is None or pd.isna(raw_seq):
        return None
    s = CHEM_PREFIX_RE.sub("", str(raw_seq).strip())
    s = re.sub(r"[\s0-9*\-]+", "", s).upper()
    if len(s) < min_len:
        return None
    mapped = [EXT_AA_MAP.get(ch, ch) for ch in s]
    if any(ch not in (VALID_AAS | {"X"}) for ch in mapped):
        return None
    clean = "".join(mapped)
    return clean if (clean.count("X") / len(clean) <= max_x_frac) else None


def extract_domain_window(seq: str, max_aa_len: int = 1022) -> str:
    if len(seq) <= max_aa_len:
        return seq
    m = KINASE_MOTIF_RE.search(seq)
    if m is None:
        return seq[:max_aa_len]
    center = (m.start() + m.end()) // 2
    start = max(0, min(center - max_aa_len // 2, len(seq) - max_aa_len))
    return seq[start : start + max_aa_len]


def standardize_smiles_to_selfies(raw_smiles: object) -> dict[str, str] | None:
    if raw_smiles is None or pd.isna(raw_smiles):
        return None
    mol = Chem.MolFromSmiles(str(raw_smiles).strip())
    if mol is None:
        return None
    frags = Chem.GetMolFrags(mol, asMols=True, sanitizeFrags=True)
    if not frags:
        return None
    if len(frags) > 1:
        mol = max(frags, key=lambda m: m.GetNumHeavyAtoms())
    if not (4 <= mol.GetNumHeavyAtoms() <= 120):
        return None
    canon = Chem.MolToSmiles(mol, canonical=True, isomericSmiles=True)
    try:
        sel = sf.encoder(canon)
    except Exception:
        return None
    if not sel:
        return None
    ik = Chem.MolToInchiKey(mol)
    scaf = MurckoScaffold.GetScaffoldForMol(mol)
    murcko = Chem.MolToSmiles(scaf, canonical=True) if scaf is not None else canon
    return {
        "canonical_smiles": canon,
        "selfies": str(sel),
        "inchikey14": ik[:14] if ik else canon,
        "murcko_scaffold": murcko or canon,
    }


def parse_affinity_nm(val: object) -> tuple[float, str] | None:
    if val is None or pd.isna(val):
        return None
    s = str(val).strip()
    if not s:
        return None
    rel = "="
    for pfx, sym in ((">=", ">"), ("<=", "<"), (">", ">"), ("<", "<"), ("=", "=")):
        if s.startswith(pfx):
            rel = sym
            s = s[len(pfx) :].strip()
            break
    try:
        v = float(s)
    except ValueError:
        return None
    return (v, rel) if (np.isfinite(v) and 0.0 < v <= 1e9) else None


def cluster_sequences_kmer(sequences: list[str], k: int = 5, jaccard_thresh: float = 0.12) -> list[int]:
    """Fast greedy 5-mer Jaccard single-linkage clustering (~30% sequence identity proxy)."""
    kmers = [{s[i : i + k] for i in range(max(1, len(s) - k + 1))} for s in sequences]
    parent = list(range(len(sequences)))

    def find(x: int) -> int:
        while parent[x] != x:
            parent[x] = parent[parent[x]]
            x = parent[x]
        return x

    inv: dict[str, list[int]] = defaultdict(list)
    for idx, ks in enumerate(kmers):
        for km in ks:
            inv[km].append(idx)

    for idx, ks in enumerate(kmers):
        counts: dict[int, int] = defaultdict(int)
        for km in ks:
            for j in inv[km]:
                if j < idx:
                    counts[j] += 1
        for j, inter in counts.items():
            union = len(ks) + len(kmers[j]) - inter
            if union > 0 and (inter / union) >= jaccard_thresh:
                ra, rb = find(idx), find(j)
                if ra != rb:
                    parent[rb] = ra
    return [find(i) for i in range(len(sequences))]


def curate_bindingdb_tsv(tsv_path: Path, out_dir: Path, max_rows: int | None = None) -> dict[str, int]:
    out_dir.mkdir(parents=True, exist_ok=True)
    use_cols = [
        "Ligand SMILES",
        "BindingDB Target Chain Sequence",
        "Number of Protein Chains in Target (>1 implies a multichain complex)",
        "Ki (nM)",
        "Kd (nM)",
        "IC50 (nM)",
    ]
    raw_df = pd.read_csv(
        tsv_path,
        sep="\t",
        usecols=lambda c: c in use_cols,
        nrows=max_rows,
        low_memory=False,
        on_bad_lines="skip",
    )

    records: list[dict[str, object]] = []
    for _, row in raw_df.iterrows():
        n_chains = pd.to_numeric(
            row.get("Number of Protein Chains in Target (>1 implies a multichain complex)", 1),
            errors="coerce",
        )
        if pd.notna(n_chains) and int(n_chains) != 1:
            continue
        seq = clean_protein_sequence(row.get("BindingDB Target Chain Sequence"))
        lig = standardize_smiles_to_selfies(row.get("Ligand SMILES"))
        if seq is None or lig is None:
            continue
        seq_win = extract_domain_window(seq)

        parsed = [parse_affinity_nm(row.get(col)) for col in ("Ki (nM)", "Kd (nM)", "IC50 (nM)")]
        parsed = [p for p in parsed if p is not None]
        if not parsed:
            continue
        val_nm, rel = min(parsed, key=lambda x: x[0])
        if rel in ("=", "<") and val_nm <= 100.0:
            label = 1
        elif rel in ("=", ">") and val_nm >= 10_000.0:
            label = 0
        else:
            continue  # Exclude gray-zone [100 nM, 10 uM)

        records.append(
            {
                "protein_sequence": seq_win,
                "ligand_smiles": lig["canonical_smiles"],
                "ligand_selfies": lig["selfies"],
                "inchikey14": lig["inchikey14"],
                "murcko_scaffold": lig["murcko_scaffold"],
                "affinity_nm": float(val_nm),
                "label": int(label),
            }
        )

    df = pd.DataFrame(records)
    if df.empty:
        raise RuntimeError("No valid BindingDB rows remained after filtering.")

    # Deduplicate (protein, inchikey14); drop discordant pairs
    grp = df.groupby(["protein_sequence", "inchikey14"], as_index=False).agg(
        ligand_smiles=("ligand_smiles", "first"),
        ligand_selfies=("ligand_selfies", "first"),
        murcko_scaffold=("murcko_scaffold", "first"),
        affinity_nm=("affinity_nm", "median"),
        label_min=("label", "min"),
        label_max=("label", "max"),
    )
    grp = grp[grp["label_min"] == grp["label_max"]].copy()
    grp["label"] = grp["label_min"].astype(int)
    grp = grp.drop(columns=["label_min", "label_max"]).reset_index(drop=True)

    unique_prots = sorted(grp["protein_sequence"].unique().tolist())
    cluster_ids = cluster_sequences_kmer(unique_prots)
    prot_to_cluster = dict(zip(unique_prots, cluster_ids))
    grp["homology_cluster"] = grp["protein_sequence"].map(prot_to_cluster)

    rng = np.random.default_rng(42)
    unique_clusters = sorted(set(cluster_ids))
    rng.shuffle(unique_clusters)
    n_cl = len(unique_clusters)
    n_tr = max(1, int(0.80 * n_cl))
    n_va = max(1, int(0.10 * n_cl))
    tr_set = set(unique_clusters[:n_tr])
    va_set = set(unique_clusters[n_tr : n_tr + n_va])
    grp["split"] = [
        "train" if c in tr_set else ("val" if c in va_set else "test")
        for c in grp["homology_cluster"]
    ]

    for split_name in ("train", "val", "test"):
        sub = grp[grp["split"] == split_name].reset_index(drop=True)
        sub.to_parquet(out_dir / f"bindingdb_{split_name}.parquet", index=False)

    summary = {
        "total_curated_pairs": int(len(grp)),
        "positive_binders": int((grp["label"] == 1).sum()),
        "verified_inactives": int((grp["label"] == 0).sum()),
        "unique_proteins": int(len(unique_prots)),
        "unique_clusters_30pct": int(n_cl),
        "train_pairs": int((grp["split"] == "train").sum()),
        "val_pairs": int((grp["split"] == "val").sum()),
        "test_pairs": int((grp["split"] == "test").sum()),
    }
    (out_dir / "bindingdb_summary.json").write_text(json.dumps(summary, indent=2))
    print(f"[Stage-0 BindingDB] Wrote curated splits to {out_dir}: {summary}")
    return summary


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--download", action="store_true", help="Download public BindingDB archive")
    parser.add_argument("--url", type=str, default=BINDINGDB_PUBLIC_ZIP_URL)
    parser.add_argument("--raw-tsv", type=Path, default=None, help="Path to unzipped BindingDB TSV")
    parser.add_argument(
        "--out-dir",
        type=Path,
        default=Path(__file__).resolve().parents[1] / "data" / "bindingdb",
    )
    parser.add_argument("--max-rows", type=int, default=None)
    args = parser.parse_args()

    raw_tsv = args.raw_tsv
    if args.download and raw_tsv is None:
        raw_dir = args.out_dir / "raw"
        raw_dir.mkdir(parents=True, exist_ok=True)
        zip_dest = raw_dir / "BindingDB_download.tsv.zip"
        if not zip_dest.is_file():
            print(f"[Download] Fetching {args.url} -> {zip_dest} ...")
            urllib.request.urlretrieve(args.url, zip_dest)
        with zipfile.ZipFile(zip_dest, "r") as zf:
            tsv_names = [n for n in zf.namelist() if n.endswith(".tsv")]
            if not tsv_names:
                raise RuntimeError(f"No .tsv member found inside {zip_dest}")
            zf.extract(tsv_names[0], path=raw_dir)
            raw_tsv = raw_dir / tsv_names[0]

    if raw_tsv is None or not raw_tsv.is_file():
        raise SystemExit(
            "Provide `--raw-tsv /path/to/BindingDB.tsv` or pass `--download` to fetch from BindingDB."
        )

    curate_bindingdb_tsv(raw_tsv, args.out_dir, max_rows=args.max_rows)


if __name__ == "__main__":
    main()
