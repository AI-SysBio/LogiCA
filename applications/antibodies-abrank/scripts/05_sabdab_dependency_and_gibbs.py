#!/usr/bin/env python3
"""Step 5: SAbDab Download, `<5 Å` Structural Contact Extraction, Cross-Modal Dependency Maps,
and `AbLang2`–`ESM-2` Coordinate Gibbs Evaluation.

Provides a complete structural antibody–antigen evaluation pipeline built on `logica`:
  1. **SAbDab2 Download & Curation**:
       Fetches the live SAbDab2 summary (`https://sabdab.opig.stats.ox.ac.uk/api/download/all-summary`),
       filters to non-redundant high-resolution protein-antigen antibody (`VH|VL`) and nanobody (`VHH|`)
       crystal complexes, and downloads RCSB mmCIF structures (`https://files.rcsb.org/download/<PDB>.cif.gz`).
  2. **Exact `<5.0 Å` Heavy-Atom Contact & Interface Extraction (Pure Python + NumPy)**:
       Parses mmCIF coordinates with zero extra structural dependencies, aligns crystal chains to
       SAbDab's `VH`/`VL` variable domains, and computes exact residue-level and residue-pair `<5 Å`
       paratope–epitope heavy-atom contacts alongside SAbDab CDR-H1/H2/H3/L1/L2/L3 annotations.
  3. **Cross-Modal Dependency Maps (`AbLang2-Paired` + `ESM-2`)**:
       Computes single-pass zero-shot KL paratope profiles (`detect_kl_hotspots`) and 2D Antigen ->
       Antibody perturbation dependency maps (`dependency_map`), evaluating `<5 Å` paratope contact
       AUROC/AUPR, 2D residue-pair contact AUROC/AUPR, and CDR loop enrichment.
  4. **Antigen-Conditioned Coordinate Gibbs Evaluation**:
       Runs `model.gibbs_sample(...)` over SAbDab-annotated `CDR-H3` loops (or `<5 Å` true paratope /
       zero-shot KL hotspots), benchmarking Antigen-Conditioned (`holo`/guided) vs. Unconditional
       (`apo` `AbLang2`) Gibbs sampling on native sequence recovery, `<5 Å` contact recovery,
       native-logit score gain, and effective sample diversity (`N_eff`).

Usage:
  # Download SAbDab2 + evaluate top-25 high-resolution complexes on GPU
  python applications/antibodies-abrank/scripts/05_sabdab_dependency_and_gibbs.py \\
      --checkpoint applications/antibodies-abrank/outputs/ablang_esm_logica/ablang_esm_logica.pt \\
      --max-complexes 25 --device cuda:0

  # Evaluate specific well-known PDB structures (e.g. Trastuzumab-HER2 1N8Z, HyHEL-10 1FDL)
  python applications/antibodies-abrank/scripts/05_sabdab_dependency_and_gibbs.py \\
      --pdb-ids 1N8Z,1FDL,2FJG --device cpu

  # Fast end-to-end live CPU smoke test (downloads real SAbDab2 + 1N8Z/1FDL mmCIFs)
  python applications/antibodies-abrank/scripts/05_sabdab_dependency_and_gibbs.py --smoke-test
"""

from __future__ import annotations

import argparse
import concurrent.futures
from dataclasses import dataclass
from difflib import SequenceMatcher
import gzip
import json
from pathlib import Path
import re
import shlex
import sys
import time
import urllib.request

import matplotlib

matplotlib.use("Agg")
import numpy as np
import pandas as pd
import torch

SCRIPT_DIR = Path(__file__).resolve().parent
SUBFOLDER_ROOT = SCRIPT_DIR.parent
REPO_ROOT = SUBFOLDER_ROOT.parents[1]
for _p in (str(REPO_ROOT), str(SCRIPT_DIR)):
    if _p not in sys.path:
        sys.path.insert(0, _p)

from abrank_loader import (  # noqa: E402
    detect_cdrh3_positions_0based,
    format_antibody_sequence,
)
from logica import (  # noqa: E402
    DependencyMapper,
    LogiCA,
    evaluate_contact_localization,
    plot_dependency_map,
    plot_gibbs_design,
)

SABDAB2_ALL_SUMMARY_URL = "https://sabdab.opig.stats.ox.ac.uk/api/download/all-summary"
RCSB_MMCIF_URL_TMPL = "https://files.rcsb.org/download/{pdb_id}.cif.gz"
USER_AGENT = "Mozilla/5.0 (AI-SysBio-LogiCA-SAbDab/1.0)"

AA3_TO_AA1: dict[str, str] = {
    "ALA": "A",
    "ARG": "R",
    "ASN": "N",
    "ASP": "D",
    "CYS": "C",
    "GLN": "Q",
    "GLU": "E",
    "GLY": "G",
    "HIS": "H",
    "ILE": "I",
    "LEU": "L",
    "LYS": "K",
    "MET": "M",
    "PHE": "F",
    "PRO": "P",
    "SER": "S",
    "THR": "T",
    "TRP": "W",
    "TYR": "Y",
    "VAL": "V",
    # Common modified protein residues
    "MSE": "M",
    "SEC": "C",
    "PYL": "K",
    "SEP": "S",
    "TPO": "T",
    "PTR": "Y",
}
VALID_AA_RE = re.compile(r"^[ACDEFGHIKLMNPQRSTVWY]+$")


@dataclass
class ChainStructure:
    """Resolved heavy-atom coordinates per residue for one polymer chain."""

    chain_id: str
    sequence: str
    residue_coords: list[np.ndarray]  # each (n_heavy_atoms, 3) float32
    residue_ca: np.ndarray  # (L, 3) float32 centroid / CA coordinate for fast prefiltering


@dataclass
class CuratedSAbDabComplex:
    """Curated SAbDab antibody–antigen complex with ground-truth `<5 Å` heavy-atom contacts."""

    instance_id: str
    pdb_id: str
    hchain: str
    lchain: str
    antigen_chains: list[str]
    antigen_name: str
    resolution: float
    vh_seq: str
    vl_seq: str
    ab_paired_seq: str  # "VH|VL" or "VHH|"
    ag_seq: str
    ag_window_seq: str
    ag_window_start: int
    cdr_ranges: dict[str, tuple[int, int]]  # 0-based [start, end) in concatenated VH+VL valid-token space
    ab_contact_mask: np.ndarray  # (len(vh)+len(vl),) int {0,1}
    ab_min_dist: np.ndarray  # (len(vh)+len(vl),) float32
    pair_contact_matrix: np.ndarray  # (len(vh)+len(vl), len(ag_window_seq)) int {0,1}
    pair_min_dist: np.ndarray  # (len(vh)+len(vl), len(ag_window_seq)) float32
    n_ag_contacts: int


def _download_file(url: str, dest: Path, *, timeout: int = 90) -> Path:
    dest.parent.mkdir(parents=True, exist_ok=True)
    if dest.exists() and dest.stat().st_size > 0:
        return dest
    req = urllib.request.Request(url, headers={"User-Agent": USER_AGENT})
    tmp = dest.with_suffix(dest.suffix + ".tmp")
    with urllib.request.urlopen(req, timeout=timeout) as resp, open(tmp, "wb") as out:
        while True:
            chunk = resp.read(1 << 16)
            if not chunk:
                break
            out.write(chunk)
    tmp.replace(dest)
    return dest


def normalize_pdb_id(raw_pdb: str) -> str:
    """Convert SAbDab2 extended IDs (e.g. `pdb_00001n8z`) or 4-char IDs (`1n8z`) to `1N8Z`."""
    s = str(raw_pdb).strip()
    if s.lower().startswith("pdb_"):
        s = s[4:]
        if len(s) == 8 and s.startswith("0000"):
            s = s[4:]
    return s.upper()


def _clean_seq(val: object) -> str:
    if val is None or (isinstance(val, float) and np.isnan(val)):
        return ""
    s = re.sub(r"[^A-Z]", "", str(val).strip().upper())
    return s


def _extract_protein_antigen_chains(ag_chain_str: object, ag_type_str: object) -> list[str]:
    """Select only protein/peptide antigen chains from SAbDab pipe-separated annotations."""
    if ag_chain_str is None or (isinstance(ag_chain_str, float) and np.isnan(ag_chain_str)):
        return []
    chains = [c.strip() for c in str(ag_chain_str).split("|") if c.strip() and c.strip().upper() != "NA"]
    types = [t.strip().upper() for t in str(ag_type_str or "").split("|")]
    if len(types) == len(chains):
        prot_chains = [
            c for c, t in zip(chains, types) if ("PROTEIN" in t or "PEPTIDE" in t) and "NUCLEIC" not in t
        ]
    else:
        t_all = str(ag_type_str or "").upper()
        prot_chains = chains if ("PROTEIN" in t_all or "PEPTIDE" in t_all) else []
    return list(dict.fromkeys(prot_chains))


def download_or_load_sabdab_summary(
    sabdab_dir: Path,
    summary_path: Path | None = None,
    *,
    force_download: bool = False,
) -> pd.DataFrame:
    """Download (or load cached) SAbDab2 full summary table."""
    dest = summary_path if summary_path is not None else sabdab_dir / "sabdab2_all_summary.csv"
    if force_download and dest.exists():
        dest.unlink()
    if not dest.exists():
        print(f"[SAbDab] Downloading official SAbDab2 summary -> {dest} ...")
        _download_file(SABDAB2_ALL_SUMMARY_URL, dest, timeout=120)
    sep = "\t" if dest.suffix.lower() == ".tsv" else ","
    return pd.read_csv(dest, sep=sep, low_memory=False)


def parse_mmcif_chains(cif_path: Path, target_chains: set[str]) -> dict[str, ChainStructure]:
    """Pure-Python mmCIF `_atom_site` heavy-atom parser (zero external structural deps).

    Supports gzipped `.cif.gz` and plain `.cif`. Matches both `auth_asym_id` (SAbDab convention)
    and `label_asym_id` fallback, keeping model 1, altLoc in {`.`, `?`, `A`}, and non-hydrogen
    heavy atoms of standard protein residues.
    """
    opener = gzip.open if cif_path.suffix == ".gz" else open
    col_names: list[str] = []
    in_atom_site = False

    # Maps chain_id -> ordered dict[(seq_id, ins_code)] -> (aa1, list[[x,y,z]], ca_xyz|None)
    auth_chains: dict[str, dict[tuple[str, str], tuple[str, list[list[float]], list[float] | None]]] = {}
    label_chains: dict[str, dict[tuple[str, str], tuple[str, list[list[float]], list[float] | None]]] = {}

    idx_group = idx_atom = idx_alt = idx_comp = idx_auth_ch = idx_lbl_ch = -1
    idx_auth_seq = idx_lbl_seq = idx_ins = idx_x = idx_y = idx_z = idx_elem = idx_model = -1
    indices_ready = False

    def _split_cif_line(ln: str) -> list[str]:
        if "'" not in ln and '"' not in ln:
            return ln.split()
        try:
            return shlex.split(ln)
        except ValueError:
            return ln.split()

    with opener(cif_path, "rt", encoding="utf-8", errors="replace") as fh:
        for raw_line in fh:
            line = raw_line.strip()
            if not line or line.startswith("#"):
                continue
            if line.startswith("loop_"):
                if in_atom_site and indices_ready:
                    break
                col_names = []
                in_atom_site = False
                continue
            if line.startswith("_atom_site."):
                in_atom_site = True
                col_names.append(line.split()[0].split(".", 1)[1])
                continue
            if in_atom_site and line.startswith("_"):
                if indices_ready:
                    break
                in_atom_site = False
                col_names = []
                continue
            if not in_atom_site:
                continue

            if not indices_ready:
                cmap = {name: i for i, name in enumerate(col_names)}
                idx_group = cmap.get("group_PDB", -1)
                idx_atom = cmap.get("label_atom_id", cmap.get("auth_atom_id", -1))
                idx_alt = cmap.get("label_alt_id", -1)
                idx_comp = cmap.get("label_comp_id", cmap.get("auth_comp_id", -1))
                idx_auth_ch = cmap.get("auth_asym_id", -1)
                idx_lbl_ch = cmap.get("label_asym_id", -1)
                idx_auth_seq = cmap.get("auth_seq_id", -1)
                idx_lbl_seq = cmap.get("label_seq_id", -1)
                idx_ins = cmap.get("pdbx_PDB_ins_code", -1)
                idx_x = cmap.get("Cartn_x", -1)
                idx_y = cmap.get("Cartn_y", -1)
                idx_z = cmap.get("Cartn_z", -1)
                idx_elem = cmap.get("type_symbol", -1)
                idx_model = cmap.get("pdbx_PDB_model_num", -1)
                indices_ready = True

            if line.startswith(";"):
                continue
            toks = _split_cif_line(line)
            if len(toks) < len(col_names):
                continue

            if idx_model >= 0 and toks[idx_model] not in ("1", ".", "?"):
                continue
            if idx_alt >= 0 and toks[idx_alt] not in (".", "?", "A", "1"):
                continue

            comp = toks[idx_comp].upper()
            aa1 = AA3_TO_AA1.get(comp)
            if aa1 is None:
                continue

            # Keep ATOM records and modified protein HETATM (e.g., MSE)
            if idx_group >= 0 and toks[idx_group] not in ("ATOM", "HETATM"):
                continue
            if idx_elem >= 0 and toks[idx_elem].upper() in ("H", "D"):
                continue
            atom_name = toks[idx_atom].strip("\"'") if idx_atom >= 0 else ""
            if atom_name.startswith("H"):
                continue

            auth_ch = toks[idx_auth_ch] if idx_auth_ch >= 0 else ""
            lbl_ch = toks[idx_lbl_ch] if idx_lbl_ch >= 0 else ""
            want_auth = auth_ch in target_chains
            want_lbl = lbl_ch in target_chains
            if not want_auth and not want_lbl:
                continue

            try:
                xyz = [float(toks[idx_x]), float(toks[idx_y]), float(toks[idx_z])]
            except ValueError:
                continue

            ins = "" if idx_ins < 0 or toks[idx_ins] in (".", "?") else toks[idx_ins]
            if want_auth:
                seq_key = (toks[idx_auth_seq] if idx_auth_seq >= 0 else toks[idx_lbl_seq], ins)
                bucket = auth_chains.setdefault(auth_ch, {})
                if seq_key not in bucket:
                    bucket[seq_key] = (aa1, [xyz], xyz if atom_name == "CA" else None)
                else:
                    old_aa, coords, ca = bucket[seq_key]
                    coords.append(xyz)
                    if ca is None and atom_name == "CA":
                        bucket[seq_key] = (old_aa, coords, xyz)

            if want_lbl:
                seq_key_l = (toks[idx_lbl_seq] if idx_lbl_seq >= 0 else toks[idx_auth_seq], ins)
                bucket_l = label_chains.setdefault(lbl_ch, {})
                if seq_key_l not in bucket_l:
                    bucket_l[seq_key_l] = (aa1, [xyz], xyz if atom_name == "CA" else None)
                else:
                    old_aa, coords, ca = bucket_l[seq_key_l]
                    coords.append(xyz)
                    if ca is None and atom_name == "CA":
                        bucket_l[seq_key_l] = (old_aa, coords, xyz)

    out: dict[str, ChainStructure] = {}
    for ch in target_chains:
        raw_res = auth_chains.get(ch) or label_chains.get(ch)
        if not raw_res:
            continue
        seq_chars: list[str] = []
        res_coords: list[np.ndarray] = []
        ca_list: list[np.ndarray] = []
        for aa1, coords, ca in raw_res.values():
            arr = np.asarray(coords, dtype=np.float32)
            seq_chars.append(aa1)
            res_coords.append(arr)
            ca_list.append(np.asarray(ca, dtype=np.float32) if ca is not None else arr.mean(axis=0))
        out[ch] = ChainStructure(
            chain_id=ch,
            sequence="".join(seq_chars),
            residue_coords=res_coords,
            residue_ca=np.stack(ca_list, axis=0),
        )
    return out


def align_domain_to_crystal(domain_seq: str, chain: ChainStructure) -> list[int | None]:
    """Map each 0-based position in `domain_seq` (SAbDab `VH` or `VL`) to its crystal residue index.

    Returns a list of length `len(domain_seq)` whose entries are crystal residue indices (or `None`
    for unresolved residues in crystal density).
    """
    c_seq = chain.sequence
    exact_idx = c_seq.find(domain_seq)
    if exact_idx >= 0:
        return [exact_idx + i for i in range(len(domain_seq))]

    # Local sequence alignment via longest contiguous matching blocks (handles missing density / CH1 tail)
    matcher = SequenceMatcher(None, domain_seq, c_seq, autojunk=False)
    mapping: list[int | None] = [None] * len(domain_seq)
    matched = 0
    for block in matcher.get_matching_blocks():
        if block.size == 0:
            continue
        for k in range(block.size):
            mapping[block.a + k] = block.b + k
            matched += 1
    if matched < max(20, int(0.70 * len(domain_seq))):
        return [None] * len(domain_seq)
    return mapping


def compute_heavy_atom_contact_matrices(
    ab_coords: list[np.ndarray | None],
    ab_ca: list[np.ndarray | None],
    ag_coords: list[np.ndarray],
    ag_ca: np.ndarray,
    *,
    contact_cutoff: float = 5.0,
    ca_prefilter: float = 18.0,
) -> tuple[np.ndarray, np.ndarray]:
    """Compute exact residue-by-residue minimum heavy-atom distances `(L_ab, L_ag)` and `<cutoff` mask."""
    n_ab = len(ab_coords)
    n_ag = len(ag_coords)
    min_dist = np.full((n_ab, n_ag), np.inf, dtype=np.float32)
    cutoff_sq = float(contact_cutoff) ** 2
    pref_sq = float(ca_prefilter) ** 2

    for i in range(n_ab):
        ca_i = ab_ca[i]
        xyz_i = ab_coords[i]
        if ca_i is None or xyz_i is None:
            continue
        d_ca_sq = np.sum((ag_ca - ca_i[None, :]) ** 2, axis=-1)
        cand_j = np.flatnonzero(d_ca_sq <= pref_sq)
        for j in cand_j:
            xyz_j = ag_coords[int(j)]
            diff = xyz_i[:, None, :] - xyz_j[None, :, :]
            d2 = float(np.min(np.sum(diff * diff, axis=-1)))
            min_dist[i, int(j)] = float(np.sqrt(d2)) if d2 < 1e9 else np.inf

    contacts = (min_dist <= contact_cutoff).astype(np.int32)
    # Use cutoff_sq sanity check only indirectly; exact comparison above is <= contact_cutoff
    _ = cutoff_sq
    return min_dist, contacts


def _find_cdr_range(domain_seq: str, cdr_seq: str, offset: int = 0) -> tuple[int, int] | None:
    c = _clean_seq(cdr_seq)
    if not c:
        return None
    pos = domain_seq.find(c)
    if pos >= 0:
        return (offset + pos, offset + pos + len(c))
    return None


def curate_sabdab_complexes(
    summary_df: pd.DataFrame,
    structures_dir: Path,
    *,
    max_complexes: int = 25,
    max_resolution: float = 3.0,
    contact_cutoff: float = 5.0,
    max_ag_window: int = 160,
    pdb_ids: list[str] | None = None,
    paired_only: bool = False,
    vhh_only: bool = False,
    num_workers: int = 6,
) -> list[CuratedSAbDabComplex]:
    """Filter SAbDab2 summary, download RCSB mmCIFs, and extract `<5 Å` ground-truth contacts."""
    df = summary_df.copy()
    df["pdb_code"] = df["PDB"].map(normalize_pdb_id)
    df["res_num"] = pd.to_numeric(df.get("resolution", np.nan), errors="coerce")

    if pdb_ids:
        want = [p.strip().upper() for p in pdb_ids if p.strip()]
        df = df[df["pdb_code"].isin(want)].copy()
        order_map = {p: i for i, p in enumerate(want)}
        df["_order"] = df["pdb_code"].map(order_map)
        df = df.sort_values(["_order", "res_num"], ascending=[True, True])
    else:
        df = df[df["res_num"].notna() & (df["res_num"] <= max_resolution)].copy()
        df = df.sort_values("res_num", ascending=True)

    candidates: list[dict] = []
    seen_vh: set[str] = set()
    seen_pdb: set[str] = set()

    for row in df.itertuples(index=False):
        pdb_code = str(getattr(row, "pdb_code"))
        if len(pdb_code) != 4:
            continue
        if not pdb_ids and pdb_code in seen_pdb:
            continue

        vh = _clean_seq(getattr(row, "VH", ""))
        vl = _clean_seq(getattr(row, "VL", ""))
        if not vh or len(vh) < 85 or len(vh) > 165 or not VALID_AA_RE.match(vh):
            continue
        if vl and (len(vl) < 80 or len(vl) > 140 or not VALID_AA_RE.match(vl)):
            continue
        if paired_only and not vl:
            continue
        if vhh_only and vl:
            continue

        hch = str(getattr(row, "Hchain", "")).strip()
        lch_raw = getattr(row, "Lchain", "")
        lch = "" if pd.isna(lch_raw) or str(lch_raw).strip() in ("", "NA", "nan") else str(lch_raw).strip()
        if not hch or hch == "NA":
            continue

        ag_chains = _extract_protein_antigen_chains(
            getattr(row, "antigen_chain", ""),
            getattr(row, "antigen_type", ""),
        )
        if not ag_chains:
            continue

        # Deduplicate near-identical antibodies (by VH sequence) so benchmark complexes are diverse
        if not pdb_ids and vh in seen_vh:
            continue
        seen_vh.add(vh)
        seen_pdb.add(pdb_code)

        candidates.append(
            {
                "instance_id": str(getattr(row, "INSTANCE", f"{pdb_code}-{hch}-{lch}")),
                "pdb_id": pdb_code,
                "hchain": hch,
                "lchain": lch,
                "ag_chains": ag_chains,
                "antigen_name": str(getattr(row, "antigen_name", "") or "")[:80],
                "resolution": float(getattr(row, "res_num", np.nan)),
                "vh": vh,
                "vl": vl,
                "cdrs_raw": {
                    "CDR-H1": getattr(row, "_8", None) if not hasattr(row, "CDR_H1") else getattr(row, "CDR_H1", None),
                },
                "row_dict": row._asdict(),
            }
        )
        # Fetch a modest buffer (~2.5x) in case any crystal structure lacks dense interface contacts
        if not pdb_ids and len(candidates) >= max(max_complexes * 3, max_complexes + 12):
            break

    # Resolve original hyphenated CDR column names cleanly from DataFrame index lookup
    summary_by_inst = (
        df.drop_duplicates("INSTANCE").set_index("INSTANCE") if "INSTANCE" in df.columns else None
    )

    # Download mmCIF files in parallel
    structures_dir.mkdir(parents=True, exist_ok=True)
    unique_pdbs = list(dict.fromkeys(c["pdb_id"] for c in candidates))
    print(f"[SAbDab] Fetching up to {len(unique_pdbs)} RCSB mmCIF structures into {structures_dir} ...")

    def _fetch_pdb(pid: str) -> tuple[str, Path | None]:
        dest = structures_dir / f"{pid}.cif.gz"
        try:
            _download_file(RCSB_MMCIF_URL_TMPL.format(pdb_id=pid), dest, timeout=60)
            return pid, dest
        except Exception as exc:
            print(f"  [WARN] Failed downloading {pid}.cif.gz: {exc}")
            return pid, None

    cif_map: dict[str, Path] = {}
    with concurrent.futures.ThreadPoolExecutor(max_workers=max(1, num_workers)) as ex:
        for pid, path in ex.map(_fetch_pdb, unique_pdbs):
            if path is not None:
                cif_map[pid] = path

    curated: list[CuratedSAbDabComplex] = []
    for cand in candidates:
        if len(curated) >= max_complexes:
            break
        pid = cand["pdb_id"]
        cif_path = cif_map.get(pid)
        if cif_path is None:
            continue

        want_chains = {cand["hchain"], *cand["ag_chains"]}
        if cand["lchain"]:
            want_chains.add(cand["lchain"])

        try:
            parsed = parse_mmcif_chains(cif_path, want_chains)
        except Exception as exc:
            print(f"  [WARN] Failed parsing {pid}: {exc}")
            continue

        if cand["hchain"] not in parsed:
            continue
        vh = cand["vh"]
        vl = cand["vl"]
        h_map = align_domain_to_crystal(vh, parsed[cand["hchain"]])
        if sum(x is not None for x in h_map) < int(0.75 * len(vh)):
            continue

        ab_coords: list[np.ndarray | None] = [
            parsed[cand["hchain"]].residue_coords[idx] if idx is not None else None for idx in h_map
        ]
        ab_ca: list[np.ndarray | None] = [
            parsed[cand["hchain"]].residue_ca[idx] if idx is not None else None for idx in h_map
        ]

        if vl and cand["lchain"] and cand["lchain"] in parsed:
            l_map = align_domain_to_crystal(vl, parsed[cand["lchain"]])
            if sum(x is not None for x in l_map) < int(0.75 * len(vl)):
                continue
            ab_coords.extend(
                parsed[cand["lchain"]].residue_coords[idx] if idx is not None else None for idx in l_map
            )
            ab_ca.extend(
                parsed[cand["lchain"]].residue_ca[idx] if idx is not None else None for idx in l_map
            )
        elif vl:
            continue

        # Concatenate resolved protein antigen chain(s)
        ag_seqs: list[str] = []
        ag_coords: list[np.ndarray] = []
        ag_ca_parts: list[np.ndarray] = []
        used_ag_chains: list[str] = []
        for ach in cand["ag_chains"]:
            if ach in parsed and len(parsed[ach].sequence) >= 15:
                ag_seqs.append(parsed[ach].sequence)
                ag_coords.extend(parsed[ach].residue_coords)
                ag_ca_parts.append(parsed[ach].residue_ca)
                used_ag_chains.append(ach)
        if not ag_coords:
            continue
        ag_seq = "".join(ag_seqs)
        ag_ca = np.concatenate(ag_ca_parts, axis=0)

        pair_dist_full, pair_contact_full = compute_heavy_atom_contact_matrices(
            ab_coords, ab_ca, ag_coords, ag_ca, contact_cutoff=contact_cutoff
        )
        ab_min_dist = np.min(pair_dist_full, axis=1)
        ab_contact_mask = (ab_min_dist <= contact_cutoff).astype(np.int32)
        ag_contact_mask = (np.min(pair_dist_full, axis=0) <= contact_cutoff).astype(np.int32)

        n_ab_c = int(ab_contact_mask.sum())
        n_ag_c = int(ag_contact_mask.sum())
        if n_ab_c < 4 or n_ag_c < 3:
            continue

        # Select an epitope-centered antigen window of length <= max_ag_window so 2D sweeps stay fast
        if len(ag_seq) <= max_ag_window:
            w_start, w_end = 0, len(ag_seq)
        else:
            ep_idx = np.flatnonzero(ag_contact_mask == 1)
            center = int(np.median(ep_idx))
            w_start = max(0, min(center - max_ag_window // 2, len(ag_seq) - max_ag_window))
            w_end = w_start + max_ag_window
        ag_win_seq = ag_seq[w_start:w_end]
        pair_contact_win = pair_contact_full[:, w_start:w_end]
        pair_dist_win = pair_dist_full[:, w_start:w_end]

        # Extract exact SAbDab CDR loop ranges in 0-based concatenated VH+VL coordinates
        cdr_ranges: dict[str, tuple[int, int]] = {}
        if summary_by_inst is not None and cand["instance_id"] in summary_by_inst.index:
            srow = summary_by_inst.loc[cand["instance_id"]]
            for cname in ("CDR-H1", "CDR-H2", "CDR-H3"):
                if cname in srow.index:
                    rng = _find_cdr_range(vh, srow[cname], offset=0)
                    if rng is not None:
                        cdr_ranges[cname] = rng
            if vl:
                for cname in ("CDR-L1", "CDR-L2", "CDR-L3"):
                    if cname in srow.index:
                        rng = _find_cdr_range(vl, srow[cname], offset=len(vh))
                        if rng is not None:
                            cdr_ranges[cname] = rng
        if "CDR-H3" not in cdr_ranges:
            h3_pos = detect_cdrh3_positions_0based(vh)
            if h3_pos:
                cdr_ranges["CDR-H3"] = (h3_pos[0], h3_pos[-1] + 1)

        curated.append(
            CuratedSAbDabComplex(
                instance_id=cand["instance_id"],
                pdb_id=pid,
                hchain=cand["hchain"],
                lchain=cand["lchain"],
                antigen_chains=used_ag_chains,
                antigen_name=cand["antigen_name"],
                resolution=cand["resolution"],
                vh_seq=vh,
                vl_seq=vl,
                ab_paired_seq=format_antibody_sequence(vh, vl),
                ag_seq=ag_seq,
                ag_window_seq=ag_win_seq,
                ag_window_start=w_start,
                cdr_ranges=cdr_ranges,
                ab_contact_mask=ab_contact_mask,
                ab_min_dist=ab_min_dist,
                pair_contact_matrix=pair_contact_win,
                pair_min_dist=pair_dist_win,
                n_ag_contacts=n_ag_c,
            )
        )

    print(
        f"[SAbDab] Curated {len(curated)} crystal antibody–antigen complexes "
        f"(mean <{contact_cutoff:.1f} Å paratope contacts = "
        f"{np.mean([c.ab_contact_mask.sum() for c in curated]):.1f} residues/complex)."
        if curated
        else "[SAbDab] WARNING: No valid antibody–antigen complexes were curated."
    )
    return curated


def _cdr_label_for_pos(pos_0based: int, cdr_ranges: dict[str, tuple[int, int]]) -> str:
    for cname, (lo, hi) in cdr_ranges.items():
        if lo <= pos_0based < hi:
            return cname
    return "FR"


def evaluate_sabdab_dependency_and_gibbs(
    model: LogiCA,
    complexes: list[CuratedSAbDabComplex],
    *,
    metric: str = "sym_kl",
    design_mode: str = "cdrh3",
    num_chains: int = 12,
    num_steps: int = 10,
    temperature: float = 0.4,
    guided_beta: float = 0.6,
    max_design_positions: int = 10,
    skip_2d_dep: bool = False,
    out_dir: Path,
    seed: int = 42,
) -> dict[str, object]:
    """Run `<5 Å` dependency-map contact localization and Holo-vs-Apo Gibbs evaluation on SAbDab."""
    out_dir.mkdir(parents=True, exist_ok=True)
    mapper = DependencyMapper(model)

    complex_rows: list[dict] = []
    residue_rows: list[dict] = []
    dep_metrics_per_complex: list[dict] = []
    gibbs_rows: list[dict] = []
    gibbs_design_rows: list[dict] = []

    saved_dep_plot: str | None = None
    saved_gibbs_plot: str | None = None

    for c_idx, cplx in enumerate(complexes):
        n_vh = len(cplx.vh_seq)
        n_ab = len(cplx.vh_seq) + len(cplx.vl_seq)
        concat_ab = cplx.vh_seq + cplx.vl_seq

        # 1. Single-pass zero-shot KL paratope hotspot profile
        kl_info = mapper.detect_kl_hotspots(
            cplx.ab_paired_seq,
            cplx.ag_window_seq,
            top_k=10,
            max_primary_len=max(260, n_ab + 8),
            max_partner_len=max(160, len(cplx.ag_window_seq) + 4),
        )
        kl_profile = np.asarray(kl_info["kl_profile"], dtype=np.float64)[:n_ab]
        kl_loc = evaluate_contact_localization(cplx.ab_contact_mask[: len(kl_profile)], kl_profile)

        # 2. Full 2D Antigen -> Antibody perturbation dependency map
        dep_profile = kl_profile.copy()
        dep_loc_1d = {"auroc": float("nan"), "aupr": float("nan")}
        dep_loc_2d = {"auroc": float("nan"), "aupr": float("nan")}
        dep_result = None

        if not skip_2d_dep:
            dep_result = mapper.compute_cross_dependency_map(
                cplx.ab_paired_seq,
                cplx.ag_window_seq,
                metric=metric,  # type: ignore[arg-type]
                reduction="mean",
                max_primary_len=max(260, n_ab + 8),
                max_partner_len=max(160, len(cplx.ag_window_seq) + 4),
                batch_size=32,
            )
            l_p = min(n_ab, len(dep_result.primary_profile))
            l_g = min(len(cplx.ag_window_seq), dep_result.matrix.shape[1])
            dep_profile = dep_result.primary_profile[:l_p]
            dep_loc_1d = evaluate_contact_localization(cplx.ab_contact_mask[:l_p], dep_profile)
            dep_loc_2d = evaluate_contact_localization(
                cplx.pair_contact_matrix[:l_p, :l_g].ravel(),
                dep_result.matrix[:l_p, :l_g].ravel(),
            )

        # Compute CDR & true-paratope enrichment among top-10 dependency sites
        top10_idx = np.argsort(-dep_profile)[: min(10, len(dep_profile))]
        top10_cdr_frac = float(np.mean([_cdr_label_for_pos(int(i), cplx.cdr_ranges) != "FR" for i in top10_idx]))
        top10_contact_frac = float(np.mean([cplx.ab_contact_mask[int(i)] == 1 for i in top10_idx]))

        if saved_dep_plot is None and dep_result is not None:
            dep_png = out_dir / f"sabdab_dependency_map_{cplx.pdb_id}.png"
            true_contacts_1based = (np.flatnonzero(cplx.ab_contact_mask == 1) + 1).tolist()
            plot_dependency_map(
                dep_result,
                contact_residues_1based=true_contacts_1based,
                title=f"SAbDab {cplx.pdb_id} ({cplx.hchain}{cplx.lchain} ← Ag {','.join(cplx.antigen_chains)})",
                save_path=dep_png,
            )
            saved_dep_plot = str(dep_png)

        # Record complex & per-residue contact/dependency tables
        complex_rows.append(
            {
                "instance_id": cplx.instance_id,
                "pdb_id": cplx.pdb_id,
                "hchain": cplx.hchain,
                "lchain": cplx.lchain,
                "antigen_chains": "|".join(cplx.antigen_chains),
                "antigen_name": cplx.antigen_name,
                "resolution_A": cplx.resolution,
                "is_vhh": int(len(cplx.vl_seq) == 0),
                "len_vh": len(cplx.vh_seq),
                "len_vl": len(cplx.vl_seq),
                "len_ag_full": len(cplx.ag_seq),
                "len_ag_window": len(cplx.ag_window_seq),
                "ag_window_start_0based": cplx.ag_window_start,
                "n_ab_contacts_lt5A": int(cplx.ab_contact_mask.sum()),
                "n_ag_contacts_lt5A": cplx.n_ag_contacts,
                "n_pair_contacts_lt5A": int(cplx.pair_contact_matrix.sum()),
                "vh_seq": cplx.vh_seq,
                "vl_seq": cplx.vl_seq,
                "cdrh3_seq": (
                    cplx.vh_seq[cplx.cdr_ranges["CDR-H3"][0] : cplx.cdr_ranges["CDR-H3"][1]]
                    if "CDR-H3" in cplx.cdr_ranges
                    else ""
                ),
            }
        )
        for pos0 in range(min(n_ab, len(dep_profile), len(kl_profile))):
            chain_tag = "VH" if pos0 < n_vh else "VL"
            chain_pos_1based = pos0 + 1 if pos0 < n_vh else (pos0 - n_vh + 1)
            residue_rows.append(
                {
                    "pdb_id": cplx.pdb_id,
                    "instance_id": cplx.instance_id,
                    "ab_pos_1based": pos0 + 1,
                    "chain": chain_tag,
                    "chain_pos_1based": chain_pos_1based,
                    "aa": concat_ab[pos0],
                    "region": _cdr_label_for_pos(pos0, cplx.cdr_ranges),
                    "min_heavy_atom_dist_A": (
                        round(float(cplx.ab_min_dist[pos0]), 3) if np.isfinite(cplx.ab_min_dist[pos0]) else np.nan
                    ),
                    "is_contact_lt5A": int(cplx.ab_contact_mask[pos0]),
                    "kl_hotspot_score": round(float(kl_profile[pos0]), 6),
                    "dep_2d_profile_score": round(float(dep_profile[pos0]), 6),
                }
            )

        dep_rec = {
            "pdb_id": cplx.pdb_id,
            "n_ab_contacts": int(cplx.ab_contact_mask.sum()),
            "kl_1d_auroc": round(float(kl_loc["auroc"]), 4) if np.isfinite(kl_loc["auroc"]) else None,
            "kl_1d_aupr": round(float(kl_loc["aupr"]), 4) if np.isfinite(kl_loc["aupr"]) else None,
            "dep_1d_auroc": round(float(dep_loc_1d["auroc"]), 4) if np.isfinite(dep_loc_1d["auroc"]) else None,
            "dep_1d_aupr": round(float(dep_loc_1d["aupr"]), 4) if np.isfinite(dep_loc_1d["aupr"]) else None,
            "dep_2d_pair_auroc": round(float(dep_loc_2d["auroc"]), 4) if np.isfinite(dep_loc_2d["auroc"]) else None,
            "dep_2d_pair_aupr": round(float(dep_loc_2d["aupr"]), 4) if np.isfinite(dep_loc_2d["aupr"]) else None,
            "top10_cdr_fraction": round(top10_cdr_frac, 4),
            "top10_contact_fraction": round(top10_contact_frac, 4),
        }
        dep_metrics_per_complex.append(dep_rec)

        # 3. Coordinate Gibbs Sampling Evaluation: Holo (Ag-Conditioned) vs. Apo (Unconditional AbLang2)
        if design_mode == "kl_hotspots":
            design_pos = sorted(int(i) for i in kl_info["top_indices_0based"][:max_design_positions])  # type: ignore[index]
        elif design_mode == "true_paratope":
            contact_idx = np.flatnonzero(cplx.ab_contact_mask == 1).tolist()
            # Rank contacts by closest heavy-atom distance
            contact_idx.sort(key=lambda idx: float(cplx.ab_min_dist[idx]))
            design_pos = sorted(contact_idx[:max_design_positions])
        else:
            # Default: SAbDab annotated CDR-H3 loop positions
            lo, hi = cplx.cdr_ranges.get("CDR-H3", (max(0, n_vh - 18), max(1, n_vh - 10)))
            h3_all = list(range(lo, hi))
            if len(h3_all) > max_design_positions:
                mid = len(h3_all) // 2
                half = max_design_positions // 2
                h3_all = h3_all[max(0, mid - half) : max(0, mid - half) + max_design_positions]
            design_pos = h3_all

        holo_gibbs = model.gibbs_sample(
            seed_sequence=cplx.ab_paired_seq,
            context=cplx.ag_window_seq,
            design_positions_0based=design_pos,
            num_chains=num_chains,
            num_steps=num_steps,
            temperature=temperature,
            guided_beta=guided_beta,
            seed=seed + c_idx,
            max_primary_len=max(260, n_ab + 8),
            max_partner_len=max(160, len(cplx.ag_window_seq) + 4),
        )
        apo_gibbs = model.gibbs_sample(
            seed_sequence=cplx.ab_paired_seq,
            context=None,
            design_positions_0based=design_pos,
            num_chains=num_chains,
            num_steps=num_steps,
            temperature=temperature,
            guided_beta=0.0,
            seed=seed + 1000 + c_idx,
            max_primary_len=max(260, n_ab + 8),
            max_partner_len=max(160, len(cplx.ag_window_seq) + 4),
        )

        if saved_gibbs_plot is None:
            gibbs_png = out_dir / f"sabdab_gibbs_{cplx.pdb_id}.png"
            plot_gibbs_design(
                holo_gibbs,
                baseline_result=apo_gibbs,
                title=f"AbLang2+ESM-2 Coordinate Gibbs ({cplx.pdb_id}, mode={design_mode})",
                save_path=gibbs_png,
            )
            saved_gibbs_plot = str(gibbs_png)

        def _recovery_stats(seqs: list[str]) -> tuple[float, float]:
            rec_all: list[float] = []
            rec_contact: list[float] = []
            contact_design_pos = [p for p in design_pos if cplx.ab_contact_mask[p] == 1]
            for s in seqs:
                s_flat = s.replace("|", "")
                rec_all.append(float(np.mean([s_flat[p] == concat_ab[p] for p in design_pos])))
                if contact_design_pos:
                    rec_contact.append(float(np.mean([s_flat[p] == concat_ab[p] for p in contact_design_pos])))
            return (
                float(np.mean(rec_all)),
                float(np.mean(rec_contact)) if rec_contact else float("nan"),
            )

        holo_rec, holo_rec_c = _recovery_stats(holo_gibbs.final_sequences)
        apo_rec, apo_rec_c = _recovery_stats(apo_gibbs.final_sequences)

        gibbs_rows.append(
            {
                "pdb_id": cplx.pdb_id,
                "design_mode": design_mode,
                "n_designed_sites": len(design_pos),
                "n_designed_contact_sites": int(sum(cplx.ab_contact_mask[p] == 1 for p in design_pos)),
                "holo_mean_score": round(float(np.mean(holo_gibbs.final_scores)), 4),
                "holo_top1_score": round(float(np.max(holo_gibbs.final_scores)), 4),
                "apo_mean_score": round(float(np.mean(apo_gibbs.final_scores)), 4),
                "holo_native_recovery": round(holo_rec, 4),
                "apo_native_recovery": round(apo_rec, 4),
                "holo_contact_recovery": round(holo_rec_c, 4) if np.isfinite(holo_rec_c) else None,
                "apo_contact_recovery": round(apo_rec_c, 4) if np.isfinite(apo_rec_c) else None,
                "holo_n_eff": round(float(holo_gibbs.effective_sample_size), 2),
                "apo_n_eff": round(float(apo_gibbs.effective_sample_size), 2),
                "holo_unique_designs": len(set(holo_gibbs.final_sequences)),
            }
        )
        for rank_i, (seq, sc) in enumerate(holo_gibbs.top_designs(k=min(5, num_chains)), 1):
            s_flat = seq.replace("|", "")
            gibbs_design_rows.append(
                {
                    "pdb_id": cplx.pdb_id,
                    "rank": rank_i,
                    "score": round(float(sc), 4),
                    "wt_designed_motif": "".join(concat_ab[p] for p in design_pos),
                    "sampled_motif": "".join(s_flat[p] for p in design_pos),
                    "n_mutations": int(sum(s_flat[p] != concat_ab[p] for p in design_pos)),
                    "designed_ab_seq": seq,
                }
            )

        print(
            f"  [{c_idx + 1:2d}/{len(complexes)}] {cplx.pdb_id} | "
            f"contacts={int(cplx.ab_contact_mask.sum()):2d} | "
            f"KL-AUROC={dep_rec['kl_1d_auroc']} | Dep2D-1D-AUROC={dep_rec['dep_1d_auroc']} | "
            f"Gibbs holo_rec={holo_rec:.2f} apo_rec={apo_rec:.2f} N_eff={holo_gibbs.effective_sample_size:.1f}"
        )

    # Save curated SAbDab tables & evaluation summaries
    pd.DataFrame(complex_rows).to_csv(out_dir / "sabdab_complexes.csv", index=False)
    pd.DataFrame(residue_rows).to_csv(out_dir / "sabdab_dependency_profiles.csv", index=False)
    pd.DataFrame(gibbs_design_rows).to_csv(out_dir / "sabdab_gibbs_designs.csv", index=False)

    def _mean_finite(key: str, rows: list[dict]) -> float | None:
        vals = [float(r[key]) for r in rows if r.get(key) is not None and np.isfinite(r[key])]
        return round(float(np.mean(vals)), 4) if vals else None

    dep_summary = {
        "n_complexes": len(complexes),
        "metric": metric,
        "kl_1d_auroc_mean": _mean_finite("kl_1d_auroc", dep_metrics_per_complex),
        "kl_1d_aupr_mean": _mean_finite("kl_1d_aupr", dep_metrics_per_complex),
        "dep_1d_auroc_mean": _mean_finite("dep_1d_auroc", dep_metrics_per_complex),
        "dep_1d_aupr_mean": _mean_finite("dep_1d_aupr", dep_metrics_per_complex),
        "dep_2d_pair_auroc_mean": _mean_finite("dep_2d_pair_auroc", dep_metrics_per_complex),
        "dep_2d_pair_aupr_mean": _mean_finite("dep_2d_pair_aupr", dep_metrics_per_complex),
        "top10_cdr_fraction_mean": _mean_finite("top10_cdr_fraction", dep_metrics_per_complex),
        "top10_contact_fraction_mean": _mean_finite("top10_contact_fraction", dep_metrics_per_complex),
        "example_dependency_plot": saved_dep_plot,
        "per_complex": dep_metrics_per_complex,
    }
    gibbs_summary = {
        "n_complexes": len(complexes),
        "design_mode": design_mode,
        "num_chains": num_chains,
        "num_steps": num_steps,
        "temperature": temperature,
        "guided_beta": guided_beta,
        "holo_mean_score": _mean_finite("holo_mean_score", gibbs_rows),
        "apo_mean_score": _mean_finite("apo_mean_score", gibbs_rows),
        "holo_native_recovery_mean": _mean_finite("holo_native_recovery", gibbs_rows),
        "apo_native_recovery_mean": _mean_finite("apo_native_recovery", gibbs_rows),
        "holo_contact_recovery_mean": _mean_finite("holo_contact_recovery", gibbs_rows),
        "apo_contact_recovery_mean": _mean_finite("apo_contact_recovery", gibbs_rows),
        "holo_n_eff_mean": _mean_finite("holo_n_eff", gibbs_rows),
        "apo_n_eff_mean": _mean_finite("apo_n_eff", gibbs_rows),
        "example_gibbs_plot": saved_gibbs_plot,
        "per_complex": gibbs_rows,
    }
    (out_dir / "sabdab_dependency_metrics.json").write_text(json.dumps(dep_summary, indent=2) + "\n", encoding="utf-8")
    (out_dir / "sabdab_gibbs_eval.json").write_text(json.dumps(gibbs_summary, indent=2) + "\n", encoding="utf-8")
    return {"dependency": dep_summary, "gibbs": gibbs_summary}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", type=str, default=None, help="Trained AbLang2+ESM-2 LogiCA `.pt` path or HF repo (`Yale-CompBio/logica-abag`)")
    parser.add_argument("--hf-repo", type=str, default="Yale-CompBio/logica-abag", help="Hugging Face Hub repo ID (default: `Yale-CompBio/logica-abag`; pass `none` for untrained)")
    parser.add_argument("--scale", type=str, default="esm35m", help="Checkpoint scale when loading from Hugging Face Hub")
    parser.add_argument("--ab-model", type=str, default="ablang2-paired", help="Primary antibody BioFM (`ablang2-paired` or HF id)")
    parser.add_argument("--ag-model", type=str, default="facebook/esm2_t12_35M_UR50D", help="Partner antigen BioFM")
    parser.add_argument("--sabdab-dir", type=Path, default=SUBFOLDER_ROOT / "data" / "sabdab", help="Cache directory for SAbDab2 summary & mmCIF files")
    parser.add_argument("--sabdab-summary", type=Path, default=None, help="Optional local pre-downloaded SAbDab2 summary CSV/TSV")
    parser.add_argument("--force-download", action="store_true", help="Re-download SAbDab2 summary even if cached")
    parser.add_argument("--pdb-ids", type=str, default=None, help="Optional comma-separated PDB IDs (e.g. `1N8Z,1FDL,2FJG`)")
    parser.add_argument("--max-complexes", type=int, default=20, help="Number of non-redundant SAbDab complexes to evaluate")
    parser.add_argument("--max-resolution", type=float, default=3.0, help="Maximum crystal resolution in Å")
    parser.add_argument("--contact-cutoff", type=float, default=5.0, help="Heavy-atom contact distance cutoff in Å")
    parser.add_argument("--max-ag-window", type=int, default=160, help="Epitope-centered antigen window length for 2D dependency maps")
    parser.add_argument("--paired-only", action="store_true", help="Restrict to paired VH|VL antibodies")
    parser.add_argument("--vhh-only", action="store_true", help="Restrict to single-domain VHH nanobodies")
    parser.add_argument("--metric", type=str, choices=["sym_kl", "l2", "jsd"], default="sym_kl")
    parser.add_argument("--design-mode", type=str, choices=["cdrh3", "kl_hotspots", "true_paratope"], default="cdrh3")
    parser.add_argument("--num-chains", type=int, default=12)
    parser.add_argument("--num-steps", type=int, default=10)
    parser.add_argument("--temperature", type=float, default=0.4)
    parser.add_argument("--guided-beta", type=float, default=0.6)
    parser.add_argument("--max-design-positions", type=int, default=10)
    parser.add_argument("--skip-2d-dep", action="store_true", help="Only compute single-pass KL hotspots (skip 2D perturbation sweep)")
    parser.add_argument("--num-workers", type=int, default=6, help="Parallel RCSB mmCIF download threads")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--device", type=str, default="cuda:0" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--out-dir", type=Path, default=SUBFOLDER_ROOT / "outputs" / "sabdab_dependency_and_gibbs")
    parser.add_argument("--smoke-test", action="store_true", help="Fast live CPU end-to-end test on 1N8Z & 1FDL")
    args = parser.parse_args()

    t0 = time.time()
    pdb_list = [p.strip().upper() for p in args.pdb_ids.split(",")] if args.pdb_ids else None

    if args.smoke_test:
        args.ag_model = "facebook/esm2_t6_8M_UR50D"
        pdb_list = pdb_list or ["1N8Z", "1FDL"]
        args.max_complexes = len(pdb_list)
        args.max_ag_window = 36
        args.num_chains = 4
        args.num_steps = 4
        args.max_design_positions = 6

    summary_df = download_or_load_sabdab_summary(
        args.sabdab_dir,
        summary_path=args.sabdab_summary,
        force_download=args.force_download,
    )
    complexes = curate_sabdab_complexes(
        summary_df,
        structures_dir=args.sabdab_dir / "structures",
        max_complexes=args.max_complexes,
        max_resolution=args.max_resolution,
        contact_cutoff=args.contact_cutoff,
        max_ag_window=args.max_ag_window,
        pdb_ids=pdb_list,
        paired_only=args.paired_only,
        vhh_only=args.vhh_only,
        num_workers=args.num_workers,
    )
    if not complexes:
        raise SystemExit("No valid SAbDab complexes could be curated; check filters or network connectivity.")

    device = torch.device(args.device)
    if args.smoke_test and args.checkpoint is None and args.hf_repo == "Yale-CompBio/logica-abag":
        args.hf_repo = None
    if args.checkpoint is not None:
        model = LogiCA.from_checkpoint(args.checkpoint, scale=args.scale, device=device)
    elif args.hf_repo is not None and str(args.hf_repo).lower() != "none":
        model = LogiCA.from_checkpoint(args.hf_repo, scale=args.scale, device=device)
    else:
        model = LogiCA.from_pretrained(
            args.ab_model,
            args.ag_model,
            width=128 if args.smoke_test else 256,
            layers=1 if args.smoke_test else 2,
            gate_mode="tanh",
            gate_init=0.25 if args.smoke_test else 0.1,
        ).to(device)
    model.eval()

    results = evaluate_sabdab_dependency_and_gibbs(
        model,
        complexes,
        metric=args.metric,
        design_mode=args.design_mode,
        num_chains=args.num_chains,
        num_steps=args.num_steps,
        temperature=args.temperature,
        guided_beta=args.guided_beta,
        max_design_positions=args.max_design_positions,
        skip_2d_dep=args.skip_2d_dep,
        out_dir=args.out_dir,
        seed=args.seed,
    )

    elapsed = round(time.time() - t0, 2)
    print(f"\n[SAbDab Pipeline Complete in {elapsed}s] Outputs saved to {args.out_dir}")
    print(
        f"  Dependency Localization: KL-1D AUROC={results['dependency']['kl_1d_auroc_mean']} | "
        f"Dep2D-1D AUROC={results['dependency']['dep_1d_auroc_mean']} | "
        f"Top-10 CDR Frac={results['dependency']['top10_cdr_fraction_mean']}"
    )
    print(
        f"  Gibbs Evaluation ({args.design_mode}): "
        f"Holo Recovery={results['gibbs']['holo_native_recovery_mean']} vs. "
        f"Apo Recovery={results['gibbs']['apo_native_recovery_mean']} | "
        f"Holo N_eff={results['gibbs']['holo_n_eff_mean']}"
    )


if __name__ == "__main__":
    main()
