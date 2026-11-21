"""Shared Protein–Ligand dataset resolver & loader for `AI-SysBio/LogiCA`.

Automatically loads the curated Protein–Ligand DMS, Stage-0 BindingDB, and BioLiP-175
benchmark dataset from Kaggle (`aurlienplissier/protein-ligand-dms` via `kagglehub` or
the `kaggle` CLI) OR from a local checkout (`--data-dir` / `LOGICA_PL_DATA_DIR`).

Supports both the clean Kaggle layout (`cohorts.csv`, `dms_tasks/`, `baseline_scores/`,
`bindingdb_pretrain/`, `biolip_175/`) and the legacy nested `processed/` layout.
"""

from __future__ import annotations

import os
from pathlib import Path
import shutil
import subprocess

import pandas as pd

DEFAULT_KAGGLE_DATASET = "aurlienplissier/protein-ligand-dms"
SUBFOLDER_ROOT = Path(__file__).resolve().parents[1]


def _is_valid_pl_root(base: Path) -> bool:
    return (base / "cohorts.csv").is_file() or (base / "processed" / "cohorts.parquet").is_file()


def resolve_pl_data_dir(
    data_dir: Path | str | None = None,
    kaggle_dataset: str | None = DEFAULT_KAGGLE_DATASET,
) -> Path:
    """Resolves the Protein–Ligand benchmark directory (local override -> cache -> Kaggle)."""
    candidates: list[Path] = []
    if data_dir is not None:
        candidates.append(Path(data_dir).expanduser().resolve())
    env_dir = os.environ.get("LOGICA_PL_DATA_DIR")
    if env_dir:
        candidates.append(Path(env_dir).expanduser().resolve())
    candidates.append((SUBFOLDER_ROOT / "data").resolve())

    for cand in candidates:
        if _is_valid_pl_root(cand):
            return cand

    slug = (kaggle_dataset or DEFAULT_KAGGLE_DATASET).strip()
    try:
        import kagglehub  # type: ignore[import-not-found]

        dl = Path(kagglehub.dataset_download(slug)).resolve()
        if _is_valid_pl_root(dl):
            return dl
    except Exception as exc:
        print(f"[pl_data_loader] kagglehub download notice: {exc}")

    kaggle_bin = shutil.which("kaggle")
    if kaggle_bin:
        cache_dir = Path.home() / ".cache" / "logica" / slug.replace("/", "__")
        cache_dir.mkdir(parents=True, exist_ok=True)
        if not _is_valid_pl_root(cache_dir):
            subprocess.run(
                [kaggle_bin, "datasets", "download", "-d", slug, "-p", str(cache_dir), "--unzip"],
                check=True,
            )
        if _is_valid_pl_root(cache_dir):
            return cache_dir

    raise FileNotFoundError(
        f"Could not locate Protein–Ligand benchmark data locally or download `{slug}` from Kaggle.\n"
        f"Install `kagglehub` (`pip install kagglehub`) or download manually from:\n"
        f"  https://www.kaggle.com/datasets/{slug}"
    )


def load_cohorts(data_dir: Path) -> pd.DataFrame:
    if (data_dir / "cohorts.csv").is_file():
        return pd.read_csv(data_dir / "cohorts.csv")
    return pd.read_parquet(data_dir / "processed" / "cohorts.parquet")


def list_task_names(data_dir: Path) -> list[str]:
    if (data_dir / "dms_tasks").is_dir():
        return sorted(p.stem for p in (data_dir / "dms_tasks").glob("*.parquet"))
    return sorted(p.stem for p in (data_dir / "processed" / "tasks").glob("*.parquet"))


def load_task_df(
    data_dir: Path,
    task_name: str,
    include_baselines: bool = False,
) -> pd.DataFrame:
    """Loads one task table by name (with or without `_zeroshot` suffix)."""
    clean_stem = task_name.replace("_zeroshot", "")
    kaggle_pq = data_dir / "dms_tasks" / f"{clean_stem}.parquet"
    if kaggle_pq.is_file():
        df = pd.read_parquet(kaggle_pq)
        if include_baselines:
            base_pq = data_dir / "baseline_scores" / f"{clean_stem}_baselines.parquet"
            if base_pq.is_file():
                bdf = pd.read_parquet(base_pq)
                extra_cols = [c for c in bdf.columns if c not in df.columns]
                if len(bdf) == len(df) and extra_cols:
                    df = pd.concat([df.reset_index(drop=True), bdf[extra_cols].reset_index(drop=True)], axis=1)
        return df

    legacy_pq = data_dir / "processed" / "tasks" / f"{task_name}.parquet"
    if legacy_pq.is_file():
        return pd.read_parquet(legacy_pq)
    alt_pq = data_dir / "processed" / "tasks" / f"{clean_stem}_zeroshot.parquet"
    if alt_pq.is_file():
        return pd.read_parquet(alt_pq)
    raise FileNotFoundError(f"Task table `{task_name}` not found under {data_dir}")


def resolve_biolip_dir(data_dir: Path) -> Path:
    bdir = data_dir / "biolip_175"
    if bdir.is_dir():
        return bdir
    raise FileNotFoundError(f"BioLiP-175 directory not found under {data_dir}")


def load_bindingdb_positive_pairs(
    data_dir: Path,
    bindingdb_dir: Path | None = None,
    smoke_test: bool = False,
) -> tuple[list[str], list[str]]:
    """Loads Stage-1 positive `(protein_sequence, ligand_selfies)` pairs from BindingDB."""
    # 1. Explicit local bindingdb_dir (legacy flat bindingdb_train.parquet or Kaggle bindingdb_pretrain/)
    search_dirs: list[Path] = []
    if bindingdb_dir is not None:
        search_dirs.append(Path(bindingdb_dir))
    search_dirs.append(data_dir / "bindingdb_pretrain")

    for sdir in search_dirs:
        if (sdir / "train_pairs.parquet").is_file() and (sdir / "proteins.parquet").is_file():
            pairs = pd.read_parquet(sdir / "train_pairs.parquet")
            pos = pairs[pairs["label"] == 1]
            if smoke_test:
                pos = pos.head(16)
            prots = pd.read_parquet(sdir / "proteins.parquet")[["protein_id", "sequence"]]
            drugs = pd.read_parquet(sdir / "drugs.parquet")[["drug_id", "selfies"]]
            merged = pos.merge(prots, on="protein_id", how="inner").merge(drugs, on="drug_id", how="inner")
            return merged["sequence"].astype(str).tolist(), merged["selfies"].astype(str).tolist()
        if (sdir / "bindingdb_train.parquet").is_file():
            df = pd.read_parquet(sdir / "bindingdb_train.parquet")
            pos = df[df["label"] == 1].reset_index(drop=True)
            if smoke_test:
                pos = pos.head(16)
            return pos["protein_sequence"].astype(str).tolist(), pos["ligand_selfies"].astype(str).tolist()

    # Fallback: BioLiP model_inputs for quick smoke testing
    cdf = pd.read_parquet(resolve_biolip_dir(data_dir) / "model_inputs.parquet").dropna(
        subset=["protein_aaseq", "selfies"]
    )
    if smoke_test:
        cdf = cdf.head(12)
    return cdf["protein_aaseq"].astype(str).tolist(), cdf["selfies"].astype(str).tolist()
