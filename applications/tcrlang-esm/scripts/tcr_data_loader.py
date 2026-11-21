"""Shared TCR–Peptide dataset resolver & loader for `AI-SysBio/LogiCA`.

Automatically loads the curated zero-leakage TCR–Peptide pretraining, fine-tuning,
peptide-anchored DMS (71 groups), TCR-anchored CDR3 DMS (23 groups), binary
recognition, and TCR3d `<5 A` crystal contact dataset from Kaggle
(`aurlienplissier/tcr-peptide-dms` via `kagglehub` or `kaggle` CLI) OR from a
local checkout (`--data-dir` / `LOGICA_TCR_DATA_DIR`).

Supports both the clean deduplicated Kaggle layout (`pretrain/`, `finetune/`,
`peptide_dms/`, `tcr_dms/`, `binary_benchmarks/`, `tcr3d_contacts/`) and the
legacy `epact_train_all_downstream_non_overlap/` layout.
"""

from __future__ import annotations

import os
from pathlib import Path
import shutil
import subprocess

DEFAULT_KAGGLE_DATASET = "aurlienplissier/tcr-peptide-dms"
SUBFOLDER_ROOT = Path(__file__).resolve().parents[1]


def _is_valid_tcr_root(base: Path) -> bool:
    return (base / "pretrain" / "epact_stapler_iedb_train_non_overlap.csv").is_file()


def resolve_tcr_data_dir(
    data_dir: Path | str | None = None,
    kaggle_dataset: str | None = DEFAULT_KAGGLE_DATASET,
) -> Path:
    """Resolves the TCR–Peptide benchmark directory (local override -> cache -> Kaggle)."""
    candidates: list[Path] = []
    if data_dir is not None:
        candidates.append(Path(data_dir).expanduser().resolve())
    env_dir = os.environ.get("LOGICA_TCR_DATA_DIR")
    if env_dir:
        candidates.append(Path(env_dir).expanduser().resolve())
    candidates.append((SUBFOLDER_ROOT / "data").resolve())

    for cand in candidates:
        if _is_valid_tcr_root(cand):
            return cand

    slug = (kaggle_dataset or DEFAULT_KAGGLE_DATASET).strip()
    try:
        import kagglehub  # type: ignore[import-not-found]

        dl = Path(kagglehub.dataset_download(slug)).resolve()
        if _is_valid_tcr_root(dl):
            return dl
    except Exception as exc:
        print(f"[tcr_data_loader] kagglehub download notice: {exc}")

    kaggle_bin = shutil.which("kaggle")
    if kaggle_bin:
        cache_dir = Path.home() / ".cache" / "logica" / slug.replace("/", "__")
        cache_dir.mkdir(parents=True, exist_ok=True)
        if not _is_valid_tcr_root(cache_dir):
            subprocess.run(
                [kaggle_bin, "datasets", "download", "-d", slug, "-p", str(cache_dir), "--unzip"],
                check=True,
            )
        if _is_valid_tcr_root(cache_dir):
            return cache_dir

    raise FileNotFoundError(
        f"Could not locate TCR–Peptide benchmark data locally or download `{slug}` from Kaggle.\n"
        f"Install `kagglehub` (`pip install kagglehub`) or download manually from:\n"
        f"  https://www.kaggle.com/datasets/{slug}"
    )


def resolve_tcr_paths(data_dir: Path) -> dict[str, Path]:
    """Returns canonical paths across both Kaggle and legacy directory layouts."""
    if (data_dir / "finetune" / "epact_train.csv").is_file():
        return {
            "pretrain_train": data_dir / "pretrain" / "epact_stapler_iedb_train_non_overlap.csv",
            "pretrain_val": data_dir / "pretrain" / "epact_stapler_iedb_val_non_overlap.csv",
            "finetune_train": data_dir / "finetune" / "epact_train.csv",
            "finetune_val": data_dir / "finetune" / "epact_val.csv",
            "epytope_mutation": data_dir / "peptide_dms" / "epytope_mutation_test.csv",
            "batcave": data_dir / "peptide_dms" / "batcave_test.csv",
            "batcave_full": data_dir / "peptide_dms" / "batcave_test_full.csv",
            "atlas_pep": data_dir / "peptide_dms" / "atlas_pep_test.csv",
            "tcr_mutations_combined": data_dir / "tcr_dms" / "tcr_mutations_combined_test.csv",
            "tcr_affinity_expanded": data_dir / "tcr_dms" / "tcr_affinity_expanded_test.csv",
            "tcr_dms_868z11_cdr3": data_dir / "tcr_dms" / "tcr_dms_868z11_cdr3_test.csv",
            "atlas_cdr": data_dir / "tcr_dms" / "atlas_cdr_test.csv",
            "epytope_binary": data_dir / "binary_benchmarks" / "epytope_binary_test.csv",
            "immrep2025": data_dir / "binary_benchmarks" / "immrep2025_test.csv",
            "tcr3d_complexes": data_dir / "tcr3d_contacts" / "tcr3d_complexes.csv",
            "tcr3d_contacts": data_dir / "tcr3d_contacts" / "tcr3d_residue_contacts.parquet",
        }

    # Legacy repository layout fallback
    ft = data_dir / "finetune" / "epact_train_all_downstream_non_overlap"
    return {
        "pretrain_train": data_dir / "pretrain" / "epact_stapler_iedb_train_non_overlap.csv",
        "pretrain_val": data_dir / "pretrain" / "epact_stapler_iedb_val_non_overlap.csv",
        "finetune_train": ft / "epact_train_epytope_binary_test" / "train.csv",
        "finetune_val": ft / "epact_train_epytope_binary_test" / "val.csv",
        "epytope_mutation": ft / "epact_train_epytope_mutation_test" / "test.csv",
        "batcave": ft / "epact_train_batcave_test" / "test.csv",
        "batcave_full": ft / "epact_train_batcave_test_full" / "test.csv",
        "atlas_pep": ft / "epact_train_atlaspep_test" / "test.csv",
        "tcr_mutations_combined": data_dir / "expanded_tcr_mut" / "tcr_mutations_combined_test.csv",
        "tcr_affinity_expanded": data_dir / "expanded_tcr_mut" / "tcr_affinity_expanded_test.csv",
        "tcr_dms_868z11_cdr3": data_dir / "expanded_tcr_mut" / "tcr_dms_868z11_cdr3_test.csv",
        "atlas_cdr": ft / "epact_train_atlas_cdr_test" / "test.csv",
        "epytope_binary": ft / "epact_train_epytope_binary_test" / "test.csv",
        "immrep2025": ft / "epact_train_immrep2025_test" / "test.csv",
    }
