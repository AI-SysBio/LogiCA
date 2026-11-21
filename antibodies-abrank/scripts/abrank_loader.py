"""Shared AbRank data loader & evaluator bridge for `AI-SysBio/logica/antibodies-abrank`.

Connects `AI-SysBio/logica` seamlessly to the public `AI-SysBio/AbRank` benchmark dataset:
  1. Loads AbRank from Kaggle (`aisysbio/abrank`, via `kagglehub` or `kaggle` CLI),
     OR from a local checkout (`--abrank-dir` / `ABRANK_DATA_DIR`).
  2. Formats paired IgG/Fab antibodies as `"VH|VL"` and single-domain VHH nanobodies as `"VHH|"`
     for `ablang2-paired`, paired with antigen ectodomain sequences for `ESM-2`.
  3. Evaluates generated submission TSVs directly with `abrank.evaluate.evaluate_flagship_suite`
     when `abrank` is installed (`pip install git+https://github.com/AI-SysBio/AbRank.git`),
     with a numerically identical self-contained 3-state evaluator fallback otherwise.
"""

from __future__ import annotations

import os
from pathlib import Path
import re
import shutil
import subprocess
from typing import Any

import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score

DEFAULT_KAGGLE_DATASET = "aisysbio/abrank"
ABRANK_REPO_URL = "https://github.com/AI-SysBio/AbRank"
SENTINEL_FILES = ("assays_kd.tsv", "splits/unseen_ag/test.tsv")


def _find_benchmark_root(base: Path) -> Path | None:
    base = Path(base).expanduser().resolve()
    for cand in (
        base,
        base / "benchmark",
        base / "data" / "benchmark",
        base / "abrank" / "benchmark",
        base / "abrank" / "data" / "benchmark",
    ):
        if all((cand / rel).exists() for rel in SENTINEL_FILES):
            return cand
    return None


def resolve_abrank_dir(
    abrank_dir: Path | str | None = None,
    *,
    kaggle_dataset: str | None = None,
    auto_download: bool = True,
) -> Path:
    """Resolve the AbRank benchmark directory from local disk, `ABRANK_DATA_DIR`, or Kaggle."""
    # Prefer the official abrank.kaggle_io resolver if `abrank` is installed
    try:
        from abrank.kaggle_io import resolve_benchmark_dir

        return resolve_benchmark_dir(
            data_dir=abrank_dir,
            kaggle_dataset=kaggle_dataset,
            auto_download=auto_download,
        )
    except ImportError:
        pass
    except FileNotFoundError:
        if not auto_download or abrank_dir is not None:
            raise

    if abrank_dir is not None:
        found = _find_benchmark_root(Path(abrank_dir))
        if found is None:
            raise FileNotFoundError(
                f"Could not find AbRank files {SENTINEL_FILES} under `{abrank_dir}`.\n"
                f"See {ABRANK_REPO_URL} for dataset layout and download instructions."
            )
        return found

    env_dir = os.environ.get("ABRANK_DATA_DIR")
    if env_dir:
        found = _find_benchmark_root(Path(env_dir))
        if found is not None:
            return found

    if not auto_download:
        raise FileNotFoundError(
            "AbRank data not found locally. Pass `--abrank-dir /path/to/AbRank/data/benchmark`, "
            f"set `ABRANK_DATA_DIR`, or allow Kaggle download (`--kaggle-dataset {DEFAULT_KAGGLE_DATASET}`)."
        )

    slug = kaggle_dataset or os.environ.get("ABRANK_KAGGLE_DATASET", DEFAULT_KAGGLE_DATASET)

    # 1. Preferred: kagglehub
    try:
        import kagglehub  # type: ignore[import-not-found]

        dl_path = Path(kagglehub.dataset_download(slug))
        found = _find_benchmark_root(dl_path)
        if found is not None:
            return found
        raise FileNotFoundError(
            f"Downloaded Kaggle dataset `{slug}` to `{dl_path}`, but {SENTINEL_FILES} were not found."
        )
    except ImportError:
        pass

    # 2. Fallback: official `kaggle` CLI
    kaggle_bin = shutil.which("kaggle")
    if kaggle_bin:
        cache_dir = Path.home() / ".cache" / "abrank" / "kaggle" / slug.replace("/", "__")
        cache_dir.mkdir(parents=True, exist_ok=True)
        found = _find_benchmark_root(cache_dir)
        if found is not None:
            return found
        subprocess.run(
            [kaggle_bin, "datasets", "download", "-d", slug, "-p", str(cache_dir), "--unzip"],
            check=True,
        )
        found = _find_benchmark_root(cache_dir)
        if found is not None:
            return found

    raise RuntimeError(
        "Unable to locate AbRank data locally or download from Kaggle.\n"
        "Choose one of:\n"
        f"  1. Install KaggleHub (`pip install kagglehub`) to auto-fetch `{slug}`\n"
        f"  2. Clone {ABRANK_REPO_URL}, run `scripts/00_download_from_kaggle.py`, and pass `--abrank-dir`\n"
        "  3. Export `ABRANK_DATA_DIR=/path/to/AbRank/data/benchmark`"
    )


def format_antibody_sequence(vh: str | float | None, vl: str | float | None = None) -> str:
    """Format paired `(VH, VL)` or single-domain `VHH` into AbLang2-Paired pipe-delimited string."""
    h = "" if vh is None or (isinstance(vh, float) and np.isnan(vh)) else str(vh).strip().upper()
    l = "" if vl is None or (isinstance(vl, float) and np.isnan(vl)) else str(vl).strip().upper()
    if l in ("", "NAN", "NONE", "NA"):
        return f"{h}|"
    return f"{h}|{l}"


def detect_cdrh3_positions_0based(vh_seq: str) -> list[int]:
    """Heuristically locate CDR-H3 residue indices (0-based within `VH`) via conserved Cys..WGxG flanking motifs."""
    seq = vh_seq.strip().upper()
    # Standard IMGT/Chothia CDR-H3 anchor: conserved C[A/T/V/S/G][R/K/S/T] ... WG[A-Z]G near C-terminus of VH
    m = re.search(r"C[A-Z]{2}([A-Z]{4,24})WG[A-Z]G", seq[65:])
    if m is not None:
        start = 65 + m.start(1)
        end = 65 + m.end(1)
        return list(range(start, end))
    # Fallback to canonical ~pos 95..106 window if motif is non-canonical
    lo = min(max(len(seq) - 22, 0), max(len(seq) - 8, 0))
    hi = min(lo + 10, len(seq))
    return list(range(lo, hi))


def load_complex_lookup(bdir: Path, include_vhh: bool = True, include_transfer: bool = False) -> dict[str, dict[str, Any]]:
    """Build an `AbAgID -> {ab_seq, vh, vl, ag_seq, binding_score, ag_group, ag_cluster, dataset}` map."""
    tables = [bdir / "assays_kd.tsv"]
    if include_vhh and (bdir / "assays_vhh.tsv").exists():
        tables.append(bdir / "assays_vhh.tsv")
    if include_transfer:
        for fname in ("assays_transfer.tsv", "assays_cross_modal.tsv"):
            if (bdir / fname).exists():
                tables.append(bdir / fname)

    lookup: dict[str, dict[str, Any]] = {}
    for tp in tables:
        cols = [
            "AbAgID",
            "Ab_heavy_chain_seq",
            "Ab_light_chain_seq",
            "Ag_seq",
            "binding_score",
            "Ag_group",
            "Ag_cluster",
            "Dataset",
        ]
        avail_cols = pd.read_csv(tp, sep="\t", nrows=0).columns.tolist()
        usecols = [c for c in cols if c in avail_cols]
        df = pd.read_csv(tp, sep="\t", usecols=usecols, low_memory=False).drop_duplicates("AbAgID")
        for row in df.itertuples(index=False):
            abag_id = str(getattr(row, "AbAgID"))
            if abag_id in lookup:
                continue
            vh = getattr(row, "Ab_heavy_chain_seq", "")
            vl = getattr(row, "Ab_light_chain_seq", "")
            ag = str(getattr(row, "Ag_seq", "")).strip().upper()
            ab = format_antibody_sequence(vh, vl)
            lookup[abag_id] = {
                "ab_seq": ab,
                "vh": str(vh).strip().upper(),
                "vl": "" if pd.isna(vl) else str(vl).strip().upper(),
                "ag_seq": ag,
                "binding_score": float(getattr(row, "binding_score", np.nan)),
                "ag_group": str(getattr(row, "Ag_group", "")),
                "ag_cluster": str(getattr(row, "Ag_cluster", "")),
                "dataset": str(getattr(row, "Dataset", "")),
            }
    return lookup


def compute_3state_metrics_fallback(
    labels: np.ndarray,
    binding_diff: np.ndarray,
    pred_diff: np.ndarray,
    pred_impact: np.ndarray | None = None,
) -> dict[str, float | int | None]:
    """Self-contained orientation-invariant 3-state evaluator matching `abrank.evaluate.compute_3state_metrics`."""
    lbl = np.asarray(labels, dtype=int)
    bdiff = np.asarray(binding_diff, dtype=np.float64)
    pdiff = np.asarray(pred_diff, dtype=np.float64)
    pimp = np.abs(pdiff) if pred_impact is None else np.asarray(pred_impact, dtype=np.float64)

    imp_mask = lbl != 0
    neu_mask = lbl == 0
    n_imp = int(imp_mask.sum())
    n_neu = int(neu_mask.sum())

    if n_imp > 0:
        y_imp = (bdiff[imp_mask] > 0).astype(int)
        p_imp = pdiff[imp_mask]
        auc_dir = float(roc_auc_score(np.concatenate([y_imp, 1 - y_imp]), np.concatenate([p_imp, -p_imp])))
    else:
        auc_dir = float("nan")

    if n_imp > 0 and n_neu > 0:
        auc_impact = float(roc_auc_score(imp_mask.astype(int), pimp))
        s_plus = np.where(bdiff[imp_mask] > 0, pdiff[imp_mask], -pdiff[imp_mask])
        s_zero = np.concatenate([pdiff[neu_mask], -pdiff[neu_mask]])
        y_0p = np.concatenate([np.ones(n_imp, dtype=int), np.zeros(2 * n_neu, dtype=int)])
        p_0p = np.concatenate([s_plus, s_zero])
        auc_0p = float(roc_auc_score(y_0p, p_0p))
        w_pm = float(n_imp) * float(n_imp)
        w_0p = 2.0 * float(n_imp) * float(n_neu)
        auc_ordinal = (w_pm * auc_dir + 2.0 * w_0p * auc_0p) / (w_pm + 2.0 * w_0p)
    elif n_imp > 0:
        auc_impact = float("nan")
        auc_ordinal = auc_dir
    else:
        auc_impact = float("nan")
        auc_ordinal = float("nan")

    def _r(v: float) -> float | None:
        return round(float(v), 4) if np.isfinite(v) else None

    return {
        "n_pairs": int(len(lbl)),
        "n_impactful": n_imp,
        "n_neutral": n_neu,
        "auc_dir": _r(auc_dir),
        "auc_impact": _r(auc_impact),
        "auc_ordinal": _r(auc_ordinal),
    }


def evaluate_predictions_with_abrank(
    suite: str,
    preds_df: pd.DataFrame,
    bdir: Path,
    *,
    pairs_df: pd.DataFrame | None = None,
    track: str = "prospective",
    n_bootstrap: int = 500,
) -> dict[str, Any]:
    """Evaluate via `abrank.evaluate.evaluate_flagship_suite` when installed, or fallback cleanly."""
    if suite in ("maturation", "discovery", "generative", "transfer"):
        try:
            from abrank.evaluate import evaluate_flagship_suite

            # For smoke-test partial subsets on maturation/generative/transfer, filter cleanly if needed
            if pairs_df is not None and suite == "maturation" and len(preds_df) < 12267:
                from abrank.evaluate import compute_per_antigen_macro_metrics

                pimp = preds_df["pred_impact"].to_numpy() if "pred_impact" in preds_df.columns else None
                return {
                    "suite": "AbRank-Maturation (subset)",
                    "track": track,
                    **compute_per_antigen_macro_metrics(
                        pairs_df, preds_df["pred_diff"].to_numpy(), pimp, min_pairs=5, n_bootstrap=n_bootstrap
                    ),
                }
            return evaluate_flagship_suite(
                suite,
                preds_df,
                data_dir=bdir,
                track=track,
                n_bootstrap=n_bootstrap,
            )
        except ImportError:
            pass

    if pairs_df is not None and "pred_diff" in preds_df.columns:
        pimp = preds_df["pred_impact"].to_numpy() if "pred_impact" in preds_df.columns else None
        return {
            "suite": suite,
            "evaluator": "fallback_3state",
            **compute_3state_metrics_fallback(
                pairs_df["label"].to_numpy(),
                pairs_df["binding_diff"].to_numpy(),
                preds_df["pred_diff"].to_numpy(),
                pimp,
            ),
        }

    return {"suite": suite, "n_predictions": int(len(preds_df))}
