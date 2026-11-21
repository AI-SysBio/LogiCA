# TCR–Peptide Benchmark Data (Hosted on Kaggle)

All processed zero-leakage datasets for `applications/tcrlang-esm/` (`~24 MB`) are hosted on Kaggle at:

- **Kaggle Dataset**: [`aurlienplissier/tcr-peptide-dms`](https://www.kaggle.com/datasets/aurlienplissier/tcr-peptide-dms)
- **Starter Notebook**: [`tcr-peptide-binding-bidirectional-dms-quickstart`](https://www.kaggle.com/code/aurlienplissier/tcr-peptide-binding-bidirectional-dms-quickstart)

Includes:
- `pretrain/` — Stage-1 zero-leakage `EPACT + STAPLER + IEDB` repertoire pretraining (`234,147` train + `26,016` val paired `CDR3β|CDR3α`–epitope pairs)
- `finetune/` — deduplicated Stage-2 supervised `EPACT` binary split (`19,882` train, `2,210` val)
- `peptide_dms/` — **71 peptide-anchored DMS groups** (`ePytope-Mutation`, `BATCAVE`, `ATLAS-Pep`)
- `tcr_dms/` — **23 TCR-anchored CDR3 DMS groups (`703` variants)** (`ATLAS-CDR`, `TRAIT 2025`, `SKEMPI v2`, `868-Z11`)
- `binary_benchmarks/` — held-out `ePytope-Binary` (`7,176`) and `IMMREP2025` (`10,000`) splits
- `tcr3d_contacts/` — `371` resolved paired TCR–pMHC crystal complexes with exact `<5 Å` heavy-atom CDR3–epitope contact labels (`101,962` residue pairs, `5,612` contacts)
- `baseline_scores/` — precomputed 12-model predictions and per-group correlations

All scripts in `applications/tcrlang-esm/scripts/` download and cache this dataset automatically via `kagglehub` (`--kaggle-dataset aurlienplissier/tcr-peptide-dms`), or accept a local folder via `--data-dir` / `LOGICA_TCR_DATA_DIR`.
