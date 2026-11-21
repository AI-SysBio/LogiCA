# Protein–Ligand Benchmark Data (Hosted on Kaggle)

All processed datasets for `applications/protein-ligand/` (`~175 MB` compressed) are hosted on Kaggle at:

- **Kaggle Dataset**: [`aurlienplissier/protein-ligand-dms`](https://www.kaggle.com/datasets/aurlienplissier/protein-ligand-dms)
- **Starter Notebook**: [`protein-ligand-dms-biolip-175-quickstart`](https://www.kaggle.com/code/aurlienplissier/protein-ligand-dms-biolip-175-quickstart)

Includes:
- `cohorts.csv` & `dms_tasks/*.parquet` — `137` harmonized protein–ligand DMS & enzyme kinetics cohorts (`561,059` variants; pure experimental columns)
- `baseline_scores/*_baselines.parquet` — precomputed zero-shot scores across 10 protein/ligand/structure baselines
- `bindingdb_pretrain/` — Stage-0 curated BindingDB pretraining corpus (`10,235` single-chain proteins, `1,127,953` compounds, `1.85M` `<30%`-homology train/val pairs)
- `biolip_175/` — held-out `<5 Å` contact-localization benchmark (`175` crystal complexes, `44,337` residues) and precomputed per-residue dependency maps

All scripts in `applications/protein-ligand/scripts/` download and cache this dataset automatically via `kagglehub` (`--kaggle-dataset aurlienplissier/protein-ligand-dms`), or accept a local folder via `--data-dir` / `LOGICA_PL_DATA_DIR`.
