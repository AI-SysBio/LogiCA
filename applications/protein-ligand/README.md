# Protein–Ligand Modeling with LogiCA (`ESM-2` + `SELFormer`)

[![Kaggle Dataset](https://img.shields.io/badge/Kaggle-aurlienplissier%2Fprotein--ligand--dms-20BEFF.svg)](https://www.kaggle.com/datasets/aurlienplissier/protein-ligand-dms)
[![Kaggle Notebook](https://img.shields.io/badge/Notebook-Quickstart-20BEFF.svg)](https://www.kaggle.com/code/aurlienplissier/protein-ligand-dms-biolip-175-quickstart)
[![HF Checkpoints](https://img.shields.io/badge/HuggingFace-Yale--CompBio%2Flogica-FFD21E.svg)](https://huggingface.co/Yale-CompBio/logica)

This subfolder applies the model-agnostic **`logica`** library to couple protein language models (`facebook/esm2_t6_8M_UR50D` through `facebook/esm2_t33_650M_UR50D`) with small-molecule SELFIES language models (`HUBioDataLab/SELFormer`) via native-head-preserving Gated Cross-Attention.

It contains:
1. **Ready-to-Download Kaggle Dataset (`aurlienplissier/protein-ligand-dms`)** + Stage-0 raw curation scripts covering **BindingDB** (`<30%` sequence-identity low-homology pretraining), the **137-Cohort, 561,059-Variant Protein–Ligand DMS & Enzyme Kinetics Benchmark** (with separated precomputed 10-model baseline scores), and the **BioLiP-175 `<5 Å` Contact Benchmark & Precomputed Dependency Maps**;
2. **Pretrained Checkpoint Loading** (`Yale-CompBio/logica` on Hugging Face Hub across `8m`, `35m`, `150m`, and `650m` scales);
3. **Stage-1 Pretraining & Stage-2 Unseen-LOO Fine-Tuning** (`LOCO-30%` unseen protein superfamily, `LOLO` unseen chemical scaffold, and `Oncology LOO` drug resistance), plus **Single-BioFM (`ESM-2` only)** and **`LogiMLM`** controls;
4. **BioLiP Cross-Modal Dependency Maps** and **Context-Conditioned Coordinate Gibbs Design**.

---

## 1. Pretrained Model Weights (Hugging Face)

Official pretrained `ESM-2` $\leftrightarrow$ `SELFormer` LogiCA checkpoints are hosted on Hugging Face Hub at [**`Yale-CompBio/logica`**](https://huggingface.co/Yale-CompBio/logica) and auto-downloaded by `LogiCA.from_checkpoint(...)`:

| Scale | Protein BioFM | Ligand BioFM | Hugging Face Checkpoint |
| :--- | :--- | :--- | :--- |
| **`8m`** | `facebook/esm2_t6_8M_UR50D` | `HUBioDataLab/SELFormer` | [`Yale-CompBio/logica` (`checkpoints/8m/best.pt`)](https://huggingface.co/Yale-CompBio/logica/blob/main/checkpoints/8m/best.pt) |
| **`35m`** | `facebook/esm2_t12_35M_UR50D` | `HUBioDataLab/SELFormer` | [`Yale-CompBio/logica` (`checkpoints/35m/best.pt`)](https://huggingface.co/Yale-CompBio/logica/blob/main/checkpoints/35m/best.pt) |
| **`150m`** | `facebook/esm2_t30_150M_UR50D` | `HUBioDataLab/SELFormer` | [`Yale-CompBio/logica` (`checkpoints/150m/best.pt`)](https://huggingface.co/Yale-CompBio/logica/blob/main/checkpoints/150m/best.pt) |
| **`650m`** | `facebook/esm2_t33_650M_UR50D` | `HUBioDataLab/SELFormer` | [`Yale-CompBio/logica` (`checkpoints/650m/best.pt`)](https://huggingface.co/Yale-CompBio/logica/blob/main/checkpoints/650m/best.pt) |

---

## 2. Benchmark & Data Overview (Hosted on Kaggle)

All processed benchmark tables (`~175 MB` total) are hosted on Kaggle at [**`kaggle/protein-ligand-dms`**](https://www.kaggle.com/datasets/aurlienplissier/protein-ligand-dms) (with an interactive [Quickstart Notebook](https://www.kaggle.com/code/aurlienplissier/protein-ligand-dms-biolip-175-quickstart)) and are **automatically fetched and cached via `kagglehub`** by every script in `scripts/` (or point to a local copy via `--data-dir` / `LOGICA_PL_DATA_DIR`):

| Benchmark Component | Kaggle Path | Size & Composition |
| :--- | :--- | :--- |
| **Cohort Index & Splits** | `cohorts.csv` | Master table of all **137 cohorts** with `LOCO-30%`, `LOLO`, and `Oncology LOO` cluster/scaffold assignments |
| **Pure Experimental DMS & Kinetics** | `dms_tasks/*.parquet` | **103 mutational DMS cohorts (`558,625` variants)** + **34 WT enzyme kinetics cohorts (`2,434` homologs)** in a harmonized 22-column experimental schema |
| **Precomputed Baseline Scores** | `baseline_scores/*_baselines.parquet` | 1-to-1 aligned zero-shot scores across 10 baselines (`LogiCA`, `ESM-2 650M`, `ESM-C 600M`, `VenusREM`, `PoET`, `MoCHI`, `CatPred`, `LigandMPNN`, `SaProt`, `ThermoMPNN`) |
| **Stage-0 BindingDB Pretraining** | `bindingdb_pretrain/*.parquet` | **`10,235`** single-chain domain-windowed proteins, **`1,127,953`** desalted SMILES/SELFIES compounds, **`1,631,404`** train + **`222,547`** `<30%`-homology val pairs |
| **BioLiP-175 `<5 Å` Contacts & Dependency Maps** | `biolip_175/` & `biolip_175/dependency_maps/` | **175 held-out crystal complexes · `44,337` residues** (`0%` DMS target overlap) + 6 precomputed LogiCA per-residue dependency-map profiles and 14-model benchmark summary |

```python
from pathlib import Path
import kagglehub, pandas as pd

root = Path(kagglehub.dataset_download("aurlienplissier/protein-ligand-dms"))
cohorts = pd.read_csv(root / "cohorts.csv")
kinase_dms = pd.read_parquet(root / "dms_tasks/kinase_inhibitor_panel_dms.parquet")
biolip_deps = pd.read_parquet(root / "biolip_175/dependency_maps/biolip_dependency_profiles.parquet")
```

---

## 3. Step-by-Step Reproducibility Pipeline

Install `logica` with the `chem` extras (`pip install -e ".[chem]"` from the repository root — includes `kagglehub`, `rdkit`, `selfies`, and `pyarrow`). Every script auto-downloads `aurlienplissier/protein-ligand-dms` on first run:

```bash
# Step 0 (Optional — only if regenerating BindingDB from raw BindingDB TSV instead of using Kaggle):
python applications/protein-ligand/scripts/00_fetch_and_curate_bindingdb.py --download

# Step 1: Fetch from Kaggle & verify all 137 DMS/kinetics cohorts + BioLiP-175
python applications/protein-ligand/scripts/01_download_and_process_dms_data.py --kaggle-dataset aurlienplissier/protein-ligand-dms
python applications/protein-ligand/scripts/01_download_and_process_dms_data.py --score-logica --scale 150m --max-cohorts 5

# Step 2: Stage-1 BindingDB native-logit contrastive pretraining (auto-loads bindingdb_pretrain/ from Kaggle)
python applications/protein-ligand/scripts/02_pretrain_bindingdb.py \
    --protein-model facebook/esm2_t30_150M_UR50D \
    --ligand-model HUBioDataLab/SELFormer \
    --device cuda:0 --epochs 10

# Step 3: Stage-2 Unseen-LOO Fine-Tuning (LOCO-30%, LOLO, LogiMLM, or Single-BioFM ESM-2 control)
python applications/protein-ligand/scripts/03_finetune_dms_loo.py --scale 150m --split loco30 --objective logica --device cuda:0
python applications/protein-ligand/scripts/03_finetune_dms_loo.py --scale 150m --split lolo   --objective logica --device cuda:0
python applications/protein-ligand/scripts/03_finetune_dms_loo.py --scale 150m --single-biofm --device cuda:0

# Step 4: Held-out BioLiP-175 <5 Å cross-modal dependency map localization
python applications/protein-ligand/scripts/04_evaluate_dependency_maps_biolip.py --scale 35m --device cuda:0

# Step 5: Context-conditioned Coordinate Gibbs sampling (103 DMS cohorts + live TtgR Phloretin pocket design)
python applications/protein-ligand/scripts/05_run_gibbs_sampling.py --live-design --scale 35m

# Step 6: Hierarchical G=42 gene-collapsed & K=32 <30% cluster-collapsed statistical audit
python applications/protein-ligand/scripts/06_run_mlm_ablation_and_scaling.py
```

---

## 4. Minimal Python Snippet (`ESM-2` + `SELFormer`)

```python
from logica import LogiCA

# Automatically downloads ESM-2 (150M), HUBioDataLab/SELFormer, and Yale-CompBio/logica weights
model = LogiCA.from_checkpoint("Yale-CompBio/logica", scale="150m", device="cpu")

# 1. Rank EGFR kinase mutations under Gefitinib (accepts raw SMILES or SELFIES!)
gefitinib = "COc1cc2ncnc(Nc3ccc(F)c(Cl)c3)c2cc1OCCCN1CCOCC1"
egfr_wt = "IPVAIKELREATSPKANKEILDEAYVMASVDNPHVCRLLGICLTSTVQLITQLMPFGCLLDYVREHKDNIGSQYLLNWCVQIAKGMNYLEDRRLVHRDLAARNVLVKTPQHVKITDFGLAKLLGAEEKEYHAEGGKVPIKWMALESILHRIYTHQSDVWSYGVTVWELMTFGSKPYDGIPASEISSILEKGERLPQPPICTIDVYMIMVKCWMIDADSRPKFRELIIEFSKMARDPQRYLVI"

ranking = model.rank_variants(
    wild_type=egfr_wt,
    variants=["T84I", "L147R", "T84M_L147R"],  # local coordinates within kinase domain
    context=gefitinib,
    variants_are_mutation_strings=True,
)
print("Top ranked variants:", ranking.top_k(3))

# 2. Compute 2D residue-by-ligand dependency map & zero-shot KL binding-site hotspots
dep_map = model.dependency_map(egfr_wt, gefitinib, metric="sym_kl")
print("Top ligand-coupled residues:", dep_map.top_primary_sites(5))

# 3. Context-conditioned Coordinate Gibbs pocket design
designs = model.gibbs_sample(egfr_wt, context=gefitinib, auto_kl_top_k=4, num_chains=8, num_steps=10)
print("Top Gibbs designs:", designs.top_designs(3))
```
