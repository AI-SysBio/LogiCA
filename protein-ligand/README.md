# Protein–Ligand Modeling with LogiCA (`ESM-2` + `SELFormer`)

This subfolder applies the model-agnostic **`logica`** library to couple protein language models (`facebook/esm2_t6_8M_UR50D` through `facebook/esm2_t33_650M_UR50D`) with small-molecule SELFIES language models (`HUBioDataLab/SELFormer`) via native-head-preserving Gated Cross-Attention.

It contains:
1. **Data Fetching & Stage-0 Curation Scripts** for both **BindingDB** (`<30%` sequence-identity low-homology pretraining) and the **137-Cohort, 561,059-Variant Protein–Ligand DMS & Enzyme Kinetics Benchmark** + **175-Structure BioLiP `<5 Å` Contact Benchmark**;
2. **Pretrained Checkpoint Loading** (`Yale-CompBio/logica` on Hugging Face Hub across `8m`, `35m`, `150m`, and `650m` scales);
3. **Stage-1 Pretraining & Stage-2 Unseen-LOO Fine-Tuning** (`LOCO-30%` unseen protein superfamily, `LOLO` unseen chemical scaffold, and `Oncology LOO` drug resistance), plus **Single-BioFM (`ESM-2` only)** and **`LogiMLM`** controls;
4. **BioLiP Cross-Modal Dependency Maps** and **Context-Conditioned Coordinate Gibbs Design**.

---

## 1. Benchmark & Data Overview

All harmonized benchmark tables (`data/processed/cohorts.parquet`, `data/processed/tasks/*.parquet`, and `data/biolip_175/*.parquet`) and source manifests (`metadata/`) are tracked directly in this folder so every evaluation and fine-tuning script runs immediately out-of-the-box:

| Benchmark Component | Path | Size & Composition |
| :--- | :--- | :--- |
| **Mutational Protein–Ligand DMS** | `data/processed/tasks/*.parquet` | **103 cohorts · 558,625 variants** across 42 unique genes & 32 `<30%` homology clusters (`LOCO-30%`), 68 multi-ligand scaffold cohorts (`LOLO`), and 29 oncology drug-resistance cohorts (`Oncology LOO`) |
| **Wild-Type Enzyme Kinetics** | `data/processed/tasks/*.parquet` | **34 cohorts · 2,434 homologs** (`CatPred-DB`, `Goldman` enzyme bioprospecting, `UniKP`/`BRENDA` methyltransferases) |
| **BioLiP `<5 Å` Contact Localization** | `data/biolip_175/*.parquet` | **175 held-out crystal complexes · 44,337 residue observations** (`0%` target overlap with DMS cohorts) |
| **BindingDB `<30%` Pretraining** | `scripts/00_fetch_and_curate_bindingdb.py` | Downloads & curates single-chain BindingDB (`<100 nM` active vs. `>10 µM` inactive, RDKit desalting + SELFIES, domain-centered 1,022-AA windows) |

---

## 2. Step-by-Step Reproducibility Pipeline

Ensure the root `logica` package is installed in editable mode (`pip install -e ".[chem]"` from the repository root).

```bash
# Step 0 (Optional — only needed if pretraining from scratch on raw BindingDB):
# Downloads raw BindingDB, desalts SMILES -> SELFIES, and builds <30% homology splits
python protein-ligand/scripts/00_fetch_and_curate_bindingdb.py --download

# Step 1: Verify the 137-cohort DMS & BioLiP datasets (pass --download-raw to re-fetch
# primary archives from ProteinGym/Zenodo/GitHub, and --score-logica to run live HF checkpoint scoring)
python protein-ligand/scripts/01_download_and_process_dms_data.py
python protein-ligand/scripts/01_download_and_process_dms_data.py --score-logica --scale 150m --max-cohorts 5

# Step 2: Stage-1 BindingDB native-logit contrastive pretraining (ESM-2 + SELFormer)
python protein-ligand/scripts/02_pretrain_bindingdb.py \
    --protein-model facebook/esm2_t30_150M_UR50D \
    --ligand-model HUBioDataLab/SELFormer \
    --device cuda:0 --epochs 10

# Step 3: Stage-2 Unseen-LOO Fine-Tuning (LOCO-30%, LOLO, LogiMLM, or Single-BioFM ESM-2 control)
python protein-ligand/scripts/03_finetune_dms_loo.py --scale 150m --split loco30 --objective logica --device cuda:0
python protein-ligand/scripts/03_finetune_dms_loo.py --scale 150m --split lolo   --objective logica --device cuda:0
python protein-ligand/scripts/03_finetune_dms_loo.py --scale 150m --single-biofm --device cuda:0

# Step 4: Held-out BioLiP-175 <5 Å cross-modal dependency map localization
python protein-ligand/scripts/04_evaluate_dependency_maps_biolip.py --scale 35m --device cuda:0

# Step 5: Context-conditioned Coordinate Gibbs sampling (103 DMS cohorts + live TtgR Phloretin pocket design)
python protein-ligand/scripts/05_run_gibbs_sampling.py --live-design --scale 35m

# Step 6: Hierarchical G=42 gene-collapsed & K=32 <30% cluster-collapsed statistical audit
python protein-ligand/scripts/06_run_mlm_ablation_and_scaling.py
```

---

## 3. Minimal Python Snippet (`ESM-2` + `SELFormer`)

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
