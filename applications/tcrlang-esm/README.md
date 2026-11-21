# TCR–Epitope Recognition & Design with LogiCA (`TCRLang` + `ESM-2`)

[![Kaggle Dataset](https://img.shields.io/badge/Kaggle-aurlienplissier%2Ftcr--peptide--dms-20BEFF.svg)](https://www.kaggle.com/datasets/aurlienplissier/tcr-peptide-dms)
[![Kaggle Notebook](https://img.shields.io/badge/Notebook-Quickstart-20BEFF.svg)](https://www.kaggle.com/code/aurlienplissier/tcr-peptide-binding-bidirectional-dms-quickstart)
[![HF Checkpoints](https://img.shields.io/badge/HuggingFace-ydchen%2Flogica--tcr-FFD21E.svg)](https://huggingface.co/ydchen/logica-tcr)

This subfolder applies the model-agnostic **`logica`** library to couple paired T-cell receptor repertoire language models (`tcrlang-paired` via `ablang2`, representing `CDR3β|CDR3α`) with protein language models (`facebook/esm2_t12_35M_UR50D` / `150M` / `650M`, representing peptide epitopes) via native-head-preserving Cross-Attention.

Unconditional TCR and protein language models correlate **negatively** with experimental TCR–peptide affinity and activation because germline/evolutionary priors penalize functional CDR3 apex and MHC-exposed peptide substitutions. By aligning their native token distributions contrastively, `LogiCA` enables partner-specific mutation ranking, zero-shot `<5 Å` contact localization (`0.743` AUROC on `TCR3d` crystal structures), and mode-collapse-free Coordinate Gibbs agonist & CDR3 design.

---

## 1. Datasets & Zero-Leakage Splits (Hosted on Kaggle)

All processed zero-leakage training splits, **peptide-anchored DMS**, **TCR-anchored CDR3 DMS**, binary benchmarks, **TCR3d `<5 Å` crystal contacts**, and 12-model baseline tables (`~24 MB` total) are hosted on Kaggle at [**`aurlienplissier/tcr-peptide-dms`**](https://www.kaggle.com/datasets/aurlienplissier/tcr-peptide-dms) (with an interactive [Quickstart Notebook](https://www.kaggle.com/code/aurlienplissier/tcr-peptide-binding-bidirectional-dms-quickstart)) and are **automatically fetched via `kagglehub`** by every script below (or point to a local copy via `--data-dir` / `LOGICA_TCR_DATA_DIR`):

| Split / Benchmark | Kaggle Path | Scale & Zero-Leakage Guarantee |
| :--- | :--- | :--- |
| **Stage-1 Repertoire Pretraining** | `pretrain/epact_stapler_iedb_{train,val}_non_overlap.csv` | **234,147** train + **26,016** val positive `(CDR3β\|CDR3α, epitope)` pairs (`151,482` unique TCRs, `1,559` epitopes), strictly `0.00%` overlap with all downstream test sets |
| **Stage-2 Supervised `EPACT` Splits** | `finetune/epact_{train,val}.csv` | Deduplicated supervised `EPACT` binary split (`19,882` train, `2,210` val) |
| **Peptide-Anchored DMS (`71` groups)** | `peptide_dms/*.csv` | `ePytope-Mutation` (`26` groups, `4,244`), `BATCAVE` (`35` groups, `5,754` / `11,967` full), `ATLAS-Pep` (`10` groups, `60`) |
| **TCR-Anchored CDR3 DMS (`23` groups)** | `tcr_dms/*.csv` | **703 deduplicated pure-CDR3 variants** combining `ATLAS-CDR`, `TRAIT 2025`, `SKEMPI v2`, and `868-Z11` 20-AA saturation DMS (`0.00%` train overlap) |
| **Binary Recognition Benchmarks** | `binary_benchmarks/*.csv` | `ePytope-Binary` (`7,176` pairs) and `IMMREP2025` (`10,000` pairs with HLA alleles) |
| **TCR3d `<5 Å` Crystal Contacts** | `tcr3d_contacts/{tcr3d_complexes.csv,tcr3d_residue_contacts.parquet}` | **`371`** resolved paired TCR–pMHC crystal complexes with exact heavy-atom CDR3α/β–epitope `<5 Å` contacts (`101,962` residue pairs, `5,612` contacts) |
| **12-Model Baseline Tables** | `baseline_scores/*.csv` | Per-variant predictions (`703` TCR variants) and per-group correlations across all `71` peptide + `23` TCR DMS groups |

```python
from pathlib import Path
import kagglehub, pandas as pd

root = Path(kagglehub.dataset_download("aurlienplissier/tcr-peptide-dms"))
pretrain = pd.read_csv(root / "pretrain/epact_stapler_iedb_train_non_overlap.csv")
pep_dms = pd.read_csv(root / "peptide_dms/epytope_mutation_test.csv")
tcr_dms = pd.read_csv(root / "tcr_dms/tcr_mutations_combined_test.csv")
tcr3d = pd.read_parquet(root / "tcr3d_contacts/tcr3d_residue_contacts.parquet")
```

---

## 2. Step-by-Step Reproducibility Pipeline

Install the `tcr` extras from the repository root first (`pip install -e ".[tcr]"` — includes `ablang2` and `kagglehub`). Every script auto-downloads `aurlienplissier/tcr-peptide-dms` on first run:

```bash
# Step 1: Fetch from Kaggle & verify all zero-leakage splits + TCR3d contacts
python applications/tcrlang-esm/scripts/01_fetch_and_prepare_tcr_data.py --kaggle-dataset aurlienplissier/tcr-peptide-dms
# Optional: re-fetch raw SKEMPI v2, TRAIT 2025, and Zenodo 868-Z11 archives
python applications/tcrlang-esm/scripts/01_fetch_and_prepare_tcr_data.py --download-external

# Step 2: Stage-1 LogiCA Pretraining (TCRLang-Paired + ESM-2)
# Modes: logica_dual, logica_tcr, logica_pep, logimlm
# Negative strategies: central_contact (P4-P8 BLOSUM62<0), non_conservative, uniform, inbatch
python applications/tcrlang-esm/scripts/02_pretrain_tcrlang_esm.py \
    --mode logica_dual --neg-strategy central_contact --device cuda:0 --epochs 10

# Step 3: Stage-2 Supervised Fine-Tuning in Native Logit Space (preserves token heads!)
python applications/tcrlang-esm/scripts/03_finetune_tcrlang_esm.py \
    --hf-subfolder dual-anchored-ratio5 --device cuda:0 --epochs 6

# Step 4: Evaluate Partner-Conditioned Variant Ranking across 71 Peptide DMS & 23 TCR DMS groups
# Official ydchen/logica-tcr subfolders: tcr-anchored-neg1, tcr-anchored-neg5,
# pep-anchored-neg1, pep-anchored-neg5, dual-anchored-ratio5
python applications/tcrlang-esm/scripts/04_evaluate_dms_and_checkpoints.py --hf-subfolder tcr-anchored-neg1
python applications/tcrlang-esm/scripts/04_evaluate_dms_and_checkpoints.py --hf-subfolder dual-anchored-ratio5

# Step 5: Epitope-Conditioned CDR3 Coordinate Gibbs Design & 2D Dependency Maps
python applications/tcrlang-esm/scripts/05_run_tcr_gibbs_and_dependency.py --hf-subfolder dual-anchored-ratio5
```

---

## 3. Minimal Python Snippet (`TCRLang` + `ESM-2`)

```python
from logica import LogiCA

# Load pretrained TCRLang-Paired + ESM-2 LogiCA checkpoint directly from Hugging Face Hub
model = LogiCA.from_checkpoint("ydchen/logica-tcr", subfolder="dual-anchored-ratio5", device="cpu")

tcr_1g4 = "CASSYVGNTGELFF|CAVRPTSGGSYIPTF"
ny_eso_1 = "SLLMWITQC"

# 1. Rank paired CDR3 variants conditioned on the cognate NY-ESO-1 epitope
ranking = model.rank_variants(
    wild_type=tcr_1g4,
    variants=[
        "CASSYVGNTGELFF|CAVRPTAGGSYIPTF",
        "CASSYVGATGELFF|CAVRPTSGGSYIPTF",
        "CASSWVGNTGELFF|CAVRPTSGGSYIPTF",
    ],
    context=ny_eso_1,
    mode="masked_sites",
    reduction="sum",
    component="guided",
)
print("Ranked CDR3 variants:", ranking.top_k(3))

# 2. Compute 2D Peptide -> CDR3 dependency map
dep = model.dependency_map(tcr_1g4, ny_eso_1)
print("Top peptide-sensitive CDR3 residues:", dep.top_primary_sites(5))

# 3. Epitope-conditioned Coordinate Gibbs design over CDR3β apex positions
gibbs = model.gibbs_sample(tcr_1g4, context=ny_eso_1, design_positions_0based=[4, 5, 6, 7], num_chains=8, num_steps=10)
print("Top designed TCRs:", gibbs.top_designs(3))
```
