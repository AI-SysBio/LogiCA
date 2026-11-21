# TCR–Epitope Recognition & Design with LogiCA (`TCRLang` + `ESM-2`)

This subfolder applies the model-agnostic **`logica`** library to couple paired T-cell receptor repertoire language models (`tcrlang-paired` via `ablang2`, representing `CDR3β|CDR3α`) with protein language models (`facebook/esm2_t12_35M_UR50D` / `150M` / `650M`, representing peptide epitopes) via native-head-preserving Cross-Attention.

Unconditional TCR and protein language models correlate **negatively** with experimental TCR–peptide affinity and activation because germline/evolutionary priors penalize functional CDR3 apex and MHC-exposed peptide substitutions. By aligning their native token distributions contrastively, `LogiCA` enables partner-specific mutation ranking, zero-shot `<5 Å` contact localization (`0.743` AUROC on 251 `TCR3d` crystal structures), and mode-collapse-free Coordinate Gibbs agonist & CDR3 design.

---

## 1. Datasets & Zero-Leakage Splits (`data/`)

All non-overlapping training and evaluation splits are included in `data/` and verified by `scripts/01_fetch_and_prepare_tcr_data.py`:

| Split / Benchmark | Path | Scale & Zero-Leakage Guarantee |
| :--- | :--- | :--- |
| **Stage-1 Repertoire Pretraining** | `data/pretrain/epact_stapler_iedb_train_non_overlap.csv` | **234,147** positive `(CDR3β\|CDR3α, epitope)` pairs (`151,482` unique TCRs, `1,559` peptides), strictly disjoint (`0.00%` overlap) from all downstream DMS test sets |
| **Stage-2 Binary & DMS Splits** | `data/finetune/epact_train_all_downstream_non_overlap/` | Supervised `EPACT` train/val splits + **71 Peptide DMS landscapes** (`26` `ePytope-Mutation`, `35` `BATCAVE`, `10` `ATLAS-Pep`), `ePytope-Binary`, and `IMMREP2025` (`600` complexes) |
| **Expanded TCR-Side DMS** | `data/expanded_tcr_mut/tcr_mutations_combined_test.csv` | **23 pure-CDR3 DMS groups · 703 deduplicated variants** combining `ATLAS-CDR`, `TRAIT 2025`, `SKEMPI v2`, and `868-Z11` 20-AA saturation DMS (`0.00%` train overlap) |

---

## 2. Step-by-Step Reproducibility Pipeline

Install the `tcr` extras from the repository root first (`pip install -e ".[tcr]"`).

```bash
# Step 1: Verify all non-overlapping splits (pass --download-external to re-fetch raw
# SKEMPI v2, TRAIT 2025, and HTTP Range-extracted Zenodo 868-Z11 Fitness TSVs)
python tcrlang-esm/scripts/01_fetch_and_prepare_tcr_data.py
python tcrlang-esm/scripts/01_fetch_and_prepare_tcr_data.py --download-external

# Step 2: Stage-1 LogiCA Pretraining (TCRLang-Paired + ESM-2)
# Modes: logica_dual, logica_tcr, logica_pep, logimlm
# Negative strategies: central_contact (P4-P8 BLOSUM62<0), non_conservative, uniform, inbatch
python tcrlang-esm/scripts/02_pretrain_tcrlang_esm.py \
    --mode logica_dual --neg-strategy central_contact --device cuda:0 --epochs 10

# Step 3: Stage-2 Supervised Fine-Tuning in Native Logit Space (preserves token heads!)
python tcrlang-esm/scripts/03_finetune_tcrlang_esm.py \
    --hf-subfolder dual-anchored-ratio5 --device cuda:0 --epochs 6

# Step 4: Evaluate Partner-Conditioned Variant Ranking across 71 Peptide DMS & 23 TCR DMS groups
# Official ydchen/logica-tcr subfolders: tcr-anchored-neg1, tcr-anchored-neg5,
# pep-anchored-neg1, pep-anchored-neg5, dual-anchored-ratio5
python tcrlang-esm/scripts/04_evaluate_dms_and_checkpoints.py --hf-subfolder tcr-anchored-neg1
python tcrlang-esm/scripts/04_evaluate_dms_and_checkpoints.py --hf-subfolder dual-anchored-ratio5

# Step 5: Epitope-Conditioned CDR3 Coordinate Gibbs Design & 2D Dependency Maps
python tcrlang-esm/scripts/05_run_tcr_gibbs_and_dependency.py --hf-subfolder dual-anchored-ratio5
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
