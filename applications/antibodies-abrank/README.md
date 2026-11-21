# Antibody & Nanobody Modeling with LogiCA (`AbLang2-Paired` + `ESM-2` on AbRank)

[![AbRank Benchmark](https://img.shields.io/badge/Benchmark-AI--SysBio%2FAbRank-blue.svg)](https://github.com/AI-SysBio/AbRank)
[![Kaggle Dataset](https://img.shields.io/badge/Kaggle-aurlienplissier%2Fabrank-20BEFF.svg)](https://www.kaggle.com/datasets/aurlienplissier/abrank)

This subfolder applies the model-agnostic **`logica`** Python library to couple paired antibody/nanobody repertoire foundation models (**`ablang2-paired`**, representing `VH|VL` or single-domain `VHH|`) with antigen protein language models (**`facebook/esm2_t12_35M_UR50D`**, `150M`, or `650M`) via native-head-preserving Gated Cross-Attention, and evaluates directly on the public [**AbRank v2**](https://github.com/AI-SysBio/AbRank) benchmark.

---

## 1. Cross-Repo Architecture (`AI-SysBio/LogiCA` $\leftrightarrow$ `AI-SysBio/AbRank`)

- **Benchmark Data, Curation & Official Evaluator**: Maintained in [**`AI-SysBio/AbRank`**](https://github.com/AI-SysBio/AbRank) and hosted ready-to-use on Kaggle at **`aurlienplissier/abrank`**.
- **Model Implementation (`AbLang2-Paired` + `ESM-2`)**: Implemented here using `from logica import LogiCA, InfoNCELogitLoss, AncestralKLAnchorLoss, ...`, loading AbRank directly from Kaggle (via `kagglehub` or `kaggle` CLI) or a local checkout of `AI-SysBio/AbRank`.
- **Zero-Friction Submission & Scoring**: `scripts/03_evaluate_on_abrank.py` writes official AbRank TSV files (`pair_id,pred_diff,pred_impact` for pairwise suites; `AbAgID,pred_score` for pointwise suites) and scores them directly with `abrank.evaluate.evaluate_flagship_suite`.

---

## 2. Setup

Install `logica` (with `ablang2` + `kagglehub`) and optionally install the official `abrank` evaluator directly from GitHub:

```bash
# From the root of AI-SysBio/LogiCA
pip install -e ".[all]"
pip install kagglehub

# Optional (recommended): install the official AbRank evaluator CLI & helpers
pip install "git+https://github.com/AI-SysBio/AbRank.git"
```

---

## 3. Step-by-Step Reproducibility Pipeline

All scripts accept either `--kaggle-dataset aurlienplissier/abrank` (auto-downloaded & cached via `kagglehub`) **or** `--abrank-dir /path/to/AbRank/data/benchmark` (or environment variable `ABRANK_DATA_DIR`).

```bash
# Step 1: Fetch & verify all AbRank master tables and 4 Core Community Suites
python applications/antibodies-abrank/scripts/01_load_abrank_data.py --kaggle-dataset aurlienplissier/abrank

# Step 2: Train AbLang2-Paired + ESM-2 LogiCA (Stage-1 InfoNCE + Stage-2 3-State Noise-Buffered Fine-Tuning)
python applications/antibodies-abrank/scripts/02_train_ablang_esm_logica.py \
    --kaggle-dataset aurlienplissier/abrank \
    --ab-model ablang2-paired \
    --ag-model facebook/esm2_t12_35M_UR50D \
    --device cuda:0

# Optional: Leave-One-Antigen-Cluster-Out (LOACO) fold training for `AbRank-Maturation`
python applications/antibodies-abrank/scripts/02_train_ablang_esm_logica.py \
    --kaggle-dataset aurlienplissier/abrank \
    --holdout-cluster AgClust_001 \
    --device cuda:0

# Optional: Single-BioFM antibody logit fine-tuning (`partner=None`)
python applications/antibodies-abrank/scripts/02_train_ablang_esm_logica.py \
    --kaggle-dataset aurlienplissier/abrank \
    --single-biofm \
    --device cuda:0

# Step 3: Evaluate on any of the 4 official AbRank Community Suites (`maturation`, `discovery`, `generative`, `transfer`)
python applications/antibodies-abrank/scripts/03_evaluate_on_abrank.py \
    --suite maturation \
    --checkpoint applications/antibodies-abrank/outputs/ablang_esm_logica/ablang_esm_logica.pt \
    --kaggle-dataset aurlienplissier/abrank

python applications/antibodies-abrank/scripts/03_evaluate_on_abrank.py \
    --suite generative \
    --checkpoint applications/antibodies-abrank/outputs/ablang_esm_logica/ablang_esm_logica.pt \
    --kaggle-dataset aurlienplissier/abrank

# Verify submission TSV independently with AI-SysBio/AbRank's official CLI:
python -m abrank.evaluate \
    --suite maturation --track prospective \
    --kaggle-dataset aurlienplissier/abrank \
    --preds applications/antibodies-abrank/outputs/abrank_eval/predictions_maturation.tsv

# Step 4: 2D Antigen -> Paratope Dependency Map & CDR-H3 Coordinate Gibbs Affinity Maturation
python applications/antibodies-abrank/scripts/04_paratope_dependency_and_gibbs.py \
    --checkpoint applications/antibodies-abrank/outputs/ablang_esm_logica/ablang_esm_logica.pt \
    --kaggle-dataset aurlienplissier/abrank
```

To run a quick end-to-end CPU verification of any script, append `--smoke-test`.

---

## 4. Minimal Python Snippet (`AbLang2-Paired` + `ESM-2`)

```python
from logica import LogiCA, plot_dependency_map, plot_gibbs_design

# Couple paired AbLang2 (VH|VL or single-domain VHH|) with ESM-2 (Antigen) in one line
model = LogiCA.from_pretrained(
    primary="ablang2-paired",
    partner="facebook/esm2_t12_35M_UR50D",
    width=256,
    layers=2,
    gate_mode="tanh_zero",
)

# Paired Trastuzumab Fv (`VH|VL`) + HER2 ectodomain antigen
ab_wt = f"{trastuzumab_vh}|{trastuzumab_vl}"
ab_mut = f"{trastuzumab_vh[:99]}W{trastuzumab_vh[100:]}|{trastuzumab_vl}"

# 1. Mutation-local antibody variant ranking conditioned on the target antigen
ranking = model.rank_variants(
    wild_type=ab_wt,
    variants=[ab_wt, ab_mut],
    context=her2_antigen_seq,
    mode="wt_marginal",
    component="guided",
)

# 2. Zero-shot antigen-sensitive paratope hotspot discovery + 2D dependency map
hotspots = model.detect_kl_hotspots(ab_wt, her2_antigen_seq, top_k=8)
dep = model.dependency_map(ab_wt, her2_antigen_seq[:80])
plot_dependency_map(dep, save_path="ab_ag_dependency.png")

# 3. Antigen-conditioned CDR-H3 Coordinate Gibbs affinity maturation
gibbs = model.gibbs_sample(
    seed_sequence=ab_wt,
    context=her2_antigen_seq,
    design_positions_0based=[98, 99, 100, 101, 102, 103],
    num_chains=8,
    num_steps=8,
    temperature=0.45,
)
plot_gibbs_design(gibbs, save_path="cdrh3_gibbs.png")
```
