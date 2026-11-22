# Antibody & Nanobody Modeling with LogiCA (`AbLang2-Paired` + `ESM-2` on AbRank & SAbDab2)

[![AbRank Benchmark](https://img.shields.io/badge/Benchmark-AI--SysBio%2FAbRank-blue.svg)](https://github.com/AI-SysBio/AbRank)
[![Kaggle Dataset](https://img.shields.io/badge/Kaggle-aurlienplissier%2Fabrank-20BEFF.svg)](https://www.kaggle.com/datasets/aurlienplissier/abrank)
[![HF Checkpoints](https://img.shields.io/badge/HuggingFace-Yale--CompBio%2Flogica--abag-FFD21E.svg)](https://huggingface.co/Yale-CompBio/logica-abag)

This subfolder applies the model-agnostic **`logica`** Python library to couple paired antibody/nanobody repertoire foundation models (**`ablang2-paired`**, representing `VH|VL` or single-domain `VHH|`) with antigen protein language models (**`facebook/esm2_t12_35M_UR50D`**, `150M`, or `650M`) via native-head-preserving Gated Cross-Attention, evaluating on the public [**AbRank v2**](https://github.com/AI-SysBio/AbRank) benchmark and on crystal structures downloaded from [**SAbDab2**](https://sabdab.opig.stats.ox.ac.uk/).

---

## 1. Pretrained Model Weights (Hugging Face)

Official pretrained `AbLang2-Paired` $\leftrightarrow$ `ESM-2` LogiCA weights are hosted on Hugging Face Hub at [**`Yale-CompBio/logica-abag`**](https://huggingface.co/Yale-CompBio/logica-abag) and auto-downloaded (with automatic LoRA merging) by `LogiCA.from_checkpoint(...)`:

| Scale | Antibody BioFM | Antigen BioFM | Hugging Face Checkpoint |
| :--- | :--- | :--- | :--- |
| **`esm35m`** | `ablang2-paired` | `facebook/esm2_t12_35M_UR50D` | [`Yale-CompBio/logica-abag` (`checkpoints/esm35m/best.pt`)](https://huggingface.co/Yale-CompBio/logica-abag/blob/main/checkpoints/esm35m/best.pt) · [`train_summary.json`](https://huggingface.co/Yale-CompBio/logica-abag/blob/main/checkpoints/esm35m/train_summary.json) |

### Reference Structural & Benchmark Evaluation (`esm35m`)

- **SAbDab2 `<5 Å` Contact Localization (25 non-redundant crystal complexes, zero structural supervision):**
  - Single-pass KL-1D Paratope AUROC / AUPR: **`0.7398`** / `0.3223`
  - 2D Dependency Profile (`Dep2D-1D`) AUROC / AUPR: **`0.7925`** / `0.3980` (2D Pair AUROC: `0.6561`)
  - Top-10 CDR / `<5 Å` Contact Fraction: **`56.4%`** / **`40.4%`**
- **SAbDab2 CDR-H3 Coordinate Gibbs Sampling (25 complexes, Holo vs. Unconditional Apo `AbLang2`):**
  - Native CDR-H3 Sequence Recovery: **`0.5360`** (Holo) vs. `0.5250` (Apo)
  - Mean Native-Logit Score: **`0.7873`** (Holo) vs. `0.7127` (Apo) · Effective Diversity $N_{\text{eff}} = \mathbf{11.13}$
- **AbRank Benchmark Split Note:** The released `esm35m` checkpoint was trained on `splits/unseen_ag/train.tsv` (`holdout_cluster=None`; strictly held-out `unseen_ag/test.tsv` L1–L3 per-antigen Impact AUC = **`0.6451`**). For strict 22-cluster LOACO evaluation on `AbRank-Maturation` or 26-cluster LOACO on `AbRank-Generative`, train per-cluster folds using `--holdout-cluster` in `scripts/02_train_ablang_esm_logica.py`.

---

## 2. Cross-Repo Architecture (`AI-SysBio/LogiCA` $\leftrightarrow$ `AI-SysBio/AbRank`)

- **Benchmark Data, Curation & Official Evaluator**: Maintained in [**`AI-SysBio/AbRank`**](https://github.com/AI-SysBio/AbRank) and hosted ready-to-use on Kaggle at **`aurlienplissier/abrank`**.
- **Structural Contact Benchmark (`SAbDab2`)**: `scripts/05_sabdab_dependency_and_gibbs.py` downloads the live SAbDab2 structural database summary and RCSB mmCIF crystal structures directly, extracts exact `<5 Å` heavy-atom antibody–antigen contacts, evaluates zero-shot cross-modal dependency maps, and benchmarks `AbLang2`–`ESM-2` Coordinate Gibbs sampling against unconditional `AbLang2`.
- **Zero-Friction Submission & Scoring**: `scripts/03_evaluate_on_abrank.py` writes official AbRank TSV files (`pair_id,pred_diff,pred_impact` for pairwise suites; `AbAgID,pred_score` for pointwise suites) and scores them directly with `abrank.evaluate.evaluate_flagship_suite`.

---

## 3. Setup

Install `logica` (with `ablang2` + `kagglehub`) and optionally install the official `abrank` evaluator directly from GitHub:

```bash
# From the root of AI-SysBio/LogiCA
pip install -e ".[all]"
pip install kagglehub

# Optional (recommended): install the official AbRank evaluator CLI & helpers
pip install "git+https://github.com/AI-SysBio/AbRank.git"
```

---

## 4. Step-by-Step Reproducibility Pipeline

All scripts accept either `--kaggle-dataset aurlienplissier/abrank` (auto-downloaded & cached via `kagglehub`) **or** `--abrank-dir /path/to/AbRank/data/benchmark` (or environment variable `ABRANK_DATA_DIR`), and accept `--checkpoint Yale-CompBio/logica-abag` (or any local `.pt` checkpoint).

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

# Step 3: Evaluate on AbRank suites (`unseen_ag_test`, `maturation`, `discovery`, `generative`, `transfer`)
python applications/antibodies-abrank/scripts/03_evaluate_on_abrank.py \
    --suite unseen_ag_test \
    --checkpoint Yale-CompBio/logica-abag \
    --kaggle-dataset aurlienplissier/abrank

python applications/antibodies-abrank/scripts/03_evaluate_on_abrank.py \
    --suite generative \
    --checkpoint Yale-CompBio/logica-abag \
    --kaggle-dataset aurlienplissier/abrank

# Verify submission TSV independently with AI-SysBio/AbRank's official CLI:
python -m abrank.evaluate \
    --suite maturation --track prospective \
    --kaggle-dataset aurlienplissier/abrank \
    --preds applications/antibodies-abrank/outputs/abrank_eval/predictions_maturation.tsv

# Step 4: Single-Complex Antigen -> Paratope Dependency Map & CDR-H3 Coordinate Gibbs Design
python applications/antibodies-abrank/scripts/04_paratope_dependency_and_gibbs.py \
    --checkpoint Yale-CompBio/logica-abag \
    --kaggle-dataset aurlienplissier/abrank

# Step 5: Download SAbDab2 Structures, Extract `<5 Å` Paratope Contacts, Dependency Maps & AbLang2–ESM-2 Gibbs Evaluation
python applications/antibodies-abrank/scripts/05_sabdab_dependency_and_gibbs.py \
    --hf-repo Yale-CompBio/logica-abag \
    --max-complexes 25 \
    --design-mode cdrh3 \
    --device cuda:0

# Target specific crystal complexes (e.g. Trastuzumab-HER2 1N8Z, HyHEL-10 1FDL, Bevacizumab-VEGF 2FJG)
python applications/antibodies-abrank/scripts/05_sabdab_dependency_and_gibbs.py \
    --hf-repo Yale-CompBio/logica-abag \
    --pdb-ids 1N8Z,1FDL,2FJG \
    --design-mode cdrh3 \
    --device cpu
```

To run a quick end-to-end CPU verification of any script (including live SAbDab2 + RCSB mmCIF download on `1N8Z` and `1FDL`), append `--smoke-test`.

---

## 5. Minimal Python Snippet (`AbLang2-Paired` + `ESM-2`)

```python
from logica import LogiCA, plot_dependency_map, plot_gibbs_design

# Load official pretrained AbLang2-Paired + ESM-2 (35M) LogiCA checkpoint from Hugging Face Hub
model = LogiCA.from_checkpoint("Yale-CompBio/logica-abag", scale="esm35m", device="cpu")

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
