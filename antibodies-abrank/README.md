# Antibody & Nanobody Modeling with LogiCA (`AbRank`)

> **Status: Coming Soon**

This folder will host the **Antibody–Antigen (`LogiCA-Ab`)** adaptation and the **AbRank** evaluation suite powered by the model-agnostic `logica` library.

## Planned Release Contents

- **Pretrained & Fine-Tuned Checkpoints**: Native-logit cross-modal alignment coupling paired antibody/nanobody repertoire foundation models (`ablang2-paired`, `ESM-2`) with antigen sequence and structure encoders (`ESM-2`, `SaProt`, `ESM-IF1`).
- **Standardized Sequence- & Lineage-Disjoint Splits**: Zero-leakage `<50%` antigen-homology and `<80%` clonal-lineage benchmarks spanning affinity maturation (`L1`–`L3` CDR edits), library candidate discovery, wet-lab generative antibody designs, and viral neutralization/escape transfer.
- **Paratope–Epitope Dependency Maps & CDR Gibbs Maturation**: Zero-shot cross-modal contact localization and context-conditioned CDR loop optimization using `logica.DependencyMapper` and `logica.GibbsSampler`.

Stay tuned for the public release! In the meantime, you can already instantiate paired antibody–antigen `LogiCA` models directly with the core library:

```python
from logica import LogiCA

# Couple paired AbLang2 (VH|VL) with ESM-2 (Antigen) in two lines of code
model = LogiCA.from_pretrained("ablang2-paired", "facebook/esm2_t30_150M_UR50D")
```
