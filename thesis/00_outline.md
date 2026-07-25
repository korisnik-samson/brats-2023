# Master Thesis — Annotated Outline (Blueprint)

> **University:** Univerzitet Singidunum, Departman za poslediplomske studije
> **Candidate:** Samson Offorjindu · **Mentor:** prof. dr Nebojša Bačanin Džakula · **Belgrade, 2026**
> **Language:** English body, with Serbian title page + Serbian abstract (Apstrakt/Rezime).
> **Length target:** ≤ 40 pages of original research (front/back matter excluded).
> **Style:** Times New Roman 12, A4, justified, indented paragraphs, first-person plural
> ("we"), foreign terms in *italic*, Harvard/APA citations in footnotes **and** in the
> alphabetical reference list. Numbered pages (bottom-centre) except the title page.

---

## Working title (selected)

- **EN:** **One Pipeline, Three Tumours: Unified Deep-Learning Segmentation of Adult,
  Meningioma and Paediatric Brain Tumours on a Single GPU (BraTS 2023)**
- **SR (Naziv rada):** _[to finalise]_ *Jedna metodologija, tri tumora: jedinstvena
  segmentacija tumora mozga odraslih, meningioma i pedijatrijskih tumora dubokim učenjem
  na jednom grafičkom procesoru (BraTS 2023)*

---

## Front matter (not counted in the 40 pages)

| # | Element | Notes |
|---|---------|-------|
| 1 | **Naslovna strana** (title page) | Exact template layout: MASTER RAD / title / Mentor / Kandidat / Beograd 2026. |
| 2 | **Izjava o autorstvu** (authorship statement) | Signed declaration required by Singidunum. |
| 3 | **Apstrakt** (EN) + **Rezime** (SR) | ~200–250 words each + 5–6 keywords / ključne reči. |
| 4 | **Sadržaj** (auto TOC) | Generated from heading styles. |
| 5 | List of abbreviations | BraTS, GLI, MEN, PED, WT, TC, ET, DSC, HD95, AMP, CNN, ViT, AdamW… |
| 6 | List of figures & tables | Auto-generated. |

---

## Body chapters (the ≤40 pages)

### 1. Uvod / Introduction  — *target ~3–4 pp*
The template **requires** this chapter to define subject/problem, objectives (scientific +
social), and methods. Sub-sections:

- **1.1 Predmet i problem istraživanja** — Brain-tumour segmentation from multi-parametric
  MRI; clinical importance (treatment planning, surgery, monitoring); manual segmentation
  is slow, expensive and inter-rater variable; the BraTS 2023 challenge and its three
  biologically distinct populations (GLI, MEN, PED). Problem statement: can *one* pipeline
  serve all three under consumer hardware?
- **1.2 Ciljevi istraživanja**
  - *Naučni (scientific):* establish a reproducible baseline; quantify which pipeline
    components are domain-general vs tumour-specific; compare CNN vs CNN-Transformer under
    identical conditions; evaluate cross-challenge transfer.
  - *Društveni (social):* lower the barrier to medical-imaging research by showing
    competitive results are achievable on a single consumer GPU (RTX 4070, 12 GB),
    supporting reproducibility and access in resource-limited settings.
- **1.3 Hipoteze (hypotheses)** — H1: a unified methodology with task-specific adaptations
  yields clinically meaningful segmentation across GLI/MEN/PED. H2: a hybrid
  CNN-Transformer (Swin UNETR) is competitive with / superior to a pure 3D CNN under an
  identical training budget. H3: enhancing-tumour (ET) difficulty is population-dependent
  (easy in adult GLI, hard in paediatric DMG). H4: a GLI-pretrained encoder improves PED
  segmentation versus PED-only training.
- **1.4 Metode istraživanja** — short overview: quantitative experimental study; public
  BraTS 2023 dataset (~2,350 cases ≫ the 50-unit minimum); 5-fold stratified CV; deep
  CNN + transformer models; statistical comparison of DSC/HD95.
- **1.5 Struktura rada** — one paragraph mapping the chapters.

### 2. Teorijski okvir / Theoretical Background — *target ~6–8 pp*
Reference-heavy; mostly from the competitive-landscape + domain sections.

- **2.1** Brain tumours: adult glioma, meningioma, paediatric high-grade glioma / DMG —
  biology, epidemiology, clinical relevance, why they differ.
- **2.2** Multi-parametric MRI (T1, T1c=*t1c*, T2, FLAIR=*t2f*) and the segmentation labels:
  disjoint (NCR, ED, ET) vs evaluation regions (WT ⊇ TC ⊇ ET).
- **2.3** Medical-image segmentation: from atlas/threshold methods to deep learning.
- **2.4** Convolutional architectures: U-Net → 3D U-Net → nnU-Net → SegResNet/MedNeXt.
- **2.5** Transformers for vision: ViT, Swin Transformer, and **Swin UNETR** (hybrid).
- **2.6** The BraTS challenge: history and the 2023 GLI/MEN/PED sub-challenges.
- **2.7** Related work — BraTS 2023 winners: GLI (Univ. Minho; nnU-Net + Swin UNETR +
  GliGAN synthetic data), PED (CNMC; label-wise ensemble + ET redefinition), MEN
  (SegResNet/MedNeXt + deep supervision). Universal top-team patterns.
- **2.8** Evaluation metrics: Dice (DSC), 95th-percentile Hausdorff (HD95), and the 2023
  **lesion-wise** metric.

### 3. Metodologija / Methodology — *target ~10–12 pp* (the technical core)
From the project's methodology document + the actual implementation.

- **3.1** Dataset & directory layout; per-challenge case counts; label semantics.
- **3.2** 5-fold **stratified** CV (volume quartile × ET-presence); reproducible splits.
- **3.3** Preprocessing: union brain mask, percentile clipping, z-score normalisation,
  foreground-centred crop (192×192×128), disk caching.
- **3.4** Augmentation: 128³ foreground-biased patch sampling (`RandCropByPosNegLabeld`),
  flips, rot90, intensity (scale/shift/noise).
- **3.5** Architectures: (i) 3D U-Net baseline; (ii) **Swin UNETR** (feature_size 48,
  gradient checkpointing). Overlapping-region (WT/TC/ET) prediction head.
- **3.6** Loss: compound **Dice + Focal**, region-wise.
- **3.7** Training protocol under a 12 GB constraint: **AMP**, AdamW, warmup + cosine LR,
  gradient clipping, batch size 1, ~300 epochs; single RTX 4070. *(Strong "reproducible on
  consumer hardware" narrative — include the engineering constraints honestly.)*
- **3.8** Inference & evaluation: **sliding-window** inference (128³, Gaussian, 0.5
  overlap); voxel DSC/HD95 **and** (planned) lesion-wise + full-volume evaluation;
  post-processing (connected-component / ET-suppression).
- **3.9** Experimental design: ablations (E0–E14) and the **GLI→PED transfer** experiment.
- **3.10** Reproducibility: fixed seeds, committed split JSONs, config logging.

### 4. Rezultati / Results — *target ~8–10 pp*
Report only what exists; clearly mark planned items. **Status legend: ✅ done · ⏳ pending.**

- **4.1** Training behaviour ✅ (loss curves; PED→0.47, GLI→0.16).
- **4.2** GLI fold-0 (Swin UNETR) ✅ — DSC WT 0.933 / TC 0.923 / ET 0.884; HD95 7.2/4.6/3.6.
- **4.3** PED fold-0 (Swin UNETR) ✅ — DSC WT 0.808 / TC 0.764 / ET 0.496; high variance.
- **4.4** MEN ⏳ — not yet trained (scope decision — see *Open scoping note* below).
- **4.5** Cross-challenge comparison ✅/⏳ — the **ET gap** (GLI 0.88 vs PED 0.50).
- **4.6** Ablation study ⏳ — components' marginal contribution.
- **4.7** Architecture comparison (3D U-Net vs Swin UNETR) ⏳.
- **4.8** Transfer learning (GLI→PED) ⏳.

### 5. Diskusija / Discussion — *target ~5–6 pp*

- **5.1** Why the populations differ — biology-driven interpretation of the ET gap.
- **5.2** PED ET difficulty: DMG / near-absent ET; the 11 ET-absent PED cases.
- **5.3** Failure-mode analysis (e.g. catastrophic-miss subjects, false-positive ET).
- **5.4** Comparison with the state of the art — **with the honest caveat**: our DSC/HD95
  are voxel-wise on cropped volumes and a held-out *training* fold, **not** the official
  lesion-wise full-volume test metric, so they are **not** leaderboard-comparable. (This
  intellectual honesty strengthens the thesis.)
- **5.5** Clinical relevance of each error type.
- **5.6** Limitations: single fold reported, no ensemble/TTA yet, eval protocol, 12 GB
  hardware ceiling, MEN scope.

### 6. Zaključna razmatranja / Conclusion — *target ~1 p*
Summary of findings + recommendations for further research. **No references, no new data.**
½–1 page per template.

---

## Back matter

- **Literatura / References** — Harvard/APA, alphabetical. Seed set: Ronneberger 2015
  (U-Net); Çiçek 2016 (3D U-Net); Isensee 2021 (nnU-Net); Hatamizadeh 2022 (Swin UNETR);
  Liu 2021 (Swin); Dosovitskiy 2021 (ViT); Menze 2015 + Bakas 2017 (BraTS); BraTS 2023
  challenge papers (GLI/MEN/PED winners); Cardoso 2022 (MONAI); Loshchilov 2019 (AdamW).
  *(I'll compile full entries + footnotes as we write each chapter.)*
- **Appendix A** — hyperparameter tables per challenge.
- **Appendix B** — code & reproducibility (repository structure, split JSONs).

---

## Page budget (keeps us under 40)

| Chapter | Pages |
|---|---|
| 1 Introduction | 3–4 |
| 2 Background | 6–8 |
| 3 Methodology | 10–12 |
| 4 Results | 8–10 |
| 5 Discussion | 5–6 |
| 6 Conclusion | 1 |
| **Total** | **~33–41** |

---

## Figures & tables to produce (placeholders to fill as results land)

- Fig. 1 — Example multi-modal MRI slice with WT/TC/ET overlay.
- Fig. 2 — Pipeline diagram (preprocess → patch → model → sliding-window → post-process).
- Fig. 3 — Swin UNETR architecture schematic.
- Fig. 4 — Training-loss curves (PED, GLI). *(have data)*
- Fig. 5 — Per-region DSC box plots, GLI vs PED. *(have data)*
- Fig. 6 — Qualitative predictions (best + failure cases, e.g. PED-00051). *(can generate)*
- Table 1 — Dataset summary (counts, ET-absent cases per challenge).
- Table 2 — Hyperparameters per challenge.
- Table 3 — Main results (DSC/HD95 per region, per challenge).
- Table 4 — Ablation results. *(pending)*

---

## Scope decision (provisional — confirm at Chapter 4)

**Direction (a): keep all three — GLI, MEN, PED.** GLI + PED are trained; **MEN training is
to be slotted in** (≈1,000 cases, ~1–2 days on the 4070). Target completion of training +
writing: **mid-July 2026**. Final confirmation deferred to Chapter 4, but Chapters 1–3 are
written for the full three-way framing.

---

## Writing order (selected: top-down)

1. **Ch. 1 Introduction** ← *drafting now* (`thesis/01_introduction.md`).
2. Ch. 2 Background (citation gathering in parallel).
3. Ch. 3 Methodology — enriched with hypotheses linkage, experimental design, social-impact
   framing (per candidate's note).
4. Ch. 4 Results → Ch. 5 Discussion (after MEN training + the lesion-wise evaluator).
5. Abstract + Ch. 6 Conclusion + References pass → convert to Singidunum `.docx`.
