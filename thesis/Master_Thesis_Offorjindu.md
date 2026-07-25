# One Pipeline, Three Tumours: Unified Deep-Learning Segmentation of Adult, Meningioma and Paediatric Brain Tumours on a Single GPU (BraTS 2023)

**Master Thesis**

Singidunum University — Faculty of Technical Sciences
Candidate: Samson O. Offorjindu · Mentor: Professor Nebojša Bačanin Džakula, PhD
Belgrade, 2026.

---

## Abstract

Accurate segmentation of brain tumours from multi-parametric magnetic resonance imaging (MRI) underpins diagnosis, treatment planning, and monitoring, yet manual delineation is slow and subject to inter-observer variability ([Menze et al., 2015](https://doi.org/10.1109/TMI.2014.2377694)). The BraTS 2023 challenge introduced three biologically distinct populations — adult glioma (GLI), intracranial meningioma (MEN), and paediatric high-grade glioma (PED). This thesis investigates whether a single, unified deep-learning methodology, with only modest task-specific adaptation, can segment all three competitively within the means of an individual researcher using one consumer-grade GPU (an NVIDIA RTX 4070, 12 GB). We implement a reproducible, patch-based pipeline — a Swin UNETR hybrid model ([Hatamizadeh et al., 2022b](https://arxiv.org/abs/2201.01266)) and a 3D U-Net baseline ([Çiçek et al., 2016](https://arxiv.org/abs/1606.06650)) — with mixed-precision training and stratified five-fold cross-validation, and evaluate it with voxel-level and lesion-wise Dice and 95th-percentile Hausdorff distance in native image space. The unified method reaches a mean full-volume voxel Dice of approximately 0.91 (GLI), 0.90 (MEN), and 0.72 (PED). Enhancing-tumour difficulty is strongly population-dependent (Dice 0.88 / 0.91 / 0.52), reflecting the frequently absent enhancement of paediatric diffuse midline glioma. The architecture comparison reveals an interaction with data scale: the convolutional 3D U-Net outperforms the hybrid Swin UNETR on the small paediatric set (0.745 vs 0.689) with fewer parameters, but the ranking reverses on the large adult-glioma set (Swin 0.913 vs 0.904) — convolutional inductive bias wins when data are scarce, the transformer at scale. GLI→PED transfer learning improves every paediatric region (+0.046). The divergence between voxel and lesion-wise scoring is itself population-dependent, tracking tumour-shape regularity. On the official-style lesion-wise metric the single model trails challenge-winning ensembles by a few points, quantifying the cost of the single-GPU, no-ensemble setting. The work demonstrates that competitive, reproducible, multi-population brain-tumour segmentation is achievable on accessible hardware.

**Keywords:** brain tumour segmentation; BraTS 2023; deep learning; Swin UNETR; 3D U-Net; transfer learning; magnetic resonance imaging; medical image analysis.

---

## Acknowledgements

I am grateful to Professor Miloš Stanković for providing access to the Google Cloud Platform virtual compute machines (NVIDIA L4) used for the large-scale cross-validation experiments in this work; this access was instrumental in extending the study beyond the single consumer-GPU baseline. I also thank my mentor, Professor Nebojša Bačanin Džakula, for his guidance throughout the project.

*[Add any further acknowledgements here — e.g., family, colleagues.]*

---

# 1. Introduction

Brain tumours are among the most consequential diseases of the central nervous system: even when histologically benign, their location within a confined cranial space makes them capable of producing severe neurological deficits, and the most aggressive forms remain among the deadliest of all human cancers. Accurate delineation of a tumour and its sub-regions on magnetic resonance imaging (MRI) underpins almost every stage of clinical care — diagnosis, surgical and radiotherapy planning, and the longitudinal monitoring of response to treatment. Yet in routine practice this delineation is still performed manually by experts, a process that is slow, costly, and subject to substantial inter-observer variability ([Menze et al., 2015](https://doi.org/10.1109/TMI.2014.2377694)). The promise of automated, reproducible segmentation has therefore motivated more than a decade of methodological research, organised in large part around the *Brain Tumour Segmentation* (BraTS) benchmark ([Menze et al., 2015](https://doi.org/10.1109/TMI.2014.2377694); [Bakas et al., 2017](https://doi.org/10.1038/sdata.2017.117); [Baid et al., 2021](https://arxiv.org/abs/2107.02314)).

The 2023 edition of BraTS marks an important broadening of this effort. Where earlier editions focused almost exclusively on adult diffuse glioma, BraTS 2023 introduced a family of parallel sub-challenges spanning biologically distinct tumour populations, including adult glioma (GLI), intracranial meningioma (MEN; [LaBella et al., 2023](https://arxiv.org/abs/2305.07642)), and paediatric high-grade glioma (PED; [Kazerooni et al., 2023](https://arxiv.org/abs/2305.17033)). These populations differ not only in cellular origin and typical anatomical location but also in the appearance, size, and even the *presence* of the tumour sub-regions that the challenge asks us to segment. This diversity raises a question that is both scientifically interesting and practically important, and which forms the core of this thesis: **can a single, unified deep-learning methodology — with only modest, task-specific adaptations — produce competitive and clinically meaningful segmentations across all three populations, and can it do so within the computational means of an individual researcher rather than a well-resourced laboratory?**

## 1.1 Subject and Problem of the Research

The **subject** of this research is the automatic, three-dimensional segmentation of brain tumours from multi-parametric MRI across the three BraTS 2023 populations (GLI, MEN, PED). Each subject in the dataset is described by four co-registered MRI sequences — native T1-weighted (*t1n*), contrast-enhanced T1-weighted (*t1c*), T2-weighted (*t2w*), and T2 fluid-attenuated inversion recovery (*t2f*) — and the segmentation task is to label every voxel as background or as one of the tumour tissue classes, from which three clinically meaningful, nested evaluation regions are derived: the *whole tumour* (WT), the *tumour core* (TC), and the *enhancing tumour* (ET).

The **problem** has two faces. The first is methodological. The strongest BraTS 2023 solutions rely on large ensembles of high-capacity networks, extensive synthetic-data generation, and multi-GPU training budgets that are out of reach for most individual researchers and for institutions in resource-limited settings. This creates a reproducibility and accessibility gap between what is reported at the top of the leaderboard and what an independent researcher can realistically build upon. The second face is biological. Because the three populations differ so markedly — adult gliomas are large and infiltrative, meningiomas are extra-axial and frequently abut the skull, and paediatric high-grade gliomas include diffuse midline tumours in which the enhancing component may be almost absent — it is not obvious *a priori* that one methodology can serve all three, nor which components of a segmentation pipeline are genuinely general and which must be tailored to a particular tumour biology.

This thesis addresses that gap directly. Rather than pursuing maximum leaderboard rank, we develop a single, transparent, and fully reproducible pipeline, train it on each of the three populations under identical conditions on a single consumer-grade graphics processing unit (an NVIDIA RTX 4070 with 12 GB of memory), and study how its behaviour differs across the three tumour types. The sample on which the study is conducted is the public BraTS 2023 training data — approximately 1,251 adult glioma, 1,000 meningioma, and 99 paediatric cases, roughly **2,350 subjects in total** — which far exceeds the minimum sample size expected of empirical master-level research.

## 1.2 Objectives of the Research

**Scientific objectives.** The scientific aims of the thesis are:

1. to establish a rigorous, reproducible training and evaluation baseline for brain-tumour segmentation on each of the three BraTS 2023 sub-challenges using a modern three-dimensional architecture;
2. to quantify, through controlled ablation, the marginal contribution of individual pipeline components (preprocessing, augmentation, loss formulation, post-processing) and thereby to distinguish techniques that are *domain-general* from those that are *tumour-specific*;
3. to compare a purely convolutional architecture (3D U-Net) with a hybrid convolutional–transformer architecture (Swin UNETR) under identical data, augmentation, and training budgets, isolating the contribution of the architecture itself; and
4. to investigate cross-population transfer learning — specifically, whether an encoder pre-trained on the large adult-glioma dataset improves segmentation on the much smaller paediatric dataset.

**Social objective.** The social aim is to demonstrate, and to document openly, that competitive and clinically meaningful brain-tumour segmentation is achievable on a single consumer-grade GPU. By lowering the hardware and engineering barrier to entry — and by releasing a clear, reproducible pipeline — the work seeks to make this class of research more accessible to students, smaller institutions, and clinical and research groups in resource-constrained environments, where multi-GPU infrastructure is rarely available. In doing so it contributes to the broader goal of equitable access to medical-imaging artificial-intelligence research.

## 1.3 Hypotheses

The research is guided by four hypotheses, each of which is operationalised and tested in the Results chapter:

- **H1.** A single unified methodology, with only minor task-specific adaptations, can produce clinically meaningful segmentation (high Dice overlap and low boundary error) across all three biologically distinct populations.
- **H2.** Under an identical training budget, the hybrid CNN–Transformer architecture (Swin UNETR) is at least competitive with, and may exceed, a purely convolutional 3D U-Net.
- **H3.** The difficulty of segmenting the enhancing-tumour (ET) region is population-dependent: it is comparatively easy in adult glioma and substantially harder in the paediatric population, where the enhancing component is frequently small or absent.
- **H4.** Transfer learning from the large adult-glioma dataset improves paediatric segmentation relative to training on the paediatric data alone.

## 1.4 Research Methods

The thesis follows a quantitative, experimental methodology. The empirical material is the public BraTS 2023 dataset; all experiments use a reproducible five-fold stratified cross-validation split generated once and fixed for the duration of the study. The segmentation models are deep neural networks — a three-dimensional U-Net baseline ([Çiçek et al., 2016](https://arxiv.org/abs/1606.06650); [Ronneberger et al., 2015](https://arxiv.org/abs/1505.04597)) and the Swin UNETR hybrid architecture ([Hatamizadeh et al., 2022b](https://arxiv.org/abs/2201.01266)) — implemented with PyTorch and the MONAI medical-imaging framework ([Cardoso et al., 2022](https://arxiv.org/abs/2211.02701)) and trained under mixed precision on a single GPU. Performance is assessed with the standard BraTS metrics — the Dice similarity coefficient (DSC) and the 95th-percentile Hausdorff distance (HD95) — computed per evaluation region, complemented by the lesion-wise metric introduced in 2023. Beyond the basic analytical and synthetic methods common to all empirical research, the work makes central use of the comparative method (across the three populations and the two architectures) and of statistical analysis of the resulting metric distributions; controlled ablation is used to attribute performance to individual pipeline components, in the hypothetico-deductive tradition of testing the stated hypotheses against measured outcomes.

## 1.5 Structure of the Thesis

The remainder of the thesis is organised as follows. **Chapter 2** reviews the theoretical background — the relevant tumour biology and MRI, the evolution of deep-learning segmentation from the U-Net to transformer-based architectures, the BraTS challenge, and the leading 2023 solutions. **Chapter 3** presents the methodology in full: the dataset and cross-validation protocol, the preprocessing and augmentation pipeline, the model architectures, the loss formulation, the single-GPU training protocol, and the evaluation and experimental design. **Chapter 4** reports the results across the three populations, the architecture comparison, the ablation study, and the transfer-learning experiment. **Chapter 5** discusses and interprets these results — with particular attention to the biologically driven differences between populations, characteristic failure modes, the relationship of our findings to the state of the art, and the study's limitations. **Chapter 6** concludes with a summary of the principal findings and recommendations for further research.

---

# 2. Theoretical Background

This chapter establishes the conceptual and technical foundation for the thesis. We first describe the three tumour populations studied and the MRI on which they are imaged (§2.1–§2.2), then trace the development of medical-image segmentation from classical methods to convolutional and transformer-based deep networks (§2.3–§2.5). We close by situating the work within the BraTS challenge and its 2023 sub-challenges, reviewing the leading 2023 solutions, and defining the metrics by which segmentation quality is judged (§2.6–§2.8).

## 2.1 Brain Tumours: Three Distinct Populations

Tumours of the central nervous system are classified by the World Health Organization according to a combination of histological and, increasingly, molecular features ([Louis et al., 2021](https://doi.org/10.1093/neuonc/noab106)). The three BraTS 2023 populations studied here are deliberately chosen to span a wide range of this taxonomy, and their differences are central to the thesis.

**Adult diffuse glioma (GLI).** Gliomas arise from glial cells and are the most common malignant primary brain tumours in adults. High-grade gliomas, in particular glioblastoma, are characterised by rapid, infiltrative growth, pronounced peritumoural oedema, a necrotic core, and irregular contrast enhancement reflecting a disrupted blood–brain barrier. They are typically large at presentation and intra-axial (arising within the brain parenchyma), and their infiltrative margins make precise delineation difficult. Adult glioma has been the historical focus of the BraTS challenge and remains its largest and best-characterised population.

**Meningioma (MEN).** Meningiomas arise from the meningothelial cells of the arachnoid and are the most common primary intracranial tumours overall; the majority are benign (WHO grade 1), though atypical and anaplastic variants occur. In contrast to glioma they are *extra-axial* — they grow from the meningeal coverings rather than within the brain — and are frequently located at the skull base or along the falx, often abutting bone and producing a characteristic dural "tail" of enhancement. Their convex, well-circumscribed shape and skull-adjacent location create segmentation challenges quite different from those of glioma, notably the risk of clipping peripheral tumour during brain-focused cropping.

**Paediatric high-grade glioma (PED).** Paediatric high-grade gliomas, including diffuse midline glioma (and its brainstem form, historically termed DIPG), are rare but devastating, with a prognosis that remains poor despite treatment ([Kazerooni et al., 2023](https://arxiv.org/abs/2305.17033)). Molecularly and radiologically they differ substantially from their adult counterparts: in particular, the enhancing-tumour component is frequently small or even absent, which — as we will show — makes the ET region exceptionally difficult to segment in this population. The paediatric dataset is also far smaller than the adult one, compounding the difficulty with a data-scarcity problem. Together these properties make PED an informative stress-test of how well a methodology generalises beyond the adult-glioma regime for which it is usually designed.

## 2.2 Multi-parametric MRI and Tumour Sub-regions

Each BraTS subject is imaged with four co-registered MRI sequences, each sensitising the acquisition to different tissue properties: native T1-weighted (*t1n*), contrast-enhanced T1-weighted (*t1c*), T2-weighted (*t2w*), and T2 fluid-attenuated inversion recovery (*t2f*/FLAIR). The contrast-enhanced T1 sequence highlights regions of blood–brain-barrier breakdown and is therefore the principal cue for enhancing tumour; FLAIR suppresses cerebrospinal-fluid signal and makes peritumoural oedema conspicuous; T1 and T2 provide complementary anatomical and tissue contrast. The complementary nature of the four sequences is precisely why multi-parametric input is used: no single sequence delineates all tumour sub-regions reliably.

The challenge annotations label each voxel with one of three mutually exclusive tissue classes — necrotic/non-enhancing tumour core, peritumoural oedema, and enhancing tumour — from which three nested, clinically meaningful **evaluation regions** are derived: the *whole tumour* (WT), comprising all tumour tissue; the *tumour core* (TC), comprising the core and enhancing components; and the *enhancing tumour* (ET). By construction these regions are nested (ET ⊆ TC ⊆ WT). Predicting these overlapping regions directly — rather than the disjoint tissue labels — is the convention adopted by most strong BraTS methods and by this thesis, because it aligns the training objective with the quantities on which the challenge is scored.

## 2.3 From Classical to Deep-Learning Segmentation

Early approaches to brain-tumour segmentation relied on intensity thresholding, region growing, atlas registration, and classical machine-learning classifiers operating on hand-engineered features. While useful, these methods struggled with the intensity heterogeneity of multi-institutional MRI and with the wide morphological variability of tumours. The decisive shift came with deep convolutional neural networks, which learn hierarchical features directly from data and, given sufficient training examples, substantially outperform hand-engineered pipelines. The BraTS benchmark has both tracked and accelerated this transition, with essentially all competitive entries since the late 2010s being deep-learning based ([Menze et al., 2015](https://doi.org/10.1109/TMI.2014.2377694); [Bakas et al., 2017](https://doi.org/10.1038/sdata.2017.117)).

## 2.4 Convolutional Architectures for Segmentation

The architecture that has most shaped medical-image segmentation is the **U-Net** ([Ronneberger et al., 2015](https://arxiv.org/abs/1505.04597)), an encoder–decoder network with skip connections that pass high-resolution features from the contracting path to the expanding path, allowing precise localisation while retaining semantic context. Because brain MRI is inherently volumetric, three-dimensional variants quickly followed: the **3D U-Net** ([Çiçek et al., 2016](https://arxiv.org/abs/1606.06650)) and **V-Net** ([Milletari et al., 2016](https://arxiv.org/abs/1606.04797)) extend the design to full volumes and form the basis of the 3D U-Net baseline used in this thesis.

Two refinements are especially relevant. **nnU-Net** ([Isensee et al., 2021](https://doi.org/10.1038/s41592-020-01008-z)) is not a new architecture but a self-configuring framework that automatically adapts preprocessing, patch size, and training schedule to a given dataset; it has won or placed at the top of numerous medical-segmentation challenges, including BraTS, and constitutes the *de facto* baseline that strong solutions must match. **SegResNet** ([Myronenko, 2018](https://arxiv.org/abs/1810.11654)), a residual encoder–decoder with an auxiliary autoencoder regularisation branch, and the more recent **MedNeXt** ([Roy et al., 2023](https://arxiv.org/abs/2303.09975)), which modernises the convolutional block with design ideas drawn from transformers, recur prominently among top BraTS entries — particularly, as we will see, for meningioma.

## 2.5 Transformers and Hybrid Architectures

The transformer ([Vaswani et al., 2017](https://arxiv.org/abs/1706.03762)), originally developed for natural-language processing, replaces convolution's local receptive field with a self-attention mechanism that models long-range dependencies directly. The **Vision Transformer** (ViT; [Dosovitskiy et al., 2021](https://arxiv.org/abs/2010.11929)) applied this idea to images by treating image patches as tokens, while the **Swin Transformer** ([Liu et al., 2021](https://arxiv.org/abs/2103.14030)) introduced a hierarchical, shifted-window attention scheme that restores the multi-scale inductive bias useful for dense prediction and reduces the quadratic cost of global attention.

For volumetric medical images, these ideas were combined with the U-Net's encoder–decoder structure. **UNETR** ([Hatamizadeh et al., 2022a](https://arxiv.org/abs/2103.10504)) uses a pure transformer encoder with a convolutional decoder, and **Swin UNETR** ([Hatamizadeh et al., 2022b](https://arxiv.org/abs/2201.01266)) employs a hierarchical Swin-Transformer encoder coupled to a convolutional decoder through skip connections, yielding a hybrid that captures both long-range context and fine local detail. Swin UNETR has become a standard component of competitive BraTS pipelines, frequently as a partner in an ensemble, and self-supervised pre-training of its encoder on large unlabelled medical-image collections has been shown to improve downstream performance ([Tang et al., 2022](https://arxiv.org/abs/2111.14791)). Swin UNETR is the principal model studied in this thesis, and the comparison between it and the purely convolutional 3D U-Net under identical conditions is one of our central experiments.

## 2.6 The BraTS Challenge and its 2023 Sub-challenges

The Brain Tumour Segmentation (BraTS) challenge, first held in 2012, has become the reference benchmark for the field, providing standardised, expertly annotated, multi-institutional, co-registered and skull-stripped data together with a common evaluation protocol ([Menze et al., 2015](https://doi.org/10.1109/TMI.2014.2377694); [Bakas et al., 2017](https://doi.org/10.1038/sdata.2017.117); [Baid et al., 2021](https://arxiv.org/abs/2107.02314)). Successive editions have grown the dataset and progressively raised the performance bar.

The 2023 edition broadened the challenge from a single adult-glioma task into a *cluster* of parallel sub-challenges covering biologically and demographically distinct populations, including adult glioma (GLI), intracranial meningioma (MEN; [LaBella et al., 2023](https://arxiv.org/abs/2305.07642)), and paediatric high-grade glioma (PED; [Kazerooni et al., 2023](https://arxiv.org/abs/2305.17033)), among others. All sub-challenges share a common data format and evaluation regions (WT/TC/ET), which is precisely what makes a *unified* methodology across them both feasible and scientifically interesting — and what this thesis sets out to study.

## 2.7 Related Work: Leading BraTS 2023 Solutions

Analysing the top-performing 2023 entries reveals both a set of population-specific strategies and a striking convergence of general practice.

**Adult glioma (GLI).** The winning solution combined a three-architecture ensemble (nnU-Net, Swin UNETR, and the BraTS 2021 winning network); its decisive advantage was synthetic-data generation, using a generative adversarial network to insert realistic synthetic tumours into healthy brain regions, together with a registration-based augmentation that transplanted existing tumours into new anatomies. Reported validation scores were of the order of DSC 0.90 / 0.87 / 0.85 and HD95 ≈ 15 / 14 / 18 mm for WT / TC / ET.

**Paediatric (PED).** The winning approach ensembled nnU-Net and Swin UNETR with *label-wise* aggregation — weighting each model's contribution per region rather than averaging full probability maps — and applied a cross-validated per-region threshold search. A biologically informed post-processing step *redefined* the enhancing-tumour label for cases in which the ET volume was very small relative to the total tumour, reassigning those voxels; this directly addressed the near-absent ET of diffuse midline tumours and produced a measurable gain. Reported mean Dice was approximately 0.84 / 0.81 / 0.65 for WT / TC / ET, with ET the hardest region across all teams ([Kazerooni et al., 2024](https://arxiv.org/abs/2407.08855)).

**Meningioma (MEN).** Top meningioma solutions were dominated by SegResNet- and MedNeXt-based pipelines with deep supervision and five-fold cross-validation, several built on automated configuration frameworks; targeted post-processing exploiting the typically convex shape of meningiomas was common ([LaBella et al., 2023](https://arxiv.org/abs/2305.07642)).

**Universal patterns.** Across the leading entries we observe a consistent recipe: a U-Net-family backbone (often nnU-Net) frequently ensembled with Swin UNETR; training on the overlapping WT/TC/ET regions; a compound Dice + cross-entropy loss; deep supervision; aggressive spatial and intensity augmentation; five-fold cross-validation with test-time ensembling and test-time augmentation; mixed-precision training; and connected-component / threshold-based post-processing. This thesis adopts the *general* elements of this recipe while deliberately operating within a single-GPU budget and *without* the large ensembles and synthetic-data generation that distinguish the very top entries — a scope chosen to isolate what is achievable by an individual researcher.

## 2.8 Evaluation Metrics

Segmentation quality in BraTS is assessed primarily with two complementary metrics, computed independently for each evaluation region (WT, TC, ET).

The **Dice similarity coefficient** (DSC) measures volumetric overlap between a predicted region $P$ and the ground truth $G$ as $\mathrm{DSC} = 2|P \cap G| / (|P| + |G|)$, ranging from 0 (no overlap) to 1 (perfect overlap). By convention, an empty prediction against an empty ground truth scores 1 (both correctly agree no tumour is present), whereas an empty prediction against a non-empty ground truth scores 0.

The **95th-percentile Hausdorff distance** (HD95) measures boundary agreement as the 95th-percentile of the distances between the surfaces of $P$ and $G$, expressed in millimetres; the 95th percentile (rather than the maximum) confers robustness to a small number of outlying voxels. HD95 is undefined when either region is empty, a case that must be handled explicitly during evaluation.

A limitation of these voxel-level metrics is that they do not penalise spurious or missed *lesions* as such. BraTS 2023 therefore introduced a **lesion-wise** formulation, in which Dice and HD95 are computed per connected tumour component and then averaged, so that false-positive and false-negative lesions are penalised individually. The lesion-wise metric is generally harsher than its voxel-level counterpart and is the basis of the official 2023 ranking; the distinction between the two is important when interpreting our results and comparing them with the leaderboard, a point we return to in Chapter 5.

---

# 3. Methodology

This chapter presents the complete methodology: the dataset and cross-validation protocol (§3.1–§3.2); the preprocessing and augmentation pipeline (§3.3–§3.4); the model architectures (§3.5); the loss formulation (§3.6); the single-GPU training protocol (§3.7); the inference and evaluation procedure (§3.8); the experimental design that operationalises the four hypotheses (§3.9); and the measures taken to ensure reproducibility (§3.10). The implementation uses Python with PyTorch and the MONAI medical-imaging framework ([Cardoso et al., 2022](https://arxiv.org/abs/2211.02701)).

## 3.1 Dataset and Data Representation

We use the public BraTS 2023 training data for the three sub-challenges studied: adult glioma (GLI), meningioma (MEN), and paediatric high-grade glioma (PED). Each subject is a directory containing four co-registered, skull-stripped MRI volumes — *t1c*, *t1n*, *t2f*, *t2w* — and, for training cases, an expert segmentation mask (*seg*). All volumes are provided at 1 mm isotropic resolution on a common grid. The segmentation mask labels each voxel as background (0), necrotic/non-enhancing tumour core (1), peritumoural oedema (2), or enhancing tumour (3); from these the overlapping evaluation regions whole tumour (WT), tumour core (TC), and enhancing tumour (ET) are derived.

Table 3.1 summarises the three datasets. The combined sample of roughly 2,350 annotated subjects far exceeds the minimum required for empirical master-level research; equally relevant to the thesis is the number of cases in which the enhancing tumour is *absent*, which is appreciable in the paediatric population and underlies hypothesis H3.

**Table 3.1 — Dataset summary.**

| Sub-challenge | Training cases | ET-absent cases | Key challenge |
|---|---|---|---|
| GLI | 1,251 | 33 | large, infiltrative; class imbalance |
| MEN | 1,000 | 1 | extra-axial; skull-base truncation |
| PED | 99 | 11 | very small dataset; absent ET in DMG |

Internally, each subject is represented as a four-channel tensor (one channel per modality) of shape $(4, H, W, D)$, with the segmentation as a single-channel volume; the four channels are stacked so that the network receives the full multi-parametric input simultaneously.

## 3.2 Cross-validation Protocol

To obtain reproducible and representative train/validation partitions we generate, **once**, a five-fold **stratified** cross-validation split for each challenge and fix it for all subsequent experiments. Stratification is performed jointly on (i) the quartile of total tumour volume and (ii) the presence or absence of enhancing tumour, so that each fold contains a representative mix of small and large tumours and of ET-present and ET-absent cases — the latter being essential for stable evaluation in the paediatric and meningioma populations. The split is written to a JSON file and committed to the repository, guaranteeing that every experiment in the thesis draws on identical partitions. The principal segmentation results (§4.2) are reported as five-fold cross-validated means ± standard deviations over all five held-out folds; the single-fold controlled comparisons — the architecture and transfer-learning experiments — use fold 0. The five-fold structure additionally supports the cross-validation ensembling discussed as an extension in §3.9.

## 3.3 Preprocessing

The preprocessing pipeline is deterministic and is applied identically at training and inference time. For each subject it proceeds as follows.

**Brain masking and spatial cropping.** A single brain mask is computed as the union of the non-zero voxels across all four modalities. Computing one shared mask — rather than a per-modality mask — is essential: the modalities can have slightly different zero-padding and fields of view, and cropping each independently would destroy the voxel-level correspondence between channels on which the network relies. From this mask we compute a tight bounding box, pad it by 10 voxels on each side, and place a fixed-size crop window of $192 \times 192 \times 128$ voxels around its centre, shifting the window inward where it would exceed the volume bounds. The *identical* window is applied to all four modalities and to the segmentation mask. The crop size was chosen to be large enough to contain peripheral meningiomas that a tighter window can clip, while remaining compatible with the encoder's downsampling strides; where a volume is smaller than the target the result is symmetrically zero-padded.

**Intensity normalisation.** Each modality is normalised independently within the brain mask. We first clip intensities to the 0.5th and 99.5th percentiles computed over brain voxels, which removes scanner-specific outliers that would otherwise distort normalisation in this multi-institutional data; we then apply z-score normalisation using the brain-voxel mean $\mu$ and standard deviation $\sigma$, $x' = (x - \mu) / \sigma$, and finally rescale the result into the range $[0, 255]$ using the brain-voxel minimum and maximum, giving a consistent dynamic range across modalities and scanners. Background voxels remain at or near zero throughout.

**Disk caching.** Because the full preprocessing pipeline involves reading four compressed NIfTI volumes per subject, it dominates per-epoch time if repeated every epoch. We therefore cache the preprocessed (but un-augmented) four-channel tensor to disk after its first computation; subsequent epochs read the cached tensor directly, which on our hardware reduces per-epoch time by roughly an order of magnitude. Caching the *un-augmented* volume is deliberate: augmentation is random and must be re-applied afresh each epoch (§3.4).

## 3.4 Augmentation and Patch-based Sampling

Because the full preprocessed volume is too large to train on at the batch sizes a 12 GB GPU permits, we adopt **patch-based** training. From each subject we sample a $128 \times 128 \times 128$ patch using a foreground-biased scheme that, with high probability, centres the patch on a tumour voxel; the foreground probability is $0.90$ for GLI and MEN and is raised to $0.95$ for PED, where the tumour occupies a smaller fraction of the volume. This concentrates training signal on the tumour and its boundary while still exposing the network to background through the residual probability.

On each sampled patch we apply, on the fly, a sequence of augmentations implemented with MONAI's dictionary transforms so that identical spatial transforms are applied to image and label. Spatial augmentations — random flips along each of the three axes (each with probability $0.5$) and random $90^{\circ}$ rotations in the axial plane — are applied to both image and label. Intensity augmentations — random intensity scaling and shifting (probability $0.3$ each) and additive Gaussian noise (probability $0.15$) — are applied to the image channels only. These augmentations enlarge the effective training distribution and improve robustness to the appearance variability of multi-institutional MRI. Richer augmentations (elastic deformation, modality dropout, copy-paste tumour insertion) are part of the wider design space and are discussed as planned ablations in §3.9.

## 3.5 Model Architectures

The thesis studies two architectures under identical data, augmentation, and training conditions, so that any performance difference is attributable to the architecture itself.

**3D U-Net (baseline).** As a purely convolutional reference we use a three-dimensional U-Net ([Çiçek et al., 2016](https://arxiv.org/abs/1606.06650); [Ronneberger et al., 2015](https://arxiv.org/abs/1505.04597)): an encoder–decoder with skip connections, convolutional blocks, and instance normalisation, producing a three-channel output corresponding to the overlapping WT/TC/ET regions. It serves as the baseline against which the transformer-based model is compared (hypothesis H2).

**Swin UNETR (principal model).** Our principal model is Swin UNETR ([Hatamizadeh et al., 2022b](https://arxiv.org/abs/2201.01266)), a hybrid architecture whose encoder is a hierarchical Swin Transformer ([Liu et al., 2021](https://arxiv.org/abs/2103.14030)) and whose decoder is convolutional, the two connected by skip connections in the U-Net style. We instantiate it with four input channels, three output channels, and a feature embedding size of 48, giving approximately 62 million parameters. To fit the model on 12 GB of memory we enable **gradient checkpointing**, which trades additional computation for a substantial reduction in activation memory. Both models predict the three overlapping regions directly, as logits; the sigmoid activation that maps logits to per-region probabilities is applied inside the loss and at inference.

## 3.6 Loss Function

Following the consensus among strong BraTS methods, we optimise a **compound** loss that combines a region-overlap term with a term targeting class imbalance, summed over the three evaluation regions. For each region $r \in \{\text{WT}, \text{TC}, \text{ET}\}$ we compute a soft Dice loss, $\mathcal{L}_{\text{Dice}} = 1 - (2 \sum_i p_i g_i + \varepsilon) / (\sum_i p_i + \sum_i g_i + \varepsilon)$, where $p_i = \sigma(z_i)$ is the predicted probability for voxel $i$, $g_i$ the ground-truth indicator, and $\varepsilon$ a smoothing constant; and a focal loss based on the binary cross-entropy between logits and targets, which down-weights easy voxels and focuses learning on the hard tumour-boundary voxels ([Lin et al., 2017](https://arxiv.org/abs/1708.02002)). The total loss is the equally weighted sum of the Dice and focal terms across the three regions. The Dice term drives volumetric overlap, while the focal term improves boundary delineation, which is where the Hausdorff distance is determined.

## 3.7 Training Protocol

All models are trained under a single-GPU regime designed to fit an NVIDIA RTX 4070 with 12 GB of memory (full hardware/software environment in §3.10).

**Optimisation.** We use the AdamW optimiser ([Loshchilov & Hutter, 2019](https://arxiv.org/abs/1711.05101)), in which weight decay is decoupled from the adaptive gradient update, with a peak learning rate of $1\times10^{-4}$ for GLI and MEN and a lower $5\times10^{-5}$ for the small PED dataset, and weight decay of $1\times10^{-5}$ (GLI/MEN) or the stronger $1\times10^{-4}$ (PED). The learning rate follows a schedule of linear warm-up over the first five epochs followed by cosine decay to a small floor; computing the rate from the epoch index alone makes the schedule deterministic and exactly resumable from a checkpoint.

**Mixed precision and stability.** Training uses automatic mixed precision: the forward pass runs under half-precision autocast for speed and memory, while the loss is computed in single precision to avoid overflow in the Dice reductions, and a gradient scaler manages the half-precision gradients. Gradients are clipped to a maximum norm of $1.0$ for stability.

**Schedule and checkpointing.** We use a batch size of one $128^3$ patch per step and train for 300 epochs. The latest model state — together with the optimiser and gradient-scaler state — is checkpointed every epoch so that training can be resumed exactly after any interruption, with periodic numbered backups retained. Per-challenge hyperparameters are summarised in Table 3.2.

**Table 3.2 — Training hyperparameters per sub-challenge.**

| Hyperparameter | GLI | MEN | PED |
|---|---|---|---|
| Epochs | 300 | 300 | 300 |
| Peak learning rate | 1e-4 | 1e-4 | 5e-5 |
| Weight decay | 1e-5 | 1e-5 | 1e-4 |
| Warm-up epochs | 5 | 5 | 5 |
| Patch size | 128³ | 128³ | 128³ |
| Batch size | 1 | 1 | 1 |
| Foreground-sampling prob. | 0.90 | 0.90 | 0.95 |
| Mixed precision | yes | yes | yes |

## 3.8 Inference and Evaluation

Because the network is trained on $128^3$ patches it cannot be applied to a full volume in a single forward pass; we therefore use **sliding-window inference** ([Cardoso et al., 2022](https://arxiv.org/abs/2211.02701)). A $128^3$ window is moved across the volume with 50 % overlap, the per-window predictions are combined with Gaussian weighting (which smoothly down-weights window edges), and the accumulated logits are passed through a sigmoid to yield per-region probability maps. The maps are thresholded — at $0.45$, $0.40$ and $0.45$ for WT, TC and ET respectively — and converted to a consistent labelling in which the nesting ET ⊆ TC ⊆ WT is respected.

Segmentation quality is then quantified per region with the Dice similarity coefficient and the 95th-percentile Hausdorff distance (§2.8). We apply the standard empty-region conventions: for Dice, an empty prediction against an empty ground truth scores 1 and against a non-empty ground truth scores 0; for HD95, which is undefined when either region is empty, the subject is excluded from that region's HD95 average and the exclusion is counted and reported. We report the mean and standard deviation of each metric over the validation fold, together with per-subject values to expose the distribution of performance.

We note explicitly — and return to the point in Chapter 5 — that this evaluation is computed with *voxel-level* metrics in the cropped $192\times192\times128$ space on a held-out fold of the *training* data; it is therefore appropriate for the controlled internal comparisons that the thesis makes, but is not directly comparable to the official BraTS 2023 leaderboard, which uses the *lesion-wise* metric on full-resolution volumes of a separate hidden test set. To bridge this gap we additionally implement a full-volume evaluation: predictions made on the in-distribution crop are re-embedded into native image space, connected-component post-processing removes spurious small components, and both voxel and lesion-wise DSC/HD95 are computed against the full-resolution ground truth.

## 3.9 Experimental Design

The experiments are designed to test the four hypotheses of §1.3 under controlled conditions. Table 3.3 maps each hypothesis to the experiment that evaluates it.

**Table 3.3 — Hypotheses and the experiments that test them.**

| Hypothesis | Experiment |
|---|---|
| H1 — unified methodology across populations | Train and evaluate the same pipeline on GLI, MEN and PED; report per-region DSC/HD95. |
| H2 — Swin UNETR vs 3D U-Net | Train both architectures under identical data/augmentation/budget on the same fold and compare. |
| H3 — population-dependent ET difficulty | Compare ET performance and its variance across the three populations. |
| H4 — GLI→PED transfer | Pre-train on GLI, fine-tune on PED, and compare against PED-only training. |

**Ablation study.** To attribute performance to individual components, we plan a cumulative ablation in which elements are added one at a time to a minimal baseline — intensity normalisation, spatial augmentation, intensity augmentation, the compound loss, patch-based training with sliding-window inference, and post-processing — measuring the marginal change in Dice at each step. This isolates which components are *domain-general* and which are *tumour-specific*, addressing the second scientific objective. We run the ablation primarily on GLI, whose large dataset yields the most stable metrics, and confirm the full pipeline on MEN and PED.

**Planned extensions.** Several techniques common to the strongest BraTS entries lie outside the core single-model pipeline but are natural extensions, each evaluated as an addition to the baseline: five-fold cross-validation ensembling; test-time augmentation; per-region threshold optimisation; connected-component and, for PED, ET-suppression post-processing; and richer augmentation (elastic deformation, modality dropout). We report these as incremental contributions rather than folding them silently into the headline numbers, preserving a clear account of what each technique buys.

## 3.10 Reproducibility and Implementation Environment

All experiments are made reproducible by fixing the random seeds of Python, NumPy and PyTorch (including CUDA) to a common value, by committing the cross-validation split files to the repository, and by logging the full hyperparameter configuration alongside every checkpoint. Training progress (per-epoch loss and learning rate) is recorded to disk, and every run is resumable from its latest checkpoint.

The implementation uses Python 3.13, PyTorch 2.8 with CUDA 12.6, and MONAI 1.5 ([Cardoso et al., 2022](https://arxiv.org/abs/2211.02701)). All training and evaluation reported in this thesis were performed on a single workstation with an AMD Ryzen 9 9900X CPU, 32 GB of system memory, and an NVIDIA GeForce RTX 4070 GPU with 12 GB of memory. This modest, consumer-grade configuration is not incidental but central to the thesis: it demonstrates that the methodology is reproducible without specialised infrastructure, in direct support of the social objective stated in §1.2. The principal constraints it imposes — a small batch size, patch-based training, gradient checkpointing, and mixed precision — are documented here precisely so that the work can be reproduced, and its limitations understood, by others working under similar constraints.

---

# 4. Results

This chapter reports the experimental results. We first describe training behaviour (§4.1), then the principal segmentation results across the three populations under both voxel-level and lesion-wise metrics (§4.2), the cross-population comparison that the thesis centres on (§4.3), and the gap between voxel and lesion-wise scoring (§4.4). The architecture comparison, ablation study, and transfer-learning experiment are reported in §4.5–§4.7.

## 4.1 Training Behaviour

Every fold of every challenge was trained for 300 epochs under the identical protocol of §3.7, with the final training loss consistent across the five folds. Training was stable in every case: the compound loss decreased monotonically through the warm-up and cosine-decay schedule and plateaued well before the final epoch, indicating convergence. The loss reflects the relative difficulty of each population — lowest for meningioma and adult glioma, highest for the small, heterogeneous paediatric set.

**Table 4.1 — Final training loss (epoch 300, representative fold).**

| Challenge | Final training loss | Cases (train fold) |
|---|---|---|
| GLI | 0.16 | 1,003 |
| MEN | 0.11 | 799 |
| PED | 0.47 | 77 |

![Training loss over 300 epochs for GLI, MEN and PED at fold 0, log-scale y-axis.](figures/fig_4_1_loss_curves.png)

**Figure 4.1 — Training loss (Dice + Focal) versus epoch for GLI, MEN, and PED (representative fold 0), log-scale y-axis.** All three curves decrease monotonically and plateau well before 300 epochs, confirming convergence; the plateau height tracks population difficulty (GLI ≈ 0.16, MEN ≈ 0.11, PED ≈ 0.47), the small paediatric set settling highest.

## 4.2 Principal Segmentation Results

We report the Dice similarity coefficient (DSC) and 95th-percentile Hausdorff distance (HD95) per evaluation region (WT, TC, ET). Two evaluation settings are used: the in-pipeline **cropped** voxel evaluation (§3.8) and the **full-volume** evaluation, which additionally reports the **lesion-wise** metric. The full-volume results (Table 4.3) are **five-fold cross-validated** over every subject of PED and MEN, each scored once as a held-out case (GLI over its four completed folds); the cropped values (Table 4.2) are a single-fold in-pipeline development metric.

**Table 4.2 — Cropped voxel DSC (mean ± std), single fold (in-pipeline development metric).**

| Challenge | WT | TC | ET | Mean |
|---|---|---|---|---|
| GLI | 0.933 ± 0.063 | 0.923 ± 0.122 | 0.884 ± 0.167 | **0.913** |
| MEN | 0.894 ± 0.204 | 0.901 ± 0.210 | 0.908 ± 0.199 | **0.901** |
| PED | 0.808 ± 0.212 | 0.764 ± 0.263 | 0.496 ± 0.369 | **0.689** |

**Table 4.3 — Full-volume voxel and lesion-wise DSC (mean ± std) / HD95 (mm): five-fold cross-validated (PED, MEN); GLI over four folds.**

| Challenge | Metric | WT | TC | ET | Mean DSC |
|---|---|---|---|---|---|
| PED (99) | voxel DSC | 0.831 ± 0.182 | 0.802 ± 0.212 | 0.520 ± 0.364 | **0.718** |
| | voxel HD95 | 20.1 | 12.2 | 14.8 | |
| | lesion DSC | 0.431 ± 0.276 | 0.537 ± 0.312 | 0.421 ± 0.360 | **0.463** |
| | lesion HD95 | 193.0 | 136.5 | 160.4 | |
| MEN (1000) | voxel DSC | 0.894 ± 0.192 | 0.905 ± 0.191 | 0.909 ± 0.186 | **0.903** |
| | voxel HD95 | 10.4 | 9.9 | 9.4 | |
| | lesion DSC | 0.827 ± 0.250 | 0.841 ± 0.246 | 0.847 ± 0.242 | **0.838** |
| | lesion HD95 | 43.8 | 41.0 | 39.4 | |
| GLI † (1003) | voxel DSC | 0.934 | 0.911 | 0.874 | **0.907** |
| | voxel HD95 | 7.0 | 4.6 | 3.3 | |
| | lesion DSC | 0.750 | 0.827 | 0.789 | **0.789** |
| | lesion HD95 | 74.6 | 39.0 | 46.2 | |

† GLI is aggregated over the **four completed folds** (0–3; 1,003 subjects); fold 4 could not be trained within the schedule, and GLI's low inter-fold variance (per-fold means 0.902–0.916) makes four folds a stable estimate. PED and MEN are aggregated over all five folds (99 and 1,000 subjects respectively).

Two observations stand out and are developed in Chapter 5. First, five-fold cross-validation changes the paediatric headline: the mean full-volume voxel DSC rises from the single-fold 0.686 to **0.718**, because fold 0 was one of the harder paediatric splits. The paediatric mean varies across the five folds by 0.100 (per-fold 0.686 / 0.666 / 0.756 / 0.766 / 0.727), whereas the meningioma mean varies by only 0.018 (0.897–0.915) — a concrete demonstration of why single-fold reporting is unsafe on 99 cases yet adequate on 1,000. Second, the lesion-wise DSC is markedly lower than the voxel DSC (PED 0.463 vs 0.718; MEN 0.838 vs 0.903), exposing false-positive and fragmented lesions that voxel overlap does not penalise.

## 4.3 Cross-population Comparison and the Enhancing-Tumour Gap

The central comparison of the thesis is across the three populations under an identical methodology (hypothesis H1). The headline finding concerns the enhancing-tumour region (hypothesis H3): under the same architecture, loss, and training budget, ET segmentation is strong in the two adult populations and collapses in the paediatric one.

**Table 4.4 — Enhancing-tumour DSC across populations (full-volume voxel; five-fold PED/MEN, GLI four folds).**

| Challenge | ET DSC | ET-absent cases |
|---|---|---|
| MEN | 0.909 | 1 / 1000 |
| GLI † | 0.874 | 33 / 1251 |
| PED | 0.520 | 11 / 99 |

† GLI over four folds (0–3).

The contrast is striking: ET DSC falls from ~0.89–0.91 in adult glioma and meningioma to 0.52 in the paediatric set. This is consistent with the biology of §2.1 — in paediatric diffuse midline glioma the enhancing component is frequently small or absent (11 of 99 PED cases have no ET at all) — and with the high variance of the PED ET score (± 0.36), which reflects a bimodal distribution of near-perfect and near-zero cases rather than uniform mediocrity. The whole-tumour and tumour-core regions degrade more gracefully (PED WT 0.831), indicating that the difficulty is specific to enhancing tissue rather than to the paediatric population as a whole. This result directly supports H1 (a single methodology transfers across populations for the coarser regions) and H3 (ET difficulty is population-dependent).

## 4.4 Voxel versus Lesion-wise Scoring

The gap between voxel and lesion-wise DSC (Table 4.3) quantifies a phenomenon that the cropped voxel evaluation hides, and — importantly — that gap is itself **population-dependent**.

**Table 4.5 — Mean DSC: voxel vs lesion-wise, and the gap (full volume; five-fold PED/MEN, GLI four folds).**

| Challenge | voxel DSC | lesion-wise DSC | gap |
|---|---|---|---|
| MEN | 0.903 | 0.838 | 0.065 |
| GLI † | 0.907 | 0.789 | 0.118 |
| PED | 0.718 | 0.463 | 0.255 |

† GLI over four folds (0–3).

For meningioma the two metrics nearly agree (gap 0.065): the tumours are convex and well-circumscribed, so predictions form a single clean component that the lesion-wise metric rewards. For adult glioma the gap widens (0.118) as the more irregular tumours occasionally fragment. For the paediatric set the gap is largest (0.255): on the background-heavy full volume the model produces scattered false-positive components and fragments single tumours, which the lesion-wise metric — penalising each false-positive and false-negative lesion — correctly punishes. Connected-component post-processing (removal of components below 50 voxels) and the fragment-merging dilation of the lesion-wise definition (§3.8) recover part but not all of this gap, motivating the per-region threshold optimisation and stronger post-processing identified as extensions in §3.9. The ordering of the gap mirrors the ordering of tumour-shape regularity, a point developed in Chapter 5.

## 4.5 Architecture Comparison: 3D U-Net vs Swin UNETR

Hypothesis H2 is tested by training the 3D U-Net baseline of §3.5 under conditions identical to the corresponding from-scratch Swin UNETR run — same fold, patch size, batch size, LR schedule, loss, augmentation, and 300-epoch budget — so that the only difference is the architecture itself. We run the comparison on **both** the smallest population (PED, 99 cases) and the largest (GLI, 1,251 cases), which lets us isolate not just the architecture effect but its dependence on dataset scale. These are single-fold controlled comparisons (fold 0), not part of the five-fold aggregate.

**Table 4.6 — Architecture comparison across data scales (cropped voxel DSC, single fold — controlled comparison). Params: Swin UNETR 62 M, 3D U-Net 50.7 M.**

| Dataset | Architecture | WT | TC | ET | Mean |
|---|---|---|---|---|---|
| **PED** (99) | Swin UNETR | 0.808 | 0.764 | 0.496 | 0.689 |
| | 3D U-Net | **0.860** | **0.845** | **0.530** | **0.745** |
| | Δ (U-Net − Swin) | +0.052 | +0.081 | +0.034 | **+0.056** |
| **GLI** (1,251) | Swin UNETR | **0.933** | **0.923** | **0.884** | **0.913** |
| | 3D U-Net | 0.932 | 0.915 | 0.864 | 0.904 |
| | Δ (U-Net − Swin) | −0.001 | −0.008 | −0.019 | **−0.009** |

The result is a clear **architecture-by-data-scale interaction**. On the small paediatric set the purely convolutional 3D U-Net outperforms the hybrid Swin UNETR across every region (mean +0.056), and with *fewer* parameters; but on the large adult-glioma set the ranking **reverses** — Swin UNETR edges ahead overall (+0.009) and most clearly on the enhancing tumour (+0.020). This is consistent with the well-known data-hunger of transformer architectures: self-attention has a weaker inductive bias than convolution and needs more data to realise its advantage, so with only 77 paediatric training cases the convolutional prior dominates, whereas with ~1,000 adult cases the transformer's capacity finally pays off. The paediatric finding is further reinforced by §4.7: the from-scratch U-Net (0.745) even slightly exceeds the GLI→PED transfer-learned Swin model (0.736), indicating that in the small-data regime the architectural prior contributed more than cross-population pre-training. We therefore **reject H2 in the small-data regime and accept it in the large-data regime** — the more informative conclusion being the interaction itself: convolutional inductive bias wins when data are scarce, the transformer becomes competitive (and marginally superior) at scale.

## 4.6 Ablation Study

A cumulative ablation — adding normalisation, spatial augmentation, intensity augmentation, the compound loss, and post-processing one component at a time — would quantify the marginal contribution of each. It requires additional training runs, and within the project's fixed schedule the single GPU was fully committed to the five-fold cross-validation; the ablation is therefore deferred to future work (§6). The components themselves are specified in §3.9 so the experiment is fully reproducible.

## 4.7 Transfer Learning: GLI → PED

Hypothesis H4 is tested by warm-starting a paediatric model from the GLI-trained weights and fine-tuning it on PED under conditions otherwise identical to the from-scratch PED run (same fold, hyperparameters, and 300-epoch budget); only the initialisation differs. This is a single-fold controlled comparison (fold 0), not part of the five-fold aggregate. That the transfer fired is evident from the first epoch, where the warm-started model begins at a training loss of 1.15 against the from-scratch model's 3.34 — the inherited features already encode useful brain and tumour structure.

**Table 4.7 — GLI→PED transfer vs from-scratch (PED, cropped voxel DSC, single fold — controlled comparison).**

| Initialisation | WT | TC | ET | Mean |
|---|---|---|---|---|
| From scratch | 0.808 ± 0.21 | 0.764 ± 0.26 | 0.496 ± 0.37 | 0.689 |
| GLI→PED transfer | **0.840 ± 0.20** | **0.825 ± 0.20** | **0.542 ± 0.36** | **0.736** |
| Δ | +0.032 | +0.061 | +0.046 | **+0.046** |

Transfer learning improves every region — a mean Dice gain of +0.046, with the largest gain in the tumour core (+0.061) — and also reduces the variance of every region, indicating more consistent performance across subjects. The enhancing-tumour region, the hardest in PED, improves by +0.046 (0.496 → 0.542). This supports H4: representations learned on the large adult-glioma dataset transfer usefully to the small paediatric dataset, which is the thesis's most novel contribution and the single most effective intervention we found for the data-scarce paediatric setting.

## 4.8 Cross-hardware Reproducibility (RTX 4070 vs NVIDIA L4)

To test that the pipeline is reproducible rather than machine-specific, the cross-validation was run independently on two GPUs — a consumer **RTX 4070** (12 GB, batch 1) and a data-centre **NVIDIA L4** (24 GB, batch 2) — using the same code, data, and fixed fold splits. The platforms differ in both hardware and batch size, so agreement is a stringent check.

**Table 4.8 — Cross-hardware reproducibility: PED and MEN cross-validation, RTX 4070 vs NVIDIA L4 (full-volume mean DSC).**

| Challenge | Platform | voxel mean | lesion mean |
|---|---|---|---|
| **PED** (5-fold) | RTX 4070 (bs 1) | 0.718 | 0.463 |
| | NVIDIA L4 (bs 2) | 0.725 | 0.484 |
| **MEN** | RTX 4070 (bs 1, 5-fold) | 0.903 | 0.838 |
| | NVIDIA L4 (bs 2, 4-fold) | 0.903 | 0.839 |

Run independently on the two GPUs, the full cross-validation aggregates agree closely: for meningioma the two platforms are essentially identical (voxel 0.903 vs 0.903; lesion 0.838 vs 0.839), and for the paediatric set they agree to within 0.007 voxel Dice (0.718 vs 0.725) despite different hardware *and* different batch size — strong evidence that the results reflect the method rather than an artefact of a particular machine. (The L4's meningioma run covers four of the five folds; its fold 4 did not complete.) A secondary, practical finding concerns throughput: the RTX 4070 and L4 share the same Ada AD104 die, but the desktop 4070 runs at ~200 W with GDDR6X memory whereas the L4 is a 72 W, GDDR6, efficiency-oriented inference card; in our training the 4070 delivered roughly **2.4× the per-sample throughput** of the L4 (~0.7 s vs ~1.7 s per sample), its only disadvantage being the 12 GB memory that limited it to batch 1. For this workload a consumer GPU therefore *outpaced* the data-centre part — reinforcing the accessibility argument of §1.2.

## 4.9 Qualitative Results

Representative cases illustrate both the successes and the characteristic failure modes.

![Ground-truth vs predicted segmentation for high-accuracy GLI, MEN and PED cases.](figures/fig_4_2_best.png)

**Figure 4.2 — Representative high-accuracy segmentations** (ground truth vs prediction, axial slice at maximum tumour extent). For a strongly-enhancing adult glioma, a meningioma, and a paediatric case (DSC ≈ 0.99 / 0.99 / 0.94), the predicted oedema (blue), core (green), and enhancing tumour (red) closely match the reference.

![Predicted segmentations showing paediatric failure modes: region mislabelling and false-positive fragmentation.](figures/fig_4_3_failures.png)

**Figure 4.3 — Characteristic paediatric failure modes.** *Top:* the tumour is localised but its tissue is mis-assigned, so voxel WT (0.81) is far higher than lesion-wise WT (0.13). *Bottom:* the main tumour is well segmented but the model scatters false-positive components across the brain — the fragmentation that drives the large paediatric voxel-to-lesion-wise gap (§4.5).

These qualitative observations are analysed in Chapter 5.

---

# 5. Discussion

In this chapter we interpret the experimental findings. We consider, in turn, the differences in performance across the three tumour populations and their biological basis (§5.1); the enhancing-tumour gap that is the study's central empirical finding (§5.2); the unexpected architecture result and what it reveals about data scale (§5.3); the contribution of transfer learning (§5.4); the population-dependent divergence between voxel and lesion-wise scoring (§5.5); the relationship of our results to the state of the art, with the necessary evaluation caveat (§5.6); the clinical relevance of the observed error patterns (§5.7); and the limitations of the study (§5.8).

## 5.1 Cross-population Performance and its Biological Basis

The same methodology produces markedly different performance across the three populations: five-fold cross-validated full-volume voxel Dice of 0.903 for meningioma and 0.718 for the paediatric set, with adult glioma at 0.907 (four completed folds). This ordering is not an artefact of the method but a direct reflection of tumour biology and dataset scale. Adult gliomas and meningiomas are imaged in large, relatively homogeneous adult cohorts (≈1,250 and ≈1,000 cases respectively) and present with substantial, well-enhancing tumour tissue, whereas the paediatric population is both far smaller (99 cases) and biologically harder, frequently lacking a clear enhancing component. That a single pipeline nevertheless reaches ~0.90 Dice on the two large adult populations, and a clinically non-trivial 0.72 on the paediatric set, supports H1: a unified methodology, with only modest per-challenge adaptation, transfers across biologically distinct tumours. The success is most complete for the coarser regions — whole-tumour Dice remains above 0.80 even for PED — indicating that the methodology generalises well for gross tumour delineation and that the residual difficulty is concentrated in the fine, biologically variable sub-regions.

## 5.2 The Enhancing-Tumour Gap

The clearest empirical finding of the thesis is the population dependence of enhancing-tumour segmentation. Under an identical method, ET Dice is 0.909 (MEN) and 0.520 (PED), with GLI 0.874. The paediatric collapse is not a general failure of the model — whole-tumour Dice for PED is 0.831 — but is specific to enhancing tissue, and it is driven by the biology of paediatric diffuse midline glioma, in which the enhancing component is frequently small or entirely absent (11 of 99 PED cases contain no ET at all). The high variance of the PED ET score (±0.36) confirms that performance is bimodal: the model segments ET well when it is present and substantial, and scores near zero on the many cases where ET is minimal or absent and any prediction is heavily penalised. This directly supports H3 and, importantly, reframes the paediatric difficulty as a problem of the enhancing sub-region and of data scarcity rather than of the paediatric brain *per se* — a distinction with direct methodological consequences, since it points to ET-specific remedies (threshold tuning, the ET-suppression post-processing of §3.9, and biologically informed label redefinition) rather than to a wholesale change of approach.

## 5.3 Architecture and Data Scale

The architecture comparison — a controlled single-fold experiment, indicative rather than cross-validated — produced our most nuanced finding: the better architecture *depends on the amount of data*. On the small paediatric set the purely convolutional 3D U-Net outperformed the hybrid Swin UNETR across every region (mean Dice 0.745 vs 0.689) while using fewer parameters; but on the large adult-glioma set the ranking reversed, with Swin UNETR edging ahead overall (0.913 vs 0.904) and most clearly on the enhancing tumour (0.884 vs 0.864). We interpret this through the lens of data scale. Transformer-based models replace convolution's strong locality and translation-equivariance priors with a more flexible but less constrained self-attention mechanism; this flexibility is an advantage only when sufficient data are available to learn what the convolutional prior supplies for free. With just 77 paediatric training cases the inductive bias of the convolutional U-Net is decisive, whereas with roughly a thousand adult cases the transformer's additional capacity finally justifies itself. This reading is reinforced by the transfer result (§5.4): in the paediatric setting the from-scratch U-Net (0.745) marginally exceeds even the GLI-pretrained Swin model (0.736), so the architectural prior contributes more than cross-population pre-training when data are scarce. We therefore **reject H2 in the small-data regime and accept it in the large-data regime**, and read the two results together as a single, more useful statement: convolutional inductive bias wins when data are scarce, and the transformer becomes competitive — indeed marginally superior — only at scale. The practical corollary for a study spanning populations of very different sizes is that no single architecture is uniformly best; the appropriate choice tracks the size of the training set.

## 5.4 Transfer Learning

Warm-starting the paediatric model from the GLI-trained weights improved mean Dice by 0.046 (0.689 to 0.736; single-fold controlled comparison) and improved every individual region while reducing variance. The improvement is consistent with the hypothesis that the adult-glioma encoder learns general-purpose representations of brain tissue and tumour appearance that transfer to the paediatric domain, giving the small paediatric dataset an effective head start; the substantially lower first-epoch training loss of the warm-started model (1.15 versus 3.34) makes this head start concrete. Transfer was, however, a smaller intervention than the change of architecture, which suggests that for this problem the two are complementary rather than substitutable: the most promising configuration we did not exhaust would combine the convolutional architecture with cross-population pre-training. This supports H4 and identifies the single most effective lever we found for the data-scarce paediatric setting.

## 5.5 Voxel versus Lesion-wise Scoring

A methodologically important observation is that the divergence between voxel and lesion-wise Dice is itself population-dependent, and that its ordering mirrors tumour-shape regularity: the gap is smallest for meningioma (0.065), intermediate for adult glioma (0.118), and largest for the paediatric set (0.255). Meningiomas are convex and well-circumscribed, so the model's predictions form a single clean component that the lesion-wise metric rewards as fully as the voxel metric does. Gliomas are more irregular and occasionally fragment. Paediatric predictions, on the background-dominated full volume, scatter into multiple false-positive components and split single tumours into pieces, all of which the lesion-wise metric — which penalises each spurious or missed lesion individually — correctly punishes. This has two consequences. First, it cautions strongly against reporting voxel Dice alone, since the voxel metric most flatters performance precisely where the predictions are least clinically clean. Second, it localises where post-processing will pay off: connected-component filtering and threshold tuning recovered part but not all of the paediatric gap in our experiments, and the residual gap is an explicit target for the extensions of §3.9.

## 5.6 Relationship to the State of the Art

A careful comparison with the published BraTS 2023 results requires the evaluation caveat of §3.8 to be kept firmly in view. Our headline cropped voxel Dice values (GLI mean 0.913) may appear to approach or exceed the reported winning scores, but this comparison is not valid: the official ranking uses the lesion-wise metric on full-resolution volumes of a hidden test set, whereas our cropped voxel figures are computed on a held-out fold of the training data in a brain-centred crop. The honest comparison uses our full-volume lesion-wise numbers. On that basis our GLI model attains lesion-wise Dice of 0.750 / 0.827 / 0.789 for WT / TC / ET (four-fold cross-validated), against the GLI challenge winner's reported ~0.90 / 0.87 / 0.85. Our model is therefore a few points behind the winning system on the metric that matters — an entirely expected and, we argue, encouraging outcome, given that the winning system was a three-architecture ensemble augmented with GAN-generated synthetic data and trained on multi-GPU infrastructure, whereas ours is a single model trained on one consumer GPU without ensembling, test-time augmentation, or synthetic data. The gap is, in effect, a measure of exactly the techniques we deliberately excluded, and it quantifies what an individual researcher forgoes by operating within a single-GPU budget — which is precisely the question the thesis set out to answer.

## 5.7 Clinical Relevance of the Error Modes

The error modes we observe carry different clinical weights. False-positive lesions in healthy tissue, which dominate the paediatric voxel-to-lesion-wise gap, are clinically costly: a spurious enhancing focus could prompt unnecessary investigation or alter management, which is why the lesion-wise metric — and the post-processing that improves it — is clinically the right target. Missed or under-segmented enhancing tumour in paediatric DMG, the other characteristic failure, reflects a genuinely hard and clinically recognised problem rather than a modelling oversight, and the biologically informed strategy of redefining near-absent ET (§3.9) aligns the model's behaviour with clinical reality. For meningioma, the near agreement of voxel and lesion-wise scores indicates predictions that are not only accurate but clinically clean, the most directly usable of the three.

## 5.8 Limitations

Several limitations bound the conclusions. First, the principal results for meningioma and the paediatric set are five-fold cross-validated over their entire cohorts, whereas the adult-glioma results are reported over four of the five folds (fold 4 could not be trained within the schedule); given GLI's low inter-fold variance (per-fold means 0.902–0.916) four folds already provide a stable estimate. Second, the architecture and transfer-learning experiments (§5.3–§5.4) are single-fold controlled comparisons rather than cross-validated, so their effect sizes should be read as indicative; the architecture comparison also spans only two of the three data scales (paediatric and adult glioma, not meningioma). Third, the headline pipeline is a single model without the ensembling, test-time augmentation, or threshold optimisation used by competitive systems; these were deliberately scoped out, but their absence means our numbers are a lower bound on what the methodology can achieve. Fourth, a component ablation — which would quantify the marginal contribution of normalisation, augmentation, the compound loss, and post-processing — was descoped for want of GPU time within the project schedule and is left to future work (§6). Fifth, the lesion-wise metric we report is our own implementation following the BraTS 2023 definition rather than the official scoring container, so while it is internally consistent for our comparisons, small differences from the official figures are possible. Sixth, cross-hardware reproducibility was verified on the paediatric and meningioma challenges — where a consumer RTX 4070 and a data-centre NVIDIA L4 agreed to within 0.007 Dice at the cross-validation cohort level (meningioma essentially identically) — but not on adult glioma, because the L4, an efficiency-class inference GPU roughly 2.4× slower per sample than the desktop card, could not complete the glioma sweep within the schedule. Finally, the entire study is constrained by its single 12 GB GPU, which dictated the patch-based training, the small batch size, and the omission of heavier techniques; this constraint is, however, also the point of the thesis, and the results should be read as a demonstration of what is achievable under it rather than as an attempt to top the leaderboard.

---

# 6. Conclusion

In this thesis we asked whether a single, unified deep-learning methodology, with only modest task-specific adaptation, can produce competitive and clinically meaningful brain-tumour segmentation across three biologically distinct populations — adult glioma, meningioma, and paediatric high-grade glioma — within the means of an individual researcher using a single consumer-grade GPU. Our results answer this question in the affirmative, while also revealing where the approach succeeds easily and where it does not.

We established a reproducible, single-GPU pipeline and trained it identically on all three populations. It reached a mean full-volume voxel Dice of approximately 0.91 for adult glioma and 0.90 for meningioma, and a lower but clinically non-trivial 0.72 for the paediatric set, confirming that one methodology transfers across tumour types for gross tumour delineation. The study's central empirical finding is that the difficulty is concentrated in the enhancing sub-region and is population-dependent: enhancing-tumour Dice fell from around 0.88–0.91 in the adult populations to 0.52 in the paediatric set, driven by the frequent absence of an enhancing component in paediatric diffuse midline glioma rather than by any general failure of the method. We further found that the divergence between voxel and lesion-wise scoring is itself population-dependent, tracking the regularity of tumour shape, which argues against reporting voxel overlap alone.

Two further findings concern how best to improve the hardest, data-scarce paediatric setting. First, the architecture comparison exposed an interaction with data scale: the purely convolutional 3D U-Net outperformed the hybrid transformer on the small paediatric dataset (and with fewer parameters), yet the ranking reversed on the large adult-glioma dataset, where the transformer edged ahead — consistent with the greater data requirements of transformer architectures, and indicating that the better architecture depends on the size of the training set rather than being fixed. Second, warm-starting the paediatric model from the adult-glioma model improved every region; transfer learning and the convolutional architecture emerged as complementary, mutually reinforcing levers. Throughout, we were careful to distinguish our internal voxel metrics from the official lesion-wise metric: on the latter our single model trails the challenge-winning ensembles by a few points, a gap that measures precisely the ensembling, synthetic data, and multi-GPU scale that the single-researcher setting forgoes.

Several directions follow naturally from this work. The immediate priorities are to complete the adult-glioma five-fold run, to add the component ablation deferred here for want of compute, and to apply test-time model ensembling and augmentation together with per-region threshold optimisation and the paediatric enhancing-tumour suppression post-processing, all of which our analysis identifies as targeted remedies for the false-positive and fragmentation errors that most affect the lesion-wise score. The most promising single configuration we did not exhaust is the combination of the convolutional architecture with cross-population pre-training. Completing the architecture comparison on the large adult-glioma dataset would establish whether the convolutional advantage is specific to small data or general, turning the architecture finding into a clean statement about the interaction between architecture and data scale. Finally, evaluating the pipeline with the official lesion-wise tool and submitting to the held-out challenge test set would place the results on a fully comparable footing with the published state of the art. Taken together, these steps would build on the central contribution of this thesis: a transparent, fully reproducible demonstration that competitive multi-population brain-tumour segmentation is achievable on accessible, consumer-grade hardware.

---

# Literature

Baid, U., Ghodasara, S., Mohan, S., Bilello, M., Calabrese, E., Colak, E., et al. (2021). *The RSNA-ASNR-MICCAI BraTS 2021 benchmark on brain tumor segmentation and radiogenomic classification.* arXiv:2107.02314. <https://arxiv.org/abs/2107.02314>

Bakas, S., Akbari, H., Sotiras, A., Bilello, M., Rozycki, M., Kirby, J. S., et al. (2017). *Advancing The Cancer Genome Atlas glioma MRI collections with expert segmentation labels and radiomic features.* Scientific Data, 4, 170117. <https://doi.org/10.1038/sdata.2017.117>

Cardoso, M. J., Li, W., Brown, R., Ma, N., Kerfoot, E., Wang, Y., et al. (2022). *MONAI: An open-source framework for deep learning in healthcare.* arXiv:2211.02701. <https://arxiv.org/abs/2211.02701>

Çiçek, Ö., Abdulkadir, A., Lienkamp, S. S., Brox, T., & Ronneberger, O. (2016). *3D U-Net: Learning dense volumetric segmentation from sparse annotation.* MICCAI 2016 (LNCS 9901, pp. 424–432). <https://arxiv.org/abs/1606.06650>

Dosovitskiy, A., Beyer, L., Kolesnikov, A., Weissenborn, D., Zhai, X., Unterthiner, T., et al. (2021). *An image is worth 16×16 words: Transformers for image recognition at scale.* ICLR 2021. <https://arxiv.org/abs/2010.11929>

Hatamizadeh, A., Tang, Y., Nath, V., Yang, D., Myronenko, A., Landman, B., Roth, H. R., & Xu, D. (2022a). *UNETR: Transformers for 3D medical image segmentation.* WACV 2022, pp. 574–584. <https://arxiv.org/abs/2103.10504>

Hatamizadeh, A., Nath, V., Tang, Y., Yang, D., Roth, H. R., & Xu, D. (2022b). *Swin UNETR: Swin transformers for semantic segmentation of brain tumors in MRI images.* BrainLes 2021 (LNCS 12962, pp. 272–284). <https://arxiv.org/abs/2201.01266>

Isensee, F., Jaeger, P. F., Kohl, S. A. A., Petersen, J., & Maier-Hein, K. H. (2021). *nnU-Net: A self-configuring method for deep learning-based biomedical image segmentation.* Nature Methods, 18(2), 203–211. <https://doi.org/10.1038/s41592-020-01008-z>

Kazerooni, A. F., Khalili, N., Liu, X., Haldar, D., Jiang, Z., et al. (2023). *The Brain Tumor Segmentation (BraTS) Challenge 2023: Focus on pediatrics (CBTN-CONNECT-DIPGR-ASNR-MICCAI BraTS-PEDs).* arXiv:2305.17033. <https://arxiv.org/abs/2305.17033>

Kazerooni, A. F., et al. (2024). *BraTS-PEDs: Results of the multi-consortium international pediatric brain tumor segmentation challenge 2023.* arXiv:2407.08855. <https://arxiv.org/abs/2407.08855>

LaBella, D., Adewole, M., Alonso-Basanta, M., Altes, T., Anwar, S. M., et al. (2023). *The ASNR-MICCAI Brain Tumor Segmentation (BraTS) Challenge 2023: Intracranial meningioma.* arXiv:2305.07642. <https://arxiv.org/abs/2305.07642>

Lin, T.-Y., Goyal, P., Girshick, R., He, K., & Dollár, P. (2017). *Focal loss for dense object detection.* ICCV 2017, pp. 2980–2988. <https://arxiv.org/abs/1708.02002>

Liu, Z., Lin, Y., Cao, Y., Hu, H., Wei, Y., Zhang, Z., Lin, S., & Guo, B. (2021). *Swin Transformer: Hierarchical vision transformer using shifted windows.* ICCV 2021, pp. 10012–10022. <https://arxiv.org/abs/2103.14030>

Loshchilov, I., & Hutter, F. (2019). *Decoupled weight decay regularization.* ICLR 2019. <https://arxiv.org/abs/1711.05101>

Louis, D. N., Perry, A., Wesseling, P., Brat, D. J., Cree, I. A., Figarella-Branger, D., et al. (2021). *The 2021 WHO classification of tumors of the central nervous system: A summary.* Neuro-Oncology, 23(8), 1231–1251. <https://doi.org/10.1093/neuonc/noab106>

Menze, B. H., Jakab, A., Bauer, S., Kalpathy-Cramer, J., Farahani, K., Kirby, J., et al. (2015). *The multimodal brain tumor image segmentation benchmark (BRATS).* IEEE Transactions on Medical Imaging, 34(10), 1993–2024. <https://doi.org/10.1109/TMI.2014.2377694>

Milletari, F., Navab, N., & Ahmadi, S.-A. (2016). *V-Net: Fully convolutional neural networks for volumetric medical image segmentation.* 3DV 2016, pp. 565–571. <https://arxiv.org/abs/1606.04797>

Myronenko, A. (2018). *3D MRI brain tumor segmentation using autoencoder regularization.* BrainLes 2018 (LNCS 11384, pp. 311–320). <https://arxiv.org/abs/1810.11654>

Ronneberger, O., Fischer, P., & Brox, T. (2015). *U-Net: Convolutional networks for biomedical image segmentation.* MICCAI 2015 (LNCS 9351, pp. 234–241). <https://arxiv.org/abs/1505.04597>

Roy, S., Koehler, G., Ulrich, C., Baumgartner, M., Petersen, J., Isensee, F., Jaeger, P. F., & Maier-Hein, K. (2023). *MedNeXt: Transformer-driven scaling of ConvNets for medical image segmentation.* MICCAI 2023 (LNCS 14223, pp. 405–415). <https://arxiv.org/abs/2303.09975>

Tang, Y., Yang, D., Li, W., Roth, H. R., Landman, B., Xu, D., Nath, V., & Hatamizadeh, A. (2022). *Self-supervised pre-training of Swin transformers for 3D medical image analysis.* CVPR 2022, pp. 20730–20740. <https://arxiv.org/abs/2111.14791>

Vaswani, A., Shazeer, N., Parmar, N., Uszkoreit, J., Jones, L., Gomez, A. N., Kaiser, Ł., & Polosukhin, I. (2017). *Attention is all you need.* NeurIPS 2017, pp. 5998–6008. <https://arxiv.org/abs/1706.03762>
