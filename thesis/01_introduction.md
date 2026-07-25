# 1. Introduction

> **Drafting notes (remove before submission).** Body is English, first-person plural, as
> required by the Singidunum template. Citations are given inline as *(Author, Year)* and
> collected at the end of the chapter; on conversion to `.docx` these become footnotes
> **and** an alphabetical entry in *Literatura*. References marked **[verify]** are real,
> well-known works whose exact bibliographic details must be confirmed before submission;
> none are invented. Specific epidemiological figures are deliberately avoided until a
> primary source is attached.

---

Brain tumours are among the most consequential diseases of the central nervous system: even
when histologically benign, their location within a confined cranial space makes them
capable of producing severe neurological deficits, and the most aggressive forms remain
among the deadliest of all human cancers. Accurate delineation of a tumour and its
sub-regions on magnetic resonance imaging (MRI) underpins almost every stage of clinical
care — diagnosis, surgical and radiotherapy planning, and the longitudinal monitoring of
response to treatment. Yet in routine practice this delineation is still performed
manually by experts, a process that is slow, costly, and subject to substantial
inter-observer variability (Menze et al., 2015). The promise of automated, reproducible
segmentation has therefore motivated more than a decade of methodological research,
organised in large part around the *Brain Tumour Segmentation* (BraTS) benchmark (Menze et
al., 2015; Bakas et al., 2017; Baid et al., 2021).

The 2023 edition of BraTS marks an important broadening of this effort. Where earlier
editions focused almost exclusively on adult diffuse glioma, BraTS 2023 introduced a family
of parallel sub-challenges spanning biologically distinct tumour populations, including
adult glioma (GLI), intracranial meningioma (MEN), and paediatric high-grade glioma (PED).
These populations differ not only in cellular origin and typical anatomical location but
also in the appearance, size, and even the *presence* of the tumour sub-regions that the
challenge asks us to segment. This diversity raises a question that is both scientifically
interesting and practically important, and which forms the core of this thesis: **can a
single, unified deep-learning methodology — with only modest, task-specific adaptations —
produce competitive and clinically meaningful segmentations across all three populations,
and can it do so within the computational means of an individual researcher rather than a
well-resourced laboratory?**

## 1.1 Subject and Problem of the Research

The **subject** of this research is the automatic, three-dimensional segmentation of brain
tumours from multi-parametric MRI across the three BraTS 2023 populations (GLI, MEN, PED).
Each subject in the dataset is described by four co-registered MRI sequences — native
T1-weighted (*t1n*), contrast-enhanced T1-weighted (*t1c*), T2-weighted (*t2w*), and T2
fluid-attenuated inversion recovery (*t2f*) — and the segmentation task is to label every
voxel as background or as one of the tumour tissue classes, from which three clinically
meaningful, nested evaluation regions are derived: the *whole tumour* (WT), the *tumour
core* (TC), and the *enhancing tumour* (ET).

The **problem** has two faces. The first is methodological. The strongest BraTS 2023
solutions rely on large ensembles of high-capacity networks, extensive synthetic-data
generation, and multi-GPU training budgets that are out of reach for most individual
researchers and for institutions in resource-limited settings. This creates a reproducibility
and accessibility gap between what is reported at the top of the leaderboard and what an
independent researcher can realistically build upon. The second face is biological. Because
the three populations differ so markedly — adult gliomas are large and infiltrative,
meningiomas are extra-axial and frequently abut the skull, and paediatric high-grade
gliomas include diffuse midline tumours in which the enhancing component may be almost
absent — it is not obvious *a priori* that one methodology can serve all three, nor which
components of a segmentation pipeline are genuinely general and which must be tailored to a
particular tumour biology.

This thesis addresses that gap directly. Rather than pursuing maximum leaderboard rank, we
develop a single, transparent, and fully reproducible pipeline, train it on each of the
three populations under identical conditions on a single consumer-grade graphics processing
unit (an NVIDIA RTX 4070 with 12 GB of memory), and study how its behaviour differs across
the three tumour types. The sample on which the study is conducted is the public BraTS 2023
training data — approximately 1,251 adult glioma, 1,000 meningioma, and 99 paediatric
cases, roughly **2,350 subjects in total** — which far exceeds the minimum sample size
expected of empirical master-level research.

## 1.2 Objectives of the Research

**Scientific objectives.** The scientific aims of the thesis are:

1. to establish a rigorous, reproducible training and evaluation baseline for brain-tumour
   segmentation on each of the three BraTS 2023 sub-challenges using a modern
   three-dimensional architecture;
2. to quantify, through controlled ablation, the marginal contribution of individual
   pipeline components (preprocessing, augmentation, loss formulation, post-processing) and
   thereby to distinguish techniques that are *domain-general* from those that are
   *tumour-specific*;
3. to compare a purely convolutional architecture (3D U-Net) with a hybrid
   convolutional–transformer architecture (Swin UNETR) under identical data, augmentation,
   and training budgets, isolating the contribution of the architecture itself; and
4. to investigate cross-population transfer learning — specifically, whether an encoder
   pre-trained on the large adult-glioma dataset improves segmentation on the much smaller
   paediatric dataset.

**Social objective.** The social aim is to demonstrate, and to document openly, that
competitive and clinically meaningful brain-tumour segmentation is achievable on a single
consumer-grade GPU. By lowering the hardware and engineering barrier to entry — and by
releasing a clear, reproducible pipeline — the work seeks to make this class of research
more accessible to students, smaller institutions, and clinical and research groups in
resource-constrained environments, where multi-GPU infrastructure is rarely available. In
doing so it contributes to the broader goal of equitable access to medical-imaging
artificial-intelligence research.

## 1.3 Hypotheses

The research is guided by four hypotheses, each of which is operationalised and tested in
the Results chapter:

- **H1.** A single unified methodology, with only minor task-specific adaptations, can
  produce clinically meaningful segmentation (high Dice overlap and low boundary error)
  across all three biologically distinct populations.
- **H2.** Under an identical training budget, the hybrid CNN–Transformer architecture (Swin
  UNETR) is at least competitive with, and may exceed, a purely convolutional 3D U-Net.
- **H3.** The difficulty of segmenting the enhancing-tumour (ET) region is
  population-dependent: it is comparatively easy in adult glioma and substantially harder in
  the paediatric population, where the enhancing component is frequently small or absent.
- **H4.** Transfer learning from the large adult-glioma dataset improves paediatric
  segmentation relative to training on the paediatric data alone.

## 1.4 Research Methods

The thesis follows a quantitative, experimental methodology. The empirical material is the
public BraTS 2023 dataset; all experiments use a reproducible five-fold stratified
cross-validation split generated once and fixed for the duration of the study. The
segmentation models are deep neural networks — a three-dimensional U-Net baseline (Çiçek et
al., 2016; Ronneberger et al., 2015) and the Swin UNETR hybrid architecture (Hatamizadeh et
al., 2022) — implemented with PyTorch and the MONAI medical-imaging framework (Cardoso et
al., 2022) and trained under mixed-precision on a single GPU. Performance is assessed with
the standard BraTS metrics — the Dice similarity coefficient (DSC) and the 95th-percentile
Hausdorff distance (HD95) — computed per evaluation region, complemented by the lesion-wise
metric introduced in 2023. Beyond the basic analytical and synthetic methods common to all
empirical research, the work makes central use of the comparative method (across the three
populations and the two architectures) and of statistical analysis of the resulting metric
distributions; controlled ablation is used to attribute performance to individual pipeline
components, in the hypothetico-deductive tradition of testing the stated hypotheses against
measured outcomes.

## 1.5 Structure of the Thesis

The remainder of the thesis is organised as follows. **Chapter 2** reviews the theoretical
background — the relevant tumour biology and MRI, the evolution of deep-learning
segmentation from the U-Net to transformer-based architectures, the BraTS challenge, and the
leading 2023 solutions. **Chapter 3** presents the methodology in full: the dataset and
cross-validation protocol, the preprocessing and augmentation pipeline, the model
architectures, the loss formulation, the single-GPU training protocol, and the evaluation
and experimental design. **Chapter 4** reports the results across the three populations,
the architecture comparison, the ablation study, and the transfer-learning experiment.
**Chapter 5** discusses and interprets these results — with particular attention to the
biologically driven differences between populations, characteristic failure modes, the
relationship of our findings to the state of the art, and the study's limitations.
**Chapter 6** concludes with a summary of the principal findings and recommendations for
further research.

---

## References cited in this chapter *(to be formatted as footnotes + Literatura entries)*

- Baid, U., et al. (2021). *The RSNA-ASNR-MICCAI BraTS 2021 Benchmark on Brain Tumor
  Segmentation and Radiogenomic Classification.* arXiv:2107.02314. **[verify]**
- Bakas, S., et al. (2017). *Advancing the Cancer Genome Atlas glioma MRI collections with
  expert segmentation labels and radiomic features.* Scientific Data, 4, 170117. **[verify]**
- Cardoso, M. J., et al. (2022). *MONAI: An open-source framework for deep learning in
  healthcare.* arXiv:2211.02701. **[verify]**
- Çiçek, Ö., Abdulkadir, A., Lienkamp, S. S., Brox, T., & Ronneberger, O. (2016). *3D U-Net:
  Learning dense volumetric segmentation from sparse annotation.* MICCAI. **[verify]**
- Hatamizadeh, A., Nath, V., Tang, Y., Yang, D., Roth, H., & Xu, D. (2022). *Swin UNETR:
  Swin Transformers for semantic segmentation of brain tumors in MRI images.* BrainLes /
  MICCAI Workshop; arXiv:2201.01266. **[verify]**
- Menze, B. H., et al. (2015). *The Multimodal Brain Tumor Image Segmentation Benchmark
  (BRATS).* IEEE Transactions on Medical Imaging, 34(10), 1993–2024. **[verify]**
- Ronneberger, O., Fischer, P., & Brox, T. (2015). *U-Net: Convolutional networks for
  biomedical image segmentation.* MICCAI. **[verify]**

> **Citations to add when the relevant passages are finalised:** WHO 2021 classification of
> CNS tumours (Louis et al., 2021) for the tumour-biology statements; the BraTS 2023
> cluster/challenge papers and the GLI/MEN/PED winning-method papers (Chapter 2); the
> lesion-wise metric definition (Chapter 3).
