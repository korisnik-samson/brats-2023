# 2. Theoretical Background

> **Drafting notes (remove before submission).** English body, first-person plural where we
> relate prior work to our own choices. Inline *(Author, Year)* citations are collected at
> the end of the chapter and become footnotes + *Literatura* entries on `.docx` conversion.
> References tagged **[verify]** are real works whose exact bibliographic details (and, for
> the BraTS 2023 winning methods, the reported scores) must be confirmed against the
> official proceedings before submission; none are invented. Quantitative winner results in
> §2.7 are reproduced from the project's competitive-landscape analysis and are flagged for
> cross-checking.

This chapter establishes the conceptual and technical foundation for the thesis. We first
describe the three tumour populations studied and the MRI on which they are imaged
(§2.1–§2.2), then trace the development of medical-image segmentation from classical methods
to convolutional and transformer-based deep networks (§2.3–§2.5). We close by situating the
work within the BraTS challenge and its 2023 sub-challenges, reviewing the leading 2023
solutions, and defining the metrics by which segmentation quality is judged (§2.6–§2.8).

## 2.1 Brain Tumours: Three Distinct Populations

Tumours of the central nervous system are classified by the World Health Organization
according to a combination of histological and, increasingly, molecular features (Louis et
al., 2021). The three BraTS 2023 populations studied here are deliberately chosen to span a
wide range of this taxonomy, and their differences are central to the thesis.

**Adult diffuse glioma (GLI).** Gliomas arise from glial cells and are the most common
malignant primary brain tumours in adults. High-grade gliomas, in particular glioblastoma,
are characterised by rapid, infiltrative growth, pronounced peritumoural oedema, a necrotic
core, and irregular contrast enhancement reflecting a disrupted blood–brain barrier. They
are typically large at presentation and intra-axial (arising within the brain parenchyma),
and their infiltrative margins make precise delineation difficult. Adult glioma has been the
historical focus of the BraTS challenge and remains its largest and best-characterised
population.

**Meningioma (MEN).** Meningiomas arise from the meningothelial cells of the arachnoid and
are the most common primary intracranial tumours overall; the majority are benign (WHO grade
1), though atypical and anaplastic variants occur. In contrast to glioma they are
*extra-axial* — they grow from the meningeal coverings rather than within the brain — and are
frequently located at the skull base or along the falx, often abutting bone and producing a
characteristic dural "tail" of enhancement. Their convex, well-circumscribed shape and
skull-adjacent location create segmentation challenges quite different from those of glioma,
notably the risk of clipping peripheral tumour during brain-focused cropping.

**Paediatric high-grade glioma (PED).** Paediatric high-grade gliomas, including diffuse
midline glioma (and its brainstem form, historically termed DIPG), are rare but
devastating, with a prognosis that remains poor despite treatment. Molecularly and
radiologically they differ substantially from their adult counterparts: in particular, the
enhancing-tumour component is frequently small or even absent, which — as we will show — makes
the ET region exceptionally difficult to segment in this population. The paediatric dataset
is also far smaller than the adult one, compounding the difficulty with a data-scarcity
problem. Together these properties make PED an informative stress-test of how well a
methodology generalises beyond the adult-glioma regime for which it is usually designed.

## 2.2 Multi-parametric MRI and Tumour Sub-regions

Each BraTS subject is imaged with four co-registered MRI sequences, each sensitising the
acquisition to different tissue properties: native T1-weighted (*t1n*), contrast-enhanced
T1-weighted (*t1c*), T2-weighted (*t2w*), and T2 fluid-attenuated inversion recovery
(*t2f*/FLAIR). The contrast-enhanced T1 sequence highlights regions of blood–brain-barrier
breakdown and is therefore the principal cue for enhancing tumour; FLAIR suppresses
cerebrospinal-fluid signal and makes peritumoural oedema conspicuous; T1 and T2 provide
complementary anatomical and tissue contrast. The complementary nature of the four sequences
is precisely why multi-parametric input is used: no single sequence delineates all tumour
sub-regions reliably.

The challenge annotations label each voxel with one of three mutually exclusive tissue
classes — necrotic/non-enhancing tumour core, peritumoural oedema, and enhancing tumour —
from which three nested, clinically meaningful **evaluation regions** are derived: the
*whole tumour* (WT), comprising all tumour tissue; the *tumour core* (TC), comprising the
core and enhancing components; and the *enhancing tumour* (ET). By construction these regions
are nested (ET ⊆ TC ⊆ WT). Predicting these overlapping regions directly — rather than the
disjoint tissue labels — is the convention adopted by most strong BraTS methods and by this
thesis, because it aligns the training objective with the quantities on which the challenge
is scored.

## 2.3 From Classical to Deep-Learning Segmentation

Early approaches to brain-tumour segmentation relied on intensity thresholding, region
growing, atlas registration, and classical machine-learning classifiers operating on
hand-engineered features. While useful, these methods struggled with the intensity
heterogeneity of multi-institutional MRI and with the wide morphological variability of
tumours. The decisive shift came with deep convolutional neural networks, which learn
hierarchical features directly from data and, given sufficient training examples,
substantially outperform hand-engineered pipelines. The BraTS benchmark has both tracked and
accelerated this transition, with essentially all competitive entries since the late 2010s
being deep-learning based (Menze et al., 2015; Bakas et al., 2017).

## 2.4 Convolutional Architectures for Segmentation

The architecture that has most shaped medical-image segmentation is the **U-Net**
(Ronneberger et al., 2015), an encoder–decoder network with skip connections that pass
high-resolution features from the contracting path to the expanding path, allowing precise
localisation while retaining semantic context. Because brain MRI is inherently volumetric,
three-dimensional variants quickly followed: the **3D U-Net** (Çiçek et al., 2016) and
**V-Net** (Milletari et al., 2016) extend the design to full volumes and form the basis of
the 3D U-Net baseline used in this thesis.

Two refinements are especially relevant. **nnU-Net** (Isensee et al., 2021) is not a new
architecture but a self-configuring framework that automatically adapts preprocessing, patch
size, and training schedule to a given dataset; it has won or placed at the top of numerous
medical-segmentation challenges, including BraTS, and constitutes the *de facto* baseline
that strong solutions must match. **SegResNet** (Myronenko, 2018), a residual encoder–decoder
with an auxiliary autoencoder regularisation branch, and the more recent **MedNeXt** (Roy et
al., 2023), which modernises the convolutional block with design ideas drawn from
transformers, recur prominently among top BraTS entries — particularly, as we will see, for
meningioma.

## 2.5 Transformers and Hybrid Architectures

The transformer (Vaswani et al., 2017), originally developed for natural-language
processing, replaces convolution's local receptive field with a self-attention mechanism that
models long-range dependencies directly. The **Vision Transformer** (ViT; Dosovitskiy et al.,
2021) applied this idea to images by treating image patches as tokens, while the **Swin
Transformer** (Liu et al., 2021) introduced a hierarchical, shifted-window attention scheme
that restores the multi-scale inductive bias useful for dense prediction and reduces the
quadratic cost of global attention.

For volumetric medical images, these ideas were combined with the U-Net's encoder–decoder
structure. **UNETR** (Hatamizadeh et al., 2022a) uses a pure transformer encoder with a
convolutional decoder, and **Swin UNETR** (Hatamizadeh et al., 2022b) employs a hierarchical
Swin-Transformer encoder coupled to a convolutional decoder through skip connections,
yielding a hybrid that captures both long-range context and fine local detail. Swin UNETR has
become a standard component of competitive BraTS pipelines, frequently as a partner in an
ensemble, and self-supervised pre-training of its encoder on large unlabelled medical-image
collections has been shown to improve downstream performance (Tang et al., 2022). Swin UNETR
is the principal model studied in this thesis, and the comparison between it and the purely
convolutional 3D U-Net under identical conditions is one of our central experiments.

## 2.6 The BraTS Challenge and its 2023 Sub-challenges

The Brain Tumour Segmentation (BraTS) challenge, first held in 2012, has become the reference
benchmark for the field, providing standardised, expertly annotated, multi-institutional,
co-registered and skull-stripped data together with a common evaluation protocol (Menze et
al., 2015; Bakas et al., 2017; Baid et al., 2021). Successive editions have grown the dataset
and progressively raised the performance bar.

The 2023 edition broadened the challenge from a single adult-glioma task into a *cluster* of
parallel sub-challenges covering biologically and demographically distinct populations,
including adult glioma (GLI), intracranial meningioma (MEN; LaBella et al., 2023 **[verify]**),
and paediatric high-grade glioma (PED; Kazerooni et al., 2023 **[verify]**), among others.
All sub-challenges share a common data format and evaluation regions (WT/TC/ET), which is
precisely what makes a *unified* methodology across them both feasible and scientifically
interesting — and what this thesis sets out to study.

## 2.7 Related Work: Leading BraTS 2023 Solutions

Analysing the top-performing 2023 entries reveals both a set of population-specific
strategies and a striking convergence of general practice.

**Adult glioma (GLI).** The winning solution combined a three-architecture ensemble
(nnU-Net, Swin UNETR, and the BraTS 2021 winning network); its decisive advantage was
synthetic-data generation, using a generative adversarial network to insert realistic
synthetic tumours into healthy brain regions, together with a registration-based augmentation
that transplanted existing tumours into new anatomies. Reported validation scores were of the
order of DSC 0.90 / 0.87 / 0.85 and HD95 ≈ 15 / 14 / 18 mm for WT / TC / ET **[verify]**.

**Paediatric (PED).** The winning approach ensembled nnU-Net and Swin UNETR with *label-wise*
aggregation — weighting each model's contribution per region rather than averaging full
probability maps — and applied a cross-validated per-region threshold search. A biologically
informed post-processing step *redefined* the enhancing-tumour label for cases in which the
ET volume was very small relative to the total tumour, reassigning those voxels; this
directly addressed the near-absent ET of diffuse midline tumours and produced a measurable
gain. Reported mean Dice was approximately 0.84 / 0.81 / 0.65 for WT / TC / ET, with ET the
hardest region across all teams **[verify]**.

**Meningioma (MEN).** Top meningioma solutions were dominated by SegResNet- and
MedNeXt-based pipelines with deep supervision and five-fold cross-validation, several built
on automated configuration frameworks; targeted post-processing exploiting the typically
convex shape of meningiomas was common **[verify]**.

**Universal patterns.** Across the leading entries we observe a consistent recipe: a U-Net-family
backbone (often nnU-Net) frequently ensembled with Swin UNETR; training on the overlapping
WT/TC/ET regions; a compound Dice + cross-entropy loss; deep supervision; aggressive spatial
and intensity augmentation; five-fold cross-validation with test-time ensembling and
test-time augmentation; mixed-precision training; and connected-component / threshold-based
post-processing. This thesis adopts the *general* elements of this recipe while deliberately
operating within a single-GPU budget and *without* the large ensembles and synthetic-data
generation that distinguish the very top entries — a scope chosen to isolate what is
achievable by an individual researcher.

## 2.8 Evaluation Metrics

Segmentation quality in BraTS is assessed primarily with two complementary metrics, computed
independently for each evaluation region (WT, TC, ET).

The **Dice similarity coefficient** (DSC) measures volumetric overlap between a predicted
region $P$ and the ground truth $G$ as $\mathrm{DSC} = 2|P \cap G| / (|P| + |G|)$, ranging
from 0 (no overlap) to 1 (perfect overlap). By convention, an empty prediction against an
empty ground truth scores 1 (both correctly agree no tumour is present), whereas an empty
prediction against a non-empty ground truth scores 0.

The **95th-percentile Hausdorff distance** (HD95) measures boundary agreement as the
95th-percentile of the distances between the surfaces of $P$ and $G$, expressed in
millimetres; the 95th percentile (rather than the maximum) confers robustness to a small
number of outlying voxels. HD95 is undefined when either region is empty, a case that must be
handled explicitly during evaluation.

A limitation of these voxel-level metrics is that they do not penalise spurious or missed
*lesions* as such. BraTS 2023 therefore introduced a **lesion-wise** formulation, in which
Dice and HD95 are computed per connected tumour component and then averaged, so that false-positive
and false-negative lesions are penalised individually. The lesion-wise metric is generally
harsher than its voxel-level counterpart and is the basis of the official 2023 ranking; the
distinction between the two is important when interpreting our results and comparing them with
the leaderboard, a point we return to in Chapter 5.

---

## References cited in this chapter *(to be formatted as footnotes + Literatura entries)*

- Baid, U., et al. (2021). *The RSNA-ASNR-MICCAI BraTS 2021 Benchmark...* arXiv:2107.02314. **[verify]**
- Bakas, S., et al. (2017). *Advancing the Cancer Genome Atlas glioma MRI collections with expert segmentation labels and radiomic features.* Scientific Data, 4, 170117. **[verify]**
- Çiçek, Ö., et al. (2016). *3D U-Net: Learning dense volumetric segmentation from sparse annotation.* MICCAI. **[verify]**
- Dosovitskiy, A., et al. (2021). *An image is worth 16×16 words: Transformers for image recognition at scale.* ICLR. **[verify]**
- Hatamizadeh, A., et al. (2022a). *UNETR: Transformers for 3D medical image segmentation.* WACV. **[verify]**
- Hatamizadeh, A., et al. (2022b). *Swin UNETR: Swin Transformers for semantic segmentation of brain tumors in MRI images.* BrainLes/MICCAI; arXiv:2201.01266. **[verify]**
- Isensee, F., et al. (2021). *nnU-Net: a self-configuring method for deep learning-based biomedical image segmentation.* Nature Methods, 18, 203–211. **[verify]**
- Kazerooni, A. F., et al. (2023). *The BraTS 2023 challenge on pediatric brain tumors (BraTS-PEDs).* **[verify]**
- LaBella, D., et al. (2023). *The BraTS 2023 challenge on intracranial meningioma segmentation.* **[verify]**
- Liu, Z., et al. (2021). *Swin Transformer: Hierarchical vision transformer using shifted windows.* ICCV. **[verify]**
- Louis, D. N., et al. (2021). *The 2021 WHO Classification of Tumors of the Central Nervous System: a summary.* Neuro-Oncology, 23(8), 1231–1251. **[verify]**
- Menze, B. H., et al. (2015). *The Multimodal Brain Tumor Image Segmentation Benchmark (BRATS).* IEEE TMI, 34(10), 1993–2024. **[verify]**
- Milletari, F., Navab, N., & Ahmadi, S.-A. (2016). *V-Net: Fully convolutional neural networks for volumetric medical image segmentation.* 3DV. **[verify]**
- Myronenko, A. (2018). *3D MRI brain tumor segmentation using autoencoder regularization.* BrainLes/MICCAI. **[verify]**
- Ronneberger, O., Fischer, P., & Brox, T. (2015). *U-Net: Convolutional networks for biomedical image segmentation.* MICCAI. **[verify]**
- Roy, S., et al. (2023). *MedNeXt: Transformer-driven scaling of ConvNets for medical image segmentation.* MICCAI. **[verify]**
- Tang, Y., et al. (2022). *Self-supervised pre-training of Swin Transformers for 3D medical image analysis.* CVPR. **[verify]**
- Vaswani, A., et al. (2017). *Attention is all you need.* NeurIPS. **[verify]**

> **To finalise:** confirm the exact BraTS 2023 sub-challenge data papers (MEN, PED) and the
> winning-method citations + reported scores against the official 2023 proceedings; optionally
> add a CBTRUS/epidemiology source (e.g., Ostrom et al.) if specific incidence figures are
> introduced in §2.1.
