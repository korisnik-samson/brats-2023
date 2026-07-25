# 3. Methodology

> **Drafting notes (remove before submission).** English, first-person plural. This chapter
> describes the methodology **as actually implemented** in the accompanying code, so that it
> faithfully accounts for the results reported in Chapter 4. Techniques that are part of the
> wider design space but were *not* used to produce the reported results (e.g. deep
> supervision, elastic deformation, model ensembling, test-time augmentation) are presented
> in §3.9 as planned extensions / ablations rather than as part of the core pipeline.
> Citations *(Author, Year)* are collected at the end of the chapter; **[verify]** tags mark
> real works whose bibliographic details are to be confirmed.

This chapter presents the complete methodology: the dataset and cross-validation protocol
(§3.1–§3.2); the preprocessing and augmentation pipeline (§3.3–§3.4); the model
architectures (§3.5); the loss formulation (§3.6); the single-GPU training protocol (§3.7);
the inference and evaluation procedure (§3.8); the experimental design that operationalises
the four hypotheses (§3.9); and the measures taken to ensure reproducibility (§3.10). The
implementation uses Python with PyTorch and the MONAI medical-imaging framework (Cardoso et
al., 2022).

## 3.1 Dataset and Data Representation

We use the public BraTS 2023 training data for the three sub-challenges studied: adult
glioma (GLI), meningioma (MEN), and paediatric high-grade glioma (PED). Each subject is a
directory containing four co-registered, skull-stripped MRI volumes — *t1c*, *t1n*, *t2f*,
*t2w* — and, for training cases, an expert segmentation mask (*seg*). All volumes are
provided at 1 mm isotropic resolution on a common grid. The segmentation mask labels each
voxel as background (0), necrotic/non-enhancing tumour core (1), peritumoural oedema (2), or
enhancing tumour (3); from these the overlapping evaluation regions whole tumour (WT), tumour
core (TC), and enhancing tumour (ET) are derived.

Table 1 summarises the three datasets. The combined sample of roughly 2,350 annotated
subjects far exceeds the minimum required for empirical master-level research; equally
relevant to the thesis is the number of cases in which the enhancing tumour is *absent*,
which is appreciable in the paediatric population and underlies hypothesis H3.

> **Table 1 — Dataset summary.** *(values from the project's split-generation step)*
>
> | Sub-challenge | Training cases | ET-absent cases | Key challenge |
> |---|---|---|---|
> | GLI | 1,251 | 33 | large, infiltrative; class imbalance |
> | MEN | 1,000 | 1 | extra-axial; skull-base truncation |
> | PED | 99 | 11 | very small dataset; absent ET in DMG |

Internally, each subject is represented as a four-channel tensor (one channel per modality)
of shape $(4, H, W, D)$, with the segmentation as a single-channel volume; the four channels
are stacked so that the network receives the full multi-parametric input simultaneously.

## 3.2 Cross-validation Protocol

To obtain reproducible and representative train/validation partitions we generate, **once**,
a five-fold **stratified** cross-validation split for each challenge and fix it for all
subsequent experiments. Stratification is performed jointly on (i) the quartile of total
tumour volume and (ii) the presence or absence of enhancing tumour, so that each fold
contains a representative mix of small and large tumours and of ET-present and ET-absent
cases — the latter being essential for stable evaluation in the paediatric and meningioma
populations. The split is written to a JSON file and committed to the repository, guaranteeing
that every experiment in the thesis draws on identical partitions. Unless otherwise stated,
we train on the training portion of fold 0 and report results on its held-out validation
portion; the five-fold structure additionally supports the cross-validation ensembling
discussed as an extension in §3.9.

## 3.3 Preprocessing

The preprocessing pipeline is deterministic and is applied identically at training and
inference time. For each subject it proceeds as follows.

**Brain masking and spatial cropping.** A single brain mask is computed as the union of the
non-zero voxels across all four modalities. Computing one shared mask — rather than a
per-modality mask — is essential: the modalities can have slightly different zero-padding and
fields of view, and cropping each independently would destroy the voxel-level correspondence
between channels on which the network relies. From this mask we compute a tight bounding box,
pad it by 10 voxels on each side, and place a fixed-size crop window of $192 \times 192 \times
128$ voxels around its centre, shifting the window inward where it would exceed the volume
bounds. The *identical* window is applied to all four modalities and to the segmentation mask.
The crop size was chosen to be large enough to contain peripheral meningiomas that a tighter
window can clip, while remaining compatible with the encoder's downsampling strides; where a
volume is smaller than the target the result is symmetrically zero-padded.

**Intensity normalisation.** Each modality is normalised independently within the brain mask.
We first clip intensities to the 0.5th and 99.5th percentiles computed over brain voxels,
which removes scanner-specific outliers that would otherwise distort normalisation in this
multi-institutional data; we then apply z-score normalisation using the brain-voxel mean
$\mu$ and standard deviation $\sigma$,
$$ x' = \frac{x - \mu}{\sigma}, $$
and finally rescale the result into the range $[0, 255]$ using the brain-voxel minimum and
maximum, giving a consistent dynamic range across modalities and scanners. Background voxels
remain at or near zero throughout.

**Disk caching.** Because the full preprocessing pipeline involves reading four compressed
NIfTI volumes per subject, it dominates per-epoch time if repeated every epoch. We therefore
cache the preprocessed (but un-augmented) four-channel tensor to disk after its first
computation; subsequent epochs read the cached tensor directly, which on our hardware reduces
per-epoch time by roughly an order of magnitude. Caching the *un-augmented* volume is
deliberate: augmentation is random and must be re-applied afresh each epoch (§3.4).

## 3.4 Augmentation and Patch-based Sampling

Because the full preprocessed volume is too large to train on at the batch sizes a 12 GB GPU
permits, we adopt **patch-based** training. From each subject we sample a $128 \times 128
\times 128$ patch using a foreground-biased scheme that, with high probability, centres the
patch on a tumour voxel; the foreground probability is $0.90$ for GLI and MEN and is raised
to $0.95$ for PED, where the tumour occupies a smaller fraction of the volume. This
concentrates training signal on the tumour and its boundary while still exposing the network
to background through the residual probability.

On each sampled patch we apply, on the fly, a sequence of augmentations implemented with
MONAI's dictionary transforms so that identical spatial transforms are applied to image and
label. Spatial augmentations — random flips along each of the three axes (each with
probability $0.5$) and random $90^{\circ}$ rotations in the axial plane — are applied to both
image and label. Intensity augmentations — random intensity scaling and shifting (probability
$0.3$ each) and additive Gaussian noise (probability $0.15$) — are applied to the image
channels only. These augmentations enlarge the effective training distribution and improve
robustness to the appearance variability of multi-institutional MRI. Richer augmentations
(elastic deformation, modality dropout, copy-paste tumour insertion) are part of the wider
design space and are discussed as planned ablations in §3.9.

## 3.5 Model Architectures

The thesis studies two architectures under identical data, augmentation, and training
conditions, so that any performance difference is attributable to the architecture itself.

**3D U-Net (baseline).** As a purely convolutional reference we use a three-dimensional
U-Net (Çiçek et al., 2016; Ronneberger et al., 2015): an encoder–decoder with skip
connections, convolutional blocks, and instance normalisation, producing a three-channel
output corresponding to the overlapping WT/TC/ET regions. It serves as the baseline against
which the transformer-based model is compared (hypothesis H2).

**Swin UNETR (principal model).** Our principal model is Swin UNETR (Hatamizadeh et al.,
2022b), a hybrid architecture whose encoder is a hierarchical Swin Transformer (Liu et al.,
2021) and whose decoder is convolutional, the two connected by skip connections in the U-Net
style. We instantiate it with four input channels, three output channels, and a feature
embedding size of 48, giving approximately 62 million parameters. To fit the model on 12 GB
of memory we enable **gradient checkpointing**, which trades additional computation for a
substantial reduction in activation memory. Both models predict the three overlapping regions
directly, as logits; the sigmoid activation that maps logits to per-region probabilities is
applied inside the loss and at inference.

## 3.6 Loss Function

Following the consensus among strong BraTS methods, we optimise a **compound** loss that
combines a region-overlap term with a term targeting class imbalance, summed over the three
evaluation regions. For each region $r \in \{\text{WT}, \text{TC}, \text{ET}\}$ we compute a
soft Dice loss,
$$ \mathcal{L}_{\text{Dice}} = 1 - \frac{2 \sum_i p_i g_i + \varepsilon}{\sum_i p_i + \sum_i g_i + \varepsilon}, $$
where $p_i = \sigma(z_i)$ is the predicted probability for voxel $i$, $g_i$ the ground-truth
indicator, and $\varepsilon$ a smoothing constant; and a focal loss based on the binary
cross-entropy between logits and targets, which down-weights easy voxels and focuses learning
on the hard tumour-boundary voxels (Lin et al., 2017). The total loss is the equally weighted
sum of the Dice and focal terms across the three regions,
$$ \mathcal{L} = \sum_{r} \big( \mathcal{L}_{\text{Dice}}^{(r)} + \mathcal{L}_{\text{Focal}}^{(r)} \big). $$
The Dice term drives volumetric overlap, while the focal term improves boundary delineation,
which is where the Hausdorff distance is determined.

## 3.7 Training Protocol

All models are trained under a single-GPU regime designed to fit an NVIDIA RTX 4070 with
12 GB of memory (full hardware/software environment in §3.10).

**Optimisation.** We use the AdamW optimiser (Loshchilov & Hutter, 2019), in which weight
decay is decoupled from the adaptive gradient update, with a peak learning rate of
$1\times10^{-4}$ for GLI and MEN and a lower $5\times10^{-5}$ for the small PED dataset, and
weight decay of $1\times10^{-5}$ (GLI/MEN) or the stronger $1\times10^{-4}$ (PED). The
learning rate follows a schedule of linear warm-up over the first five epochs followed by
cosine decay to a small floor; computing the rate from the epoch index alone makes the
schedule deterministic and exactly resumable from a checkpoint.

**Mixed precision and stability.** Training uses automatic mixed precision: the forward pass
runs under half-precision autocast for speed and memory, while the loss is computed in
single precision to avoid overflow in the Dice reductions, and a gradient scaler manages the
half-precision gradients. Gradients are clipped to a maximum norm of $1.0$ for stability.

**Schedule and checkpointing.** We use a batch size of one $128^3$ patch per step and train
for 300 epochs. The latest model state — together with the optimiser and gradient-scaler
state — is checkpointed every epoch so that training can be resumed exactly after any
interruption, with periodic numbered backups retained. Per-challenge hyperparameters are
summarised in Table 2.

> **Table 2 — Training hyperparameters per sub-challenge.**
>
> | Hyperparameter | GLI | MEN | PED |
> |---|---|---|---|
> | Epochs | 300 | 300 | 300 |
> | Peak learning rate | 1e-4 | 1e-4 | 5e-5 |
> | Weight decay | 1e-5 | 1e-5 | 1e-4 |
> | Warm-up epochs | 5 | 5 | 5 |
> | Patch size | 128³ | 128³ | 128³ |
> | Batch size | 1 | 1 | 1 |
> | Foreground-sampling prob. | 0.90 | 0.90 | 0.95 |
> | Mixed precision | yes | yes | yes |

## 3.8 Inference and Evaluation

Because the network is trained on $128^3$ patches it cannot be applied to a full volume in a
single forward pass; we therefore use **sliding-window inference** (Cardoso et al., 2022). A
$128^3$ window is moved across the volume with 50 % overlap, the per-window predictions are
combined with Gaussian weighting (which smoothly down-weights window edges), and the
accumulated logits are passed through a sigmoid to yield per-region probability maps. The
maps are thresholded — at $0.45$, $0.40$ and $0.45$ for WT, TC and ET respectively — and
converted to a consistent labelling in which the nesting ET ⊆ TC ⊆ WT is respected.

Segmentation quality is then quantified per region with the Dice similarity coefficient and
the 95th-percentile Hausdorff distance (§2.8). We apply the standard empty-region
conventions: for Dice, an empty prediction against an empty ground truth scores 1 and against
a non-empty ground truth scores 0; for HD95, which is undefined when either region is empty,
the subject is excluded from that region's HD95 average and the exclusion is counted and
reported. We report the mean and standard deviation of each metric over the validation fold,
together with per-subject values to expose the distribution of performance.

We note explicitly — and return to the point in Chapter 5 — that this evaluation is computed
with *voxel-level* metrics in the cropped $192\times192\times128$ space on a held-out fold of
the *training* data; it is therefore appropriate for the controlled internal comparisons that
the thesis makes, but is not directly comparable to the official BraTS 2023 leaderboard,
which uses the *lesion-wise* metric on full-resolution volumes of a separate hidden test set.

## 3.9 Experimental Design

The experiments are designed to test the four hypotheses of §1.3 under controlled conditions.
Table 3 maps each hypothesis to the experiment that evaluates it.

> **Table 3 — Hypotheses and the experiments that test them.**
>
> | Hypothesis | Experiment |
> |---|---|
> | H1 — unified methodology across populations | Train and evaluate the same pipeline on GLI, MEN and PED; report per-region DSC/HD95. |
> | H2 — Swin UNETR vs 3D U-Net | Train both architectures under identical data/augmentation/budget on the same fold and compare. |
> | H3 — population-dependent ET difficulty | Compare ET performance and its variance across the three populations. |
> | H4 — GLI→PED transfer | Pre-train the encoder on GLI, fine-tune on PED, and compare against PED-only training. |

**Ablation study.** To attribute performance to individual components, we plan a cumulative
ablation in which elements are added one at a time to a minimal baseline — intensity
normalisation, spatial augmentation, intensity augmentation, the compound loss, patch-based
training with sliding-window inference, and post-processing — measuring the marginal change
in Dice at each step. This isolates which components are *domain-general* and which are
*tumour-specific*, addressing the second scientific objective. We run the ablation primarily
on GLI, whose large dataset yields the most stable metrics, and confirm the full pipeline on
MEN and PED.

**Planned extensions.** Several techniques common to the strongest BraTS entries lie outside
the core single-model pipeline but are natural extensions, each evaluated as an addition to
the baseline: five-fold cross-validation ensembling; test-time augmentation; per-region
threshold optimisation; connected-component and, for PED, ET-suppression post-processing; and
richer augmentation (elastic deformation, modality dropout). We report these as incremental
contributions rather than folding them silently into the headline numbers, preserving a clear
account of what each technique buys.

## 3.10 Reproducibility and Implementation Environment

All experiments are made reproducible by fixing the random seeds of Python, NumPy and PyTorch
(including CUDA) to a common value, by committing the cross-validation split files to the
repository, and by logging the full hyperparameter configuration alongside every checkpoint.
Training progress (per-epoch loss and learning rate) is recorded to disk, and every run is
resumable from its latest checkpoint.

The implementation uses Python 3.13, PyTorch 2.8 with CUDA 12.6, and MONAI 1.5. All training
and evaluation reported in this thesis were performed on a single workstation with an AMD
Ryzen 9 9900X CPU, 32 GB of system memory, and an NVIDIA GeForce RTX 4070 GPU with 12 GB of
memory. This modest, consumer-grade configuration is not incidental but central to the
thesis: it demonstrates that the methodology is reproducible without specialised
infrastructure, in direct support of the social objective stated in §1.2. The principal
constraints it imposes — a small batch size, patch-based training, gradient checkpointing, and
mixed precision — are documented here precisely so that the work can be reproduced, and its
limitations understood, by others working under similar constraints.

---

## References cited in this chapter *(to be formatted as footnotes + Literatura entries)*

- Cardoso, M. J., et al. (2022). *MONAI: An open-source framework for deep learning in healthcare.* arXiv:2211.02701. **[verify]**
- Çiçek, Ö., et al. (2016). *3D U-Net: Learning dense volumetric segmentation from sparse annotation.* MICCAI. **[verify]**
- Hatamizadeh, A., et al. (2022b). *Swin UNETR: Swin Transformers for semantic segmentation of brain tumors in MRI images.* BrainLes/MICCAI; arXiv:2201.01266. **[verify]**
- Lin, T.-Y., Goyal, P., Girshick, R., He, K., & Dollár, P. (2017). *Focal loss for dense object detection.* ICCV. **[verify]**
- Liu, Z., et al. (2021). *Swin Transformer: Hierarchical vision transformer using shifted windows.* ICCV. **[verify]**
- Loshchilov, I., & Hutter, F. (2019). *Decoupled weight decay regularization (AdamW).* ICLR. **[verify]**
- Ronneberger, O., Fischer, P., & Brox, T. (2015). *U-Net: Convolutional networks for biomedical image segmentation.* MICCAI. **[verify]**

> **To finalise:** fill the MEN ET-absent count in Table 1 from the MEN split once generated;
> confirm exact software/driver versions for §3.10 if a reproducibility appendix is included.
