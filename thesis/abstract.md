# Abstract / Apstrakt

> **Drafting notes (remove before submission).** Front matter. The English **Abstract** is the
> primary version; the Serbian **Rezime** is a faithful translation that **must be reviewed by
> a native Serbian speaker / the mentor** for medical terminology (in particular the rendering
> of "enhancing tumour"). ~230 words each. Numbers are rounded fold-0 results.

---

## Abstract (English)

Accurate segmentation of brain tumours from multi-parametric magnetic resonance imaging (MRI)
underpins diagnosis, treatment planning, and monitoring, yet manual delineation is slow and
subject to inter-observer variability. The BraTS 2023 challenge introduced three biologically
distinct populations — adult glioma (GLI), intracranial meningioma (MEN), and paediatric
high-grade glioma (PED). This thesis investigates whether a single, unified deep-learning
methodology, with only modest task-specific adaptation, can segment all three competitively
within the means of an individual researcher using one consumer-grade GPU (an NVIDIA RTX 4070,
12 GB). We implement a reproducible, patch-based pipeline — a Swin UNETR hybrid model and a 3D
U-Net baseline — with mixed-precision training and stratified five-fold cross-validation, and
evaluate it with voxel-level and lesion-wise Dice and 95th-percentile Hausdorff distance in
native image space. The unified method reaches a mean full-volume voxel Dice of approximately
0.92 (GLI), 0.90 (MEN), and 0.69 (PED). Enhancing-tumour difficulty is strongly
population-dependent (Dice 0.88 / 0.91 / 0.50), reflecting the frequently absent enhancement
of paediatric diffuse midline glioma. On the small paediatric dataset the convolutional 3D
U-Net outperforms Swin UNETR (0.745 vs 0.689) with fewer parameters, and GLI→PED transfer
learning improves every region (+0.046). The divergence between voxel and lesion-wise scoring
is itself population-dependent, tracking tumour-shape regularity. On the official-style
lesion-wise metric the single model trails challenge-winning ensembles by a few points,
quantifying the cost of the single-GPU, no-ensemble setting. The work demonstrates that
competitive, reproducible, multi-population brain-tumour segmentation is achievable on
accessible hardware.

**Keywords:** brain tumour segmentation; BraTS 2023; deep learning; Swin UNETR; 3D U-Net;
transfer learning; magnetic resonance imaging; medical image analysis.

