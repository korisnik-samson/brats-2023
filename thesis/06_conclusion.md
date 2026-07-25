# 6. Concluding Remarks

> **Drafting notes (remove before submission).** Per the Singidunum template this chapter
> contains **no references** and introduces **no new data**: it summarises the study's own
> findings and gives recommendations for further research, in roughly half to one page. First
> person plural.

In this thesis we asked whether a single, unified deep-learning methodology, with only modest
task-specific adaptation, can produce competitive and clinically meaningful brain-tumour
segmentation across three biologically distinct populations — adult glioma, meningioma, and
paediatric high-grade glioma — within the means of an individual researcher using a single
consumer-grade GPU. Our results answer this question in the affirmative, while also revealing
where the approach succeeds easily and where it does not.

We established a reproducible, single-GPU pipeline and trained it identically on all three
populations. It reached a mean full-volume voxel Dice of approximately 0.92 for adult glioma
and 0.90 for meningioma, and a lower but clinically non-trivial 0.69 for the paediatric set,
confirming that one methodology transfers across tumour types for gross tumour delineation.
The study's central empirical finding is that the difficulty is concentrated in the enhancing
sub-region and is population-dependent: enhancing-tumour Dice fell from around 0.89–0.91 in
the adult populations to 0.50 in the paediatric set, driven by the frequent absence of an
enhancing component in paediatric diffuse midline glioma rather than by any general failure of
the method. We further found that the divergence between voxel and lesion-wise scoring is
itself population-dependent, tracking the regularity of tumour shape, which argues against
reporting voxel overlap alone.

Two further findings concern how best to improve the hardest, data-scarce paediatric setting.
First, contrary to our expectation, the purely convolutional 3D U-Net outperformed the hybrid
transformer model on the small paediatric dataset, and with fewer parameters — consistent with
the greater data requirements of transformer architectures. Second, warm-starting the
paediatric model from the adult-glioma model improved every region; transfer learning and the
convolutional architecture emerged as complementary, mutually reinforcing levers. Throughout,
we were careful to distinguish our internal voxel metrics from the official lesion-wise
metric: on the latter our single model trails the challenge-winning ensembles by a few points,
a gap that measures precisely the ensembling, synthetic data, and multi-GPU scale that the
single-researcher setting forgoes.

Several directions follow naturally from this work. The immediate priorities are to complete
the five-fold cross-validation with test-time model ensembling and test-time augmentation,
and to add per-region threshold optimisation and the paediatric enhancing-tumour suppression
post-processing, all of which our analysis identifies as targeted remedies for the
false-positive and fragmentation errors that most affect the lesion-wise score. The most
promising single configuration we did not exhaust is the combination of the convolutional
architecture with cross-population pre-training. Completing the architecture comparison on the
large adult-glioma dataset would establish whether the convolutional advantage is specific to
small data or general, turning the architecture finding into a clean statement about the
interaction between architecture and data scale. Finally, evaluating the pipeline with the
official lesion-wise tool and submitting to the held-out challenge test set would place the
results on a fully comparable footing with the published state of the art. Taken together,
these steps would build on the central contribution of this thesis: a transparent, fully
reproducible demonstration that competitive multi-population brain-tumour segmentation is
achievable on accessible, consumer-grade hardware.
