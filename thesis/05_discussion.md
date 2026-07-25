# 5. Discussion

> **Drafting notes (remove before submission).** English, first-person plural. This chapter
> interprets the results of Chapter 4 and relates them to the hypotheses (§1.3), the biology
> (§2.1), and the state of the art (§2.7). It deliberately foregrounds the study's
> limitations and the evaluation caveat, in the interest of an honest and defensible account.

In this chapter we interpret the experimental findings. We consider, in turn, the differences
in performance across the three tumour populations and their biological basis (§5.1); the
enhancing-tumour gap that is the study's central empirical finding (§5.2); the unexpected
architecture result and what it reveals about data scale (§5.3); the contribution of transfer
learning (§5.4); the population-dependent divergence between voxel and lesion-wise scoring
(§5.5); the relationship of our results to the state of the art, with the necessary
evaluation caveat (§5.6); the clinical relevance of the observed error patterns (§5.7); and
the limitations of the study (§5.8).

## 5.1 Cross-population Performance and its Biological Basis

The same methodology produces markedly different performance across the three populations:
five-fold cross-validated full-volume voxel Dice of 0.903 for meningioma and 0.718 for the
paediatric set, with adult glioma at 0.907 (three completed folds). This ordering is not an artefact of the method but a direct reflection of
tumour biology and dataset scale. Adult gliomas and meningiomas are imaged in large,
relatively homogeneous adult cohorts (≈1,250 and ≈1,000 cases respectively) and present with
substantial, well-enhancing tumour tissue, whereas the paediatric population is both far
smaller (99 cases) and biologically harder, frequently lacking a clear enhancing component.
That a single pipeline nevertheless reaches ~0.90 Dice on the two large adult populations, and
a clinically non-trivial 0.72 on the paediatric set, supports H1: a unified methodology, with only
modest per-challenge adaptation, transfers across biologically distinct tumours. The success
is most complete for the coarser regions — whole-tumour Dice remains above 0.80 even for PED
— indicating that the methodology generalises well for gross tumour delineation and that the
residual difficulty is concentrated in the fine, biologically variable sub-regions.

## 5.2 The Enhancing-Tumour Gap

The clearest empirical finding of the thesis is the population dependence of enhancing-tumour
segmentation. Under an identical method, ET Dice is 0.909 (MEN) and 0.520 (PED), with GLI 0.875. The paediatric collapse is not a general failure of the model —
whole-tumour Dice for PED is 0.831 — but is specific to enhancing tissue, and it is driven by
the biology of paediatric diffuse midline glioma, in which the enhancing component is
frequently small or entirely absent (11 of 99 PED cases contain no ET at all). The high
variance of the PED ET score (±0.36) confirms that performance is bimodal: the model segments ET well when it is
present and substantial, and scores near zero on the many cases where ET is minimal or absent
and any prediction is heavily penalised. This directly supports H3 and, importantly,
reframes the paediatric difficulty as a problem of the enhancing sub-region and of data
scarcity rather than of the paediatric brain *per se* — a distinction with direct
methodological consequences, since it points to ET-specific remedies (threshold tuning, the
ET-suppression post-processing of §3.9, and biologically informed label redefinition) rather
than to a wholesale change of approach.

## 5.3 Architecture and Data Scale

The architecture comparison — a controlled single-fold experiment, indicative rather than
cross-validated — produced our most nuanced finding: the better architecture
*depends on the amount of data*. On the small paediatric set the purely convolutional 3D
U-Net outperformed the hybrid Swin UNETR across every region (mean Dice 0.745 vs 0.689) while
using fewer parameters; but on the large adult-glioma set the ranking reversed, with Swin
UNETR edging ahead overall (0.913 vs 0.904) and most clearly on the enhancing tumour (0.884
vs 0.864). We interpret this through the lens of data scale. Transformer-based models replace
convolution's strong locality and translation-equivariance priors with a more flexible but
less constrained self-attention mechanism; this flexibility is an advantage only when
sufficient data are available to learn what the convolutional prior supplies for free. With
just 77 paediatric training cases the inductive bias of the convolutional U-Net is decisive,
whereas with roughly a thousand adult cases the transformer's additional capacity finally
justifies itself. This reading is reinforced by the transfer result (§5.4): in the paediatric
setting the from-scratch U-Net (0.745) marginally exceeds even the GLI-pretrained Swin model
(0.736), so the architectural prior contributes more than cross-population pre-training when
data are scarce. We therefore **reject H2 in the small-data regime and accept it in the
large-data regime**, and read the two results together as a single, more useful statement:
convolutional inductive bias wins when data are scarce, and the transformer becomes
competitive — indeed marginally superior — only at scale. The practical corollary for a study
spanning populations of very different sizes is that no single architecture is uniformly
best; the appropriate choice tracks the size of the training set.

## 5.4 Transfer Learning

Warm-starting the paediatric model from the GLI-trained weights improved mean Dice by 0.046
(0.689 to 0.736; single-fold controlled comparison) and improved every individual region
while reducing variance. The
improvement is consistent with the hypothesis that the adult-glioma encoder learns
general-purpose representations of brain tissue and tumour appearance that transfer to the
paediatric domain, giving the small paediatric dataset an effective head start; the
substantially lower first-epoch training loss of the warm-started model (1.15 versus 3.34)
makes this head start concrete. Transfer was, however, a smaller intervention than the change
of architecture, which suggests that for this problem the two are complementary rather than
substitutable: the most promising configuration we did not exhaust would combine the
convolutional architecture with cross-population pre-training. This supports H4 and identifies
the single most effective lever we found for the data-scarce paediatric setting.

## 5.5 Voxel versus Lesion-wise Scoring

A methodologically important observation is that the divergence between voxel and lesion-wise
Dice is itself population-dependent, and that its ordering mirrors tumour-shape regularity:
the gap is smallest for meningioma (0.065), intermediate for adult glioma (0.117),
and largest for the paediatric set (0.255). Meningiomas are convex and well-circumscribed, so the model's
predictions form a single clean component that the lesion-wise metric rewards as fully as the
voxel metric does. Gliomas are more irregular and occasionally fragment. Paediatric
predictions, on the background-dominated full volume, scatter into multiple false-positive
components and split single tumours into pieces, all of which the lesion-wise metric — which
penalises each spurious or missed lesion individually — correctly punishes. This has two
consequences. First, it cautions strongly against reporting voxel Dice alone, since the
voxel metric most flatters performance precisely where the predictions are least clinically
clean. Second, it localises where post-processing will pay off: connected-component filtering
and threshold tuning recovered part but not all of the paediatric gap in our experiments, and
the residual gap is an explicit target for the extensions of §3.9.

## 5.6 Relationship to the State of the Art

A careful comparison with the published BraTS 2023 results requires the evaluation caveat of
§3.8 to be kept firmly in view. Our headline cropped voxel Dice values (GLI mean 0.913) may
appear to approach or exceed the reported winning scores, but this comparison is not valid:
the official ranking uses the lesion-wise metric on full-resolution volumes of a hidden test
set, whereas our cropped voxel figures are computed on a held-out fold of the training data
in a brain-centred crop. The honest comparison uses our full-volume lesion-wise numbers. On
that basis our GLI model attains lesion-wise Dice of 0.755 / 0.826 / 0.790 for WT / TC
/ ET (three-fold cross-validated), against the GLI
challenge winner's reported ~0.90 / 0.87 / 0.85. Our model is therefore
a few points behind the winning system on the metric that matters — an entirely expected and,
we argue, encouraging outcome, given that the winning system was a three-architecture
ensemble augmented with GAN-generated synthetic data and trained on multi-GPU
infrastructure, whereas ours is a single model trained on one consumer GPU without ensembling,
test-time augmentation, or synthetic data. The gap is, in effect, a measure of exactly the
techniques we deliberately excluded, and it quantifies what an individual researcher forgoes
by operating within a single-GPU budget — which is precisely the question the thesis set out
to answer.

## 5.7 Clinical Relevance of the Error Modes

The error modes we observe carry different clinical weights. False-positive lesions in
healthy tissue, which dominate the paediatric voxel-to-lesion-wise gap, are clinically
costly: a spurious enhancing focus could prompt unnecessary investigation or alter management,
which is why the lesion-wise metric — and the post-processing that improves it — is clinically
the right target. Missed or under-segmented enhancing tumour in paediatric DMG, the other
characteristic failure, reflects a genuinely hard and clinically recognised problem rather
than a modelling oversight, and the biologically informed strategy of redefining near-absent
ET (§3.9) aligns the model's behaviour with clinical reality. For meningioma, the near
agreement of voxel and lesion-wise scores indicates predictions that are not only accurate but
clinically clean, the most directly usable of the three.

## 5.8 Limitations

Several limitations bound the conclusions. First, the principal results for meningioma and the
paediatric set are five-fold cross-validated over their entire cohorts, whereas the
adult-glioma results are reported over three of the five folds (fold 4 could not be trained
within the schedule); given GLI's low inter-fold variance (per-fold means 0.902–0.916) three
folds already provide a stable estimate. Second, the architecture and
transfer-learning experiments (§5.3–§5.4) are single-fold controlled comparisons rather than
cross-validated, so their effect sizes should be read as indicative; the architecture
comparison also spans only two of the three data scales (paediatric and adult glioma, not
meningioma). Third, the headline pipeline is a single model without the ensembling, test-time
augmentation, or threshold optimisation used by competitive systems; these were deliberately
scoped out, but their absence means our numbers are a lower bound on what the methodology can
achieve. Fourth, a component ablation — which would quantify the marginal contribution of
normalisation, augmentation, the compound loss, and post-processing — was descoped for want of
GPU time within the project schedule and is left to future work (§6). Fifth, the lesion-wise
metric we report is our own implementation following the BraTS 2023 definition rather than the
official scoring container, so while it is internally consistent for our comparisons, small
differences from the official figures are possible. Sixth, cross-hardware reproducibility was
verified on the paediatric and meningioma challenges — where a consumer RTX 4070 and a
data-centre NVIDIA L4 agreed to within 0.004 Dice on matched folds — but not on adult glioma,
because the L4, an efficiency-class inference GPU roughly 2.4× slower per sample than the
desktop card, could not complete the glioma sweep within the schedule. Finally, the entire
study is constrained by its single 12 GB GPU, which dictated the patch-based training, the
small batch size, and the omission of heavier techniques; this constraint is, however, also the
point of the thesis, and the results should be read as a demonstration of what is achievable
under it rather than as an attempt to top the leaderboard.

---

> **To finalise (13–14 Jul):** replace the provisional GLI figures in §5.1, §5.2, §5.5, §5.6
> and the glioma clause in §5.8 with the five-fold GLI aggregate; keep §5.3–§5.4 flagged as
> single-fold controlled comparisons.
