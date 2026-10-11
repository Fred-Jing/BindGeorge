# Primary benchmark protocol

The table in `results/main_results.tsv` compares 18 entries on the same
2,881 examples with
756 positives. The gate is the minimum AF3-reported target-binder chain-pair
PAE minimum <= 2.5 angstroms, not interface-averaged PAE.

## Evaluation

Exact binder sequences, sequence clusters and available parent/scaffold
identifiers determine leakage groups. Targets can be represented on both sides
of a split: this is not target-held-out validation. The original 590-example
validation partition informed development; the original 606-example test was
used once for frozen dMaSIF confirmation. Neither supplies the primary OOF
estimates, and that test is no longer an unseen endpoint.

For each of five grouped outer folds, neural fits use three fitting folds and
one early-stop fold, `(outer_fold + 1) % 5`. Average probabilities across seeds
20260823 and 20260824 before computing pooled metrics. Neural optimization uses
BCE with logits, AdamW, learning rate 3e-4, weight decay 1e-3, gradient clip 1,
at most 50 epochs and patience 8 on early-stop average precision.

AF2/AF3/ESMC runs use BF16 autocast; atom/surface/geometry runs use FP32.
All primary neural runs use H100. These are matched populations and evaluation
conditions, not identical precision, capacity or algorithms.

Exceptions are explicit:

- ipSAE_min is a standalone scalar, without fitted parameters.
- confidence3 standardizes ipTM, negative iPAE_min and mean pLDDT within each
  training partition and uses logistic regression fitted on four non-outer folds.
  The historical neural `confidence` variant is not this primary baseline.
- Rosetta uses 39 raw-structure features, median imputation and histogram
  gradient boosting on three fitting folds, 200 fixed iterations, learning
  rate 0.05, 15 leaves, minimum leaf size 20, L2=1 and balanced class weights.
  Its two seed fits are not independent replications if deterministic.
- dMaSIF multiview averages global and core/rim predictions with fixed 1:1
  weights, giving four members per outer-fold prediction, not two.

AUPRC is non-interpolated average precision. AUROC is secondary. Paired
uncertainty resamples 43 target-sequence clusters, retaining all rows in each
selected cluster. Folds and seeds are not independent datasets. Confidence
intervals are not stored in the aggregate score table and must not be invented
from its point estimates.

## Interpretive limits

The benchmark measures information accessible to the tested compact readouts,
not total information in a tensor. A heterogeneous, negatively sampled,
confidence-selected dataset and repeated development limit generalization.
OOF prediction does not undo study-level adaptive selection. The data do not
resolve a unique best AF3 tensor, nor establish irrelevance of geometry.
No prospective binding rate or calibrated deployment probability is claimed.
