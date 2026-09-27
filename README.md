# GRACE with Base counterfactuals

Standalone tabular implementation extracted from the verified Task100 experiments.
This repository contains only Wachter Base generation, update-aware GRACE,
classifier fitting, calibration, and future-validity evaluation. It has no dependency
on the original research checkout. Datasets, checkpoints, results, other baselines,
and image/text experiments are not included.

## Files

- `core.py`: Wachter solver, differentiated SGD updates, smooth GRACE surrogate,
  exact Wasserstein geometry, constrained support optimization, and assignment.
- `run.py`: data preprocessing, MLP/logistic fitting, calibration, and evaluation.
- `config.json`: the seed0 experimental settings and search grids.
- `requirements.txt`: dependency versions from the extraction environment.

## Installation

Use Python 3.11 (the extraction environment) and a compatible PyTorch installation.
The default run device is CUDA; `--device cpu` is also supported.

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

## Run

Always supply a **new output directory**. The runner refuses to overwrite an
existing path and does not resume partially completed runs.

```bash
OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 \
python run.py --dataset diabetes --model mlp --device cuda:0 \
  --data-home /path/to/data-cache --output /path/to/new/diabetes_mlp
```

Use `--dataset heloc`, `diabetes`, `german`, or `fico`, and
`--model mlp` (default) or `logistic`. HELOC, Diabetes, and German use OpenML;
the first run requires network access unless their data are already cached.
Diabetes is OpenML37, German is OpenML31, and HELOC is OpenML `heloc` version1.

Mixed FICO uses the original `heloc_dataset_v1.csv`, supplied separately:

```bash
python run.py --dataset fico --model mlp --device cuda:0 \
  --csv /path/to/heloc_dataset_v1.csv --data-home /path/to/data-cache \
  --output /path/to/new/fico_mlp
```

Its target is `RiskPerformance`, with favorable class `Good`.
`MaxDelq2PublicRecLast12M` and `MaxDelqEver` are categorical. Missing codes
`-7`, `-8`, and `-9` are imputed using training data. All-missing rows are retained.
HELOC and mixed FICO are two representations of one corpus, not independent datasets.
For another locally supplied version of a supported dataset, use `--csv` and
`--target-column`; the chosen dataset's label mapping and categorical rules still apply.

Independent jobs may use separate GPUs by setting `CUDA_VISIBLE_DEVICES` and
using distinct output paths. Eight dataset/classifier combinations can run as
separate jobs. This minimal repository has no multi-job scheduler.

## What runs

1. Split train/calibration/test as 60/20/20 with seed0; fit preprocessing on
   training data only. Fit an MLP with hidden widths64/32, or a logistic classifier.
2. Choose up to128 currently unfavorable factuals from each calibration/test
   partition. Retain all classifier-favorable training examples as the reference.
3. Generate one Wachter Base per factual using five restarts and500 steps,
   preserving the original hard-category Base projection and failed constructions.
4. Calibrate GRACE over `r / b_ref = [0, 0.25, 1, 3]` and
   `beta_geo = [1.05, 1.1, 1.2]`. The reference radius is the95th percentile
   of20 bootstrap Wasserstein distances on at most128 calibration examples.
   Select only geometry-feasible configurations; maximize calibration future
   validity, then minimize proximity, radius ratio, and geometry multiplier.
5. Optimize the selected test population and freeze outputs before constructing
   the test future-model bank. Evaluate the fixed outputs without validity repair.

GRACE uses `0.05 * softplus((0.5 - probability) / 0.05)` and the exact training-input
pullback norm through all10 full-batch SGD updates (update learning rate0.01).
Each factual has one equal-mass support; exact OT gives a one-to-one Base-to-Q
assignment. Geometry is `W2(Base,Q) + W2(Q,P1) <= beta_geo * W2(Base,P1)`.
Optimization uses up to800 projected augmented-Lagrangian steps, step size1.0
and penalty coefficient1.0. A feasible small objective change permits early stopping.
The final iterate is retained, including infeasible test outcomes; there is no
convergence certificate or validity correction. Wachter's original Base objective
is retained separately from GRACE's smooth loss.

Calibration and test each use45 mean/scale/tail update scenarios: three
severities0.5/1/2.5 and five realizations. Tail resampling keeps labels paired with
rows. Some scenarios repeat across random seeds and partitions; they are not45
independent distributions. Seed offsets are100000 and200000.

## Outputs and interpretation

`results.csv` contains Base/GRACE future validity (fraction in0--1), proximity
(mean standardized-feature L2), current validity, invalid counts, and geometry
status. `selection.json` records the selected absolute radius, radius ratio,
and beta. `calibration.csv` retains all attempted settings.

The output directory also saves fitted model weights, prepared tensors and split
IDs, actual Base/support/assignment tensors, solver traces, future checkpoints,
resolved configuration, source/data hashes, and the output-freeze timestamp.
These generated files belong outside source control. A no-feasible-calibration
run stops with an explicit status rather than choosing an infeasible configuration.

For German/FICO, optimization uses relaxed categorical simplex supports and
logit-softmax training perturbations. `realization=relaxed` is not legal discrete
recourse. `realization=hard` is a separate argmax diagnostic, with geometry checked
again; hardening can violate the budget. Continuous coordinates are not given
additional box/immutable-feature constraints. Classifier validity alone does not
establish actionability. Failed Base cases remain in metric denominators.

## Provenance and verification

The mathematical routines are extracted unchanged from the original Task100
`appendix_tabular.py`, `mixed_update_aware.py`, `update_aware.py`,
`geodesic_excess.py`, and Wachter helpers in `german_credit_strict_audit.py`.
Only the standalone orchestration and imports are new. This is the single-seed
reconstructed diagnostic protocol, not a verified reproduction of historical Table1.

The extraction is checked for source-equivalent core routines, mixed-domain
projection and gradient correctness, and an end-to-end synthetic CSV run.
Historical experiment outputs are neither bundled nor rerun. Fresh training and
optimization need not be bitwise identical across hardware or library versions.
