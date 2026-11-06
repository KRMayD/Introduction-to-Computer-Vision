# CIFAR-10 KNN

[한국어](README.md) | **English**

Exact KNN experiments using all 50,000 CIFAR-10 training images and all 10,000 official test images. The implementation consists of two files, both with Korean comments.

| File | Purpose |
| --- | --- |
| [knn.py](knn.py) | Load data, compute distances, find neighbors, vote, and evaluate |
| [experiment.py](experiment.py) | Choose an experiment, run 5-fold CV, select models, and evaluate the official test set |
| [results/](results/README.md) | Existing complete tables, CSV files, and figures |

## Experiment setup

Images are flattened into 3,072 raw RGB pixel features without subsampling, standardization, or PCA. Distances and voting rules are implemented explicitly; SciPy provides the CPU L1 calculation, and scikit-learn provides splits and evaluation metrics.

- Distances: Euclidean (L2), Manhattan (L1), and cosine.
- Default k values: `1, 3, 5, 7, 9, 11, 15, 21`.
- Validation: stratified 5-fold CV with shuffle and seed 42; 40,000 training and 10,000 validation images per fold.
- Selection: highest mean CV accuracy, then smaller k, then metric name. Test labels do not affect selection.
- Final evaluation: choose k for each voting rule and metric, fit on all 50,000 training images, and evaluate all 10,000 test images.
- Scores: accuracy and macro precision, recall, and F1. CV reports fold means and sample standard deviations (`ddof=1`).

| `--experiment` | Scope | Existing results |
| --- | --- | --- |
| `baseline` | 3 metrics × 8 default k values; class-ID tie breaking | [01](results/01_baseline/README.md) |
| `ties` | Manhattan × 8 default k values; compare two uniform-vote tie rules; CV only | [02](results/02_tie_analysis/README.md) |
| `voting` | 3 metrics × 8 default k values × 3 voting rules | [03](results/03_voting_comparison/README.md) |
| `k101` | 3 metrics × odd k values from 1 to 101; nearest-neighbor tie breaking | [04](results/04_k101/README.md) |

## Voting rules

- `uniform_class_id`: one vote per neighbor; ties select the smallest class ID.
- `uniform_nearest`: one vote per neighbor; ties select the nearest neighbor belonging to a tied winning class. Odd k can still produce multiclass ties such as `2:2:1`.
- `distance_weighted`: sum `1/d` using actual metric distances. If any neighbors have zero distance, only those neighbors vote with equal weights. Weight ties (`rtol=1e-12`) use the nearest tied candidate.

Equal-distance neighbors retain training-index order. Neighbor search is exact and batched.

## Run an experiment

[requirements.txt](requirements.txt) lists the Python 3.10 dependencies. CPU execution does not require PyTorch. For CUDA execution, use CUDA-enabled PyTorch in your existing environment.

Provide a trusted CIFAR-10 Python `.tar` or `.tar.gz` archive. The dataset is not included in this repository.

```bash
python experiment.py --experiment voting --data /path/to/cifar-10-python.tar --output runs/voting_01 --device cpu
```

Select `baseline`, `ties`, or `k101` to run another experiment. Every preset runs independently without previous prediction caches. Choose `--device cuda:0` or `auto` to use a visible GPU; `CUDA_VISIBLE_DEVICES` is respected. Exact search over the full dataset can take a long time on CPU. Defaults are batch size 128 and four CPU threads. See `python experiment.py --help` for options.

Outputs: `cv_folds.csv`, `cv_summary.csv`, `selected_models.csv`, and `test_results.csv`. The `ties` preset adds `tie_folds.csv` and `tie_summary.csv` instead of selection and test files. Scores are fractions from 0 to 1. Existing output directories are never overwritten. Plotting, report generation, and cache export are excluded from the execution code.

## Existing results

Values are percentages. Precision, recall, and F1 use macro averaging. Each rule's metric and k were selected by mean CV accuracy.

| Voting rule | Metric | k | CV accuracy | Test accuracy | Precision | Recall | F1 |
| --- | --- | --- | --- | --- | --- | --- | --- |
| Class-ID tie breaking | Manhattan | 1 | 37.08 ± 0.19 | 38.59 | 43.33 | 38.59 | 38.62 |
| Nearest-neighbor tie breaking | Manhattan | 5 | 37.85 ± 0.51 | 39.21 | 45.60 | 39.21 | 38.90 |
| Inverse-distance weighting | Manhattan | 5 | 38.19 ± 0.50 | 39.50 | 45.92 | 39.50 | 39.22 |

Extending nearest-neighbor tie breaking through k=101 retained the same optimum, k=5. [Complete tables and figures](results/README.md) come from the previous experiment runs; full experiments were not rerun during this refactor. Earlier code and documentation remain available in Git history.

Validation covered all four pipelines on small data and exact agreement with 11,370,000 saved predictions across 237 CV/test configurations, recomputed from saved neighbors. Actual GPU execution was not part of this validation.
