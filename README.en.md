# CIFAR-10 KNN: Comparing Distance Metrics, Voting Rules, and k

[한국어](README.md) | **English**

This project implements and evaluates K-Nearest Neighbors (KNN) using all **50,000 training images** and **10,000 official test images** in CIFAR-10. It compares distance metrics, the number of neighbors k, and voting rules using the original 3,072-dimensional RGB pixel vectors.

Distance calculations, exact neighbor search, and voting are implemented directly. scikit-learn is used for cross-validation splits, evaluation metrics, and independent verification. Fitting KNN stores the training examples for subsequent neighbor searches.

## Main Results

The distance metric and k for each voting rule were selected by **mean 5-fold cross-validation accuracy**. The table shows the selected models' official test results. All scores are percentages; precision, recall, and F1 use macro averaging.

| Voting rule | Distance | k | CV accuracy: mean ± SD | Test accuracy | Precision | Recall | F1 |
| --- | --- | --- | --- | --- | --- | --- | --- |
| Uniform majority vote; smallest class ID resolves ties | Manhattan | 1 | 37.08 ± 0.19 | 38.59 | 43.33 | 38.59 | 38.62 |
| Uniform majority vote; nearest neighbor in a tied winning class resolves ties | Manhattan | 5 | 37.85 ± 0.51 | 39.21 | 45.60 | 39.21 | 38.90 |
| Inverse-distance weighted vote | Manhattan | 5 | 38.19 ± 0.50 | **39.50** | **45.92** | **39.50** | **39.22** |

The latest experiment extended the search to k=101 for uniform majority voting with nearest-neighbor tie resolution. The best setting remained **Manhattan, k=5**, with the same **39.21%** test accuracy. Inverse-distance weighted voting was evaluated on the original eight k values.

**[All Results and Comparisons](results/README.md)** · **[Evaluation Metrics](docs/METRICS.md)**

The linked detailed result documents and supporting notes are in Korean. The original CSV files retain English column names and unrounded numerical values.

The core implementation and experiment entry points include Korean explanatory comments. A [code walkthrough in Korean](docs/CODE_WALKTHROUGH.ko.md) explains the call sequence, array shapes, distance calculations, voting, and evaluation flow.

## Experiments

The original eight k values are `1, 3, 5, 7, 9, 11, 15, 21`.

| ID | Experiment | Distances | k values | CV configurations | Fold records | Code / Results |
| --- | --- | --- | --- | --- | --- | --- |
| 01 | Baseline KNN: uniform vote with smallest-class-ID tie resolution | All three | Original eight | 24 | 120 | [Code](experiments/01_baseline/run.py) / [Results](results/01_baseline/README.md) |
| 02 | Voting tie frequency and the effect of nearest-neighbor tie resolution | Manhattan | Original eight | 8 | 40 | [Code](experiments/02_tie_analysis/run.py) / [Results](results/02_tie_analysis/README.md) |
| 03 | Comparison of all three voting rules | All three | Original eight | 72 | 360 | [Code](experiments/03_voting_comparison/run.py) / [Results](results/03_voting_comparison/README.md) |
| 04 | Extended k search with uniform vote and nearest-neighbor tie resolution | All three | Every odd k from 1 to 101: 51 values | 153 | 765 | [Code](experiments/04_k101/run.py) / [Results](results/04_k101/README.md) |

Experiment 02 is a CV-only diagnostic and does not evaluate the official test set. Experiment 03 reruns the baseline voting rule under the same conditions and verifies that its predictions match experiment 01. Configuration counts describe records within each experiment and include repeated settings across experiments.

## Shared Experimental Setup

| Item | Setting |
| --- | --- |
| Data | All 50,000 training images and all 10,000 test images |
| Features | Original 32 × 32 × 3 uint8 pixels flattened into 3,072-dimensional vectors |
| Preprocessing | Original pixels, without subsampling, PCA, learned features, or standardization |
| Distances | Euclidean (L2), Manhattan (L1), and cosine distance |
| Validation | Stratified 5-fold CV, `shuffle=True`, `random_state=42` |
| Each fold | 40,000 training images and 10,000 validation images; each training image is validated once |
| Model selection | Highest mean CV accuracy, then smallest k, then distance name |
| Final evaluation | Select k by CV for each distance, then use all training images to evaluate the official test set |
| Metrics | Accuracy, macro precision, macro recall, and macro F1 |
| CV reporting | Fold mean ± sample standard deviation (`ddof=1`), not a confidence interval |
| Neighbor search | Exact search over all training examples, with query batches of 128 |

Test scores were not used to select each model's distance or k. However, the test results were inspected repeatedly during the assignment, so the highest observed score should not be interpreted as an additional independent evaluation.

## Voting and Tie Resolution

- **`uniform_class_id`**: Every neighbor contributes one vote. If several classes share the highest vote count, the smallest class ID wins.
- **`uniform_nearest`**: Every neighbor contributes one vote. If several classes share the highest vote count, only those tied winning classes are considered. The class of their closest neighbor wins.
- **`distance_weighted`**: Each neighbor contributes the inverse of its actual distance, `1 / d`. Euclidean weights use the actual L2 distance, rather than squared distance. If any selected neighbor has distance zero, only zero-distance neighbors vote, with equal weights. Ties between class weight totals are resolved by the closest neighbor in a tied winning class (`rtol=1e-12`, `atol=0`).

Equal neighbor distances are resolved by stable sorting, placing the lower training index within the current fold first. An odd k can still produce a voting tie in multiclass classification, such as vote counts of `2:2:1`.

## Repository Structure

```text
cifar10-knn/
├── README.md                       # Korean documentation
├── README.en.md                    # English documentation
├── requirements.txt                # CPU execution, metrics, and plotting
├── requirements-gpu.txt            # Optional PyTorch dependency
├── knn_cifar10/
│   ├── core.py                     # Shared KNN implementation and baseline experiment
│   ├── tie_analysis.py             # CV voting tie diagnostic
│   ├── voting.py                   # Shared runner for voting comparisons and extended k
│   └── reporting/                  # Existing PNG and HTML exporters
├── experiments/
│   ├── 01_baseline/                # Baseline entry point and documentation
│   ├── 02_tie_analysis/            # Tie diagnostic entry point and documentation
│   ├── 03_voting_comparison/       # Three-rule comparison entry point and documentation
│   └── 04_k101/                    # Extended k entry point and documentation
├── results/
│   ├── README.md                   # Cross-experiment comparisons
│   ├── comparison.csv             # Selected-model comparison data
│   ├── 01_baseline/                # Markdown tables, CSVs, PNGs, and metadata
│   ├── 02_tie_analysis/
│   ├── 03_voting_comparison/
│   └── 04_k101/
├── scripts/
│   ├── build_result_tables.py      # Saved CSVs to complete Markdown tables
│   └── validate_repository.py      # Numerical, structural, and link checks
├── tests/test_knn.py               # Independent checks on small examples
├── docs/
│   ├── METRICS.md                  # Metric definitions and macro averaging
│   ├── PROVENANCE.md               # Original-to-organized paths and changes
│   └── source_manifest.json        # SHA-256 hashes of copied files
├── data/README.md                  # Dataset preparation instructions
└── runs/                           # New experiment outputs, excluded from Git
```

## Environment

The experiments used Python **3.10.20**, NumPy **1.26.4**, scikit-learn **1.7.2**, and PyTorch **2.6.0+cu124**. GPU runs used one NVIDIA RTX A5000. Other dependency versions are recorded in [requirements.txt](requirements.txt).

Use an existing Python or Conda environment with the required dependencies, and run commands from the repository root. Select `--device cpu` for CPU execution, or `--device cuda:0` when a compatible PyTorch installation and a visible CUDA device are available. `--device auto` uses a visible CUDA GPU if available and otherwise falls back to CPU. CUDA device numbers are logical indices that respect `CUDA_VISIBLE_DEVICES`.

If dependencies are missing, prepare them in your user environment before running the experiments. No packages were installed, and no full-dataset experiments were rerun, while organizing this repository.

## Dataset Preparation

Use the Python version from the [official CIFAR-10 dataset page](https://www.cs.toronto.edu/~kriz/cifar.html). It contains 60,000 color images of size 32×32, split into 50,000 training images and 10,000 test images. The dataset is not included in the repository.

The default archive location is `data/cifar-10-python.tar`. To use another location, specify `--data /path/to/cifar-10-python.tar`. A `.tar.gz` archive is also supported: the loader detects the compression format and reads the required members in memory. See the [dataset instructions](data/README.md) for details.

## Reproducing the Experiments

The commands below **run new experiments**. To inspect existing results, open the [result documents](results/README.md). Run all commands from the repository root.

```bash
# 01. Baseline KNN: full-training-set 5-fold CV and official test evaluation
python experiments/01_baseline/run.py --data data/cifar-10-python.tar --output runs/01_baseline --device cpu

# 02. Analyze voting ties using Manhattan CV predictions
python experiments/02_tie_analysis/run.py --results runs/01_baseline --output runs/02_tie_analysis --device cpu

# 03. Compare three voting rules using identical folds
python experiments/03_voting_comparison/run.py --baseline runs/01_baseline --output runs/03_voting_comparison --device cpu

# 04. Uniform majority vote with nearest-neighbor tie resolution, odd k=1 through 101
python experiments/04_k101/run.py --baseline runs/01_baseline --output runs/04_k101 --device cpu
```

Experiments 02–04 use the `metadata.json` and `predictions.npz` generated by experiment 01 to verify splits and predictions. The published `results/01_baseline` directory is an archive of CSVs, PNGs, and metadata; its NPZ prediction cache is excluded. For reproduction, first run experiment 01 and use the resulting `runs/01_baseline` directory.

An existing output directory causes the runner to stop. For another run, choose a new `--output` path and update subsequent `--baseline` or `--results` arguments accordingly. If the dataset has moved, experiments 02–04 also accept `--data`. Experiments 03–04 verify that the archive hash matches the baseline.

Experiment-specific defaults and commands are documented in [experiments/README.md](experiments/README.md).

## Generating Tables and Validating Results

```bash
# Regenerate complete tables from saved CSVs; does not run a model
python scripts/build_result_tables.py

# Generate tables for newly completed runs of all four experiments
python scripts/build_result_tables.py --results-root runs

# Check numerical consistency, copied files, directory structure, and document links
python scripts/validate_repository.py

# Independently verify distances, voting, and evaluation on small examples
python -m unittest discover -s tests -v
```

GPU-specific tests are skipped in a CPU-only environment. These checks do not rerun the full CIFAR-10 experiments.

The checks performed after organizing the repository are recorded in the [validation notes](docs/VALIDATION.md).

The existing image and HTML exporters can be run as modules. To generate the graphs for new runs of experiments 03–04 and the comparison between k search ranges, run the following commands before generating Markdown tables. The primary entry point for published GitHub results is the Markdown documentation.

```bash
python -m knn_cifar10.reporting.voting_report runs/03_voting_comparison
python -m knn_cifar10.reporting.extended_report runs/04_k101 --previous runs/03_voting_comparison
```

## Reading the Result Files

| File | Contents |
| --- | --- |
| `cv_results.md` | All CV means ± standard deviations, grouped by distance, k, and voting rule |
| `fold_results.md` | Every individual fold result |
| `test_results.md` | Official test results for the CV-selected k of each distance |
| `class_results.md` | Per-class precision, recall, and F1 for the selected models |
| `confusion_matrices.md` | Confusion matrices: true classes in rows and predicted classes in columns |
| `comparison.md` | Comparison of CV-selected models for voting rules, where applicable |
| `tie_results.md` | Tie frequency and accuracy changes for the tie diagnostic |
| `range_comparison.md` | Comparison of the original k grid and the search through k=101 |
| `*.csv` | Original numerical values; performance metrics are on a 0–1 scale |
| `metadata.json` | Original settings, environment, model selection, and execution records |

## GitHub Upload Contents

The repository includes code, README files, Markdown tables, CSVs, metadata, and PNG figures. `.gitignore` excludes dataset archives, `runs/`, NPZ/NPY caches, ZIP files, logs, environments, and cache directories. The mapping between original experiment files and organized files is documented in [PROVENANCE.md](docs/PROVENANCE.md).

## Dataset Reference

Alex Krizhevsky, *Learning Multiple Layers of Features from Tiny Images*, 2009. See the [official CIFAR dataset page](https://www.cs.toronto.edu/~kriz/cifar.html) for the dataset description and reference.
