# Game embedding and clustering experiments

This pipeline builds four game representations and fits K-means for each at
`k = 2, 3, 5, 8, 13, 21, 34`. That is **four embedding matrices and 28 clustering
configurations per random seed**. Changing `k` changes the clusters, not the
underlying embedding.

Run commands below from the project root, the folder containing
`games_cleaned.csv`, `games_descriptors.csv`, and `clustering/`.

The code addition to Git does not include these two input CSVs, generated
models, feature matrices, or encoder caches. Supply the CSVs locally in the
project root, or use `--cleaned PATH` and `--descriptors PATH` to read them from
another folder. When moving between checkouts, `--output-dir PATH` and
`--cache-dir PATH` can reuse existing compatible results in their original
location.

## Representations

| Combination | Existing descriptors | Description | Popularity | Expected dimensions |
|---|---:|---:|---:|---:|
| A | 100% | — | — | 509 |
| B | 6/7 (85.7%) | — | 1/7 (14.3%) | 510 |
| C | 2/3 (66.7%) | 1/3 (33.3%) | — | 893 |
| D | 60% | 30% | 10% | 894 |

Dimensions assume the supplied 509 descriptor columns and the default
384-dimensional `sentence-transformers/all-MiniLM-L6-v2` description model.
The pipeline uses the descriptor file's actual columns and the encoder's
actual dimension rather than padding to these illustrative sizes.

- **Descriptors:** reuse the existing numerical values from
  `games_descriptors.csv`. Positional weighting is already applied. No second
  positional weighting, source prefixes, or alias merging is introduced.
- **Descriptions:** clean the text from `games_cleaned.csv`, encode it in
  batches, and cache the result for reuse by both C and D. The default model
  truncates at the configured maximum sequence length. Use
  `--chunk-descriptions` to encode long descriptions in chunks instead.
- **Popularity:** reuse `Popularity Bin`, ordered as `Zero`, `Very low`, `Low`,
  `Moderate`, `High`, `Very high`, `Extreme`, and encode it as `0/6` through
  `6/6`. Do not transform or bin `Peak CCU` again.

Descriptor rows and description rows are L2-normalized separately. Each block
is multiplied by the **square root** of its weight before concatenation. For
D, for example:

```text
features = [sqrt(0.6) × normalized_descriptors,
            sqrt(0.3) × normalized_description,
            sqrt(0.1) × popularity]
```

This makes squared Euclidean distance a weighted sum of the block distances.
There is **no additional normalization of the final concatenated vector**.
The percentages are chosen experimental coefficients, not measured shares of
recommendation quality or explained variance. A zero descriptor block remains
zero. C and D reject missing descriptions and descriptions that are empty
after cleaning; they do not fabricate replacement text or vectors.

## Install

Python 3.12 is the verified interpreter for this project. Open the project
folder in VS Code, then choose **Terminal > New Terminal**. Check that the
terminal is in the project folder before running commands; do not run setup
from `C:\Windows\system32`. Copy the commands inside code blocks, without the
triple-backtick formatting lines.

Create the environment once. If `.venv-clustering` already exists with the
dependencies installed, skip installation and use the Run commands below.
In PowerShell:

```powershell


For A and B only, install the core dependencies; no text model is needed:

```powershell
.\.venv-clustering\Scripts\python.exe -m pip install -r clustering/requirements-core.txt
```

For all four combinations, install the full dependencies:

```powershell
.\.venv-clustering\Scripts\python.exe -m pip install -r clustering/requirements.txt
```

The dependency files define supported version ranges, not a locked environment.
Installed versions are recorded with experiment metadata. Keep that metadata
when comparing results. `requirements-verified.txt` is the `pip freeze` snapshot
of the **Python 3.12.14** environment used to verify this pipeline. To recreate
that tested dependency environment instead of resolving the ranges again:

```powershell
.\.venv-clustering\Scripts\python.exe -m pip install -r clustering/requirements-verified.txt
```

## Run

Inspect the plan without fitting models:

```powershell
.\.venv-clustering\Scripts\python.exe -m clustering.run_all --dry-run
```

Run A and B:

```powershell
.\.venv-clustering\Scripts\python.exe -m clustering.run_all --combinations A B
```

Run the complete experiment:

```powershell
.\.venv-clustering\Scripts\python.exe -m clustering.run_all --combinations A B C D --clusters 2 3 5 8 13 21 34
```

Run one combination through its own entry point:

```powershell
.\.venv-clustering\Scripts\python.exe -m clustering.A.train
.\.venv-clustering\Scripts\python.exe -m clustering.B.train
.\.venv-clustering\Scripts\python.exe -m clustering.C.train
.\.venv-clustering\Scripts\python.exe -m clustering.D.train
```

Execute commands sequentially when they share an output or cache directory.
`run_all` already processes its requested combinations sequentially.

Start with a small smoke run if needed:

```powershell
.\.venv-clustering\Scripts\python.exe -m clustering.run_all --combinations A B --limit 500 --clusters 2 3 --output-dir clustering/outputs-smoke --cache-dir clustering/cache-smoke
```

`--limit` selects one deterministic shared sample using `--sample-seed`. It is
for checking execution, not reporting the final full-dataset experiment. Keep
sampled and full outputs in separate folders.

For repeated initializations across independent fits:

```powershell
.\.venv-clustering\Scripts\python.exe -m clustering.run_all --seeds 42 43 44
```

Three seeds produce 84 fits across the same 28 configurations. `--n-init 10`
additionally performs ten K-means initializations within each individual fit;
it is not ten separately saved experimental runs.

## Inputs and common options

The defaults use the two project-root CSVs, `clustering/outputs` for results,
and `clustering/cache` for shared cached data. Games are associated by `AppID`,
not by assuming that source row positions always match. Use the same games
for every combination.

| Option | Purpose |
|---|---|
| `--cleaned PATH` | Cleaned game CSV |
| `--descriptors PATH` | Weighted descriptor CSV |
| `--output-dir PATH` | Experiment output directory |
| `--cache-dir PATH` | Shared cache directory |
| `--combinations A B C D` | Choose representations in `run_all` |
| `--clusters 2 3 5 8 13 21 34` | Cluster counts to fit |
| `--seeds 42` | One or more fit seeds |
| `--n-init 10` | K-means initializations per fit |
| `--max-iter 300` | Maximum K-means iterations |
| `--tol 0.0001` | K-means convergence tolerance |
| `--threads 8` | Limit numerical-library worker threads |
| `--limit N` | Use a deterministic sample of N games |
| `--sample-seed 42` | Seed for shared sample selection |
| `--description-model MODEL_OR_PATH` | SentenceTransformer model ID or local model folder |
| `--model-revision REVISION` | Pin a model revision for reproducibility |
| `--batch-size 64` | Description encoding batch size |
| `--device cpu` | Encoder device; for example `cpu` or `cuda` |
| `--max-seq-length 256` | Encoder token limit |
| `--chunk-descriptions` | Encode long texts in chunks rather than truncating |
| `--description-embeddings PATH` | Reuse a compatible precomputed `.npy` matrix |
| `--offline` | Require a locally available encoder; prohibit model downloads |
| `--metrics` | Also compute DB index and sampled silhouette |
| `--silhouette-sample-size 5000` | Size of the shared evaluation sample |
| `--dry-run` | Inspect the plan without training |
| `--force` | Rebuild feature/model artifacts and permit replacing an incompatible description cache |

The first C/D run may download the description model. Subsequent compatible
runs reuse its embeddings. For offline use, obtain the model in advance and
pass its local folder with `--description-model` and `--offline`, or provide
precomputed embeddings. A/B do not need model downloads. Pin a specific model
revision with `--model-revision` when reproducibility across machines matters;
the cache also records a resolved revision when the encoder exposes one.

External description embeddings must be an `N × D` `.npy` matrix in the
pipeline's selected AppID order, accompanied by a same-stem JSON metadata file
(for example `descriptions.npy` and `descriptions.json`). Metadata binds the
matrix to the AppID order, cleaned description text, encoder identity and
revision, maximum sequence length, pooling method, normalization, and output
dimension. Reuse the metadata written by the pipeline rather than inventing
hash values. A shape match alone does not establish compatibility.

## Results and resuming

The default output structure is:

```text
clustering/outputs/
  game_index.csv                 AppID and game name in matrix order
  app_ids.npy                    The same ordered AppIDs
  data_manifest.json             Source and preprocessing provenance
  evaluation_sample_appids.csv   Fixed sample for later silhouette calculation
  experiment_plan.json           Requested combinations, cluster counts, seeds
  run_summary.json               Completed fits and their artifact locations
  A/
    features.npy                 Weighted feature matrix
    features.json                Dimensions, settings, fingerprint, checksum
    k_02/
      seed_42/
        model.joblib             Fitted K-means model
        centers.npy              Cluster centres in weighted feature space
        labels.csv               AppID-to-cluster assignments
        manifest.json            Fit settings, provenance, timing, checksums
    k_03/ ... k_34/
  B/ ... D/                      Same structure for the other combinations
```

Description embeddings and their JSON metadata live in `clustering/cache/`.
C and D reuse the same compatible description matrix.

Finished compatible results are reused. Source hashes, AppID order, feature
settings, and model provenance must agree before an existing artifact can be
reused. A mismatch fails clearly; choose a new output/cache directory or use
`--force` deliberately to rebuild the feature/model artifacts. A fully valid
description cache is still reused with `--force`; the flag permits replacement
of an incompatible or incomplete description cache. Interrupted or incomplete
work is not treated as a completed fit.

Only load Joblib model files produced by a trusted source; model loading can
execute Python code.

## Evaluation

Metrics are off by default because the first task is preparing and fitting
the 28 configurations. Add `--metrics` when ready:

```powershell
.\.venv-clustering\Scripts\python.exe -m clustering.run_all --metrics
```

Or evaluate existing saved fits later without retraining:

```powershell
.\.venv-clustering\Scripts\python.exe -m clustering.evaluate --output-dir clustering/outputs
```

This writes `metrics.csv`, `metrics_summary.csv`, `metrics_manifest.json`, and
`metrics_sample_appids.csv`. Optional `--combinations`, `--clusters`, and
`--seeds` restrict the evaluated fits. `--sample-size N` overrides the saved
sample with one deterministic sample shared across the selected fits.

Davies–Bouldin index is better when lower; silhouette is better when higher.
Silhouette uses a reproducible sample shared across fits, avoiding a full
pairwise distance matrix for the entire catalogue. An undefined metric—for
example, a sample containing only one cluster—should be reported as such,
not replaced with a fabricated score.

Compare results within each feature combination and inspect the games in the
clusters. The best-supported `k` is selected from the tested candidates;
these scripts do not establish a universally optimal cluster count. Scores
from different weighted feature spaces also need interpretation alongside
cluster meaning and later recommendation relevance.

This is **flat K-means clustering** of game representations. Constructing a
Mapper graph additionally requires a lens, overlapping cover, local clusters,
and edges based on shared game memberships.

## Verification performed

All **28 configurations completed on all 114,958 games**, using seed 42,
`n_init=10`, and real MiniLM description embeddings. The saved matrices have
509, 510, 893, and 894 dimensions for A, B, C, and D respectively.

Post-training verification checked source and artifact hashes, feature values,
AppID/name alignment, cluster counts, cluster sizes, all saved labels against
each reloaded estimator, and predictions for the same 512-game sample across
all 28 models. All checks passed. No fit emitted a warning or reached the
maximum iteration count.

The small reports in `validation/full_run_verification.json` and
`validation/run_summary.json` record this completed run. The generated models
and feature matrices stay in the local output directory and are not included
in the Git code addition. DB index and silhouette for the full experiment have
not yet been calculated; no optimal `k` has been selected.

All 22 automated tests passed, covering descriptor alignment, feature weights,
cache provenance, and embedding validation. To run them again:

```powershell
.\.venv-clustering\Scripts\python.exe -m unittest discover -s clustering/tests -v
```

An earlier 512-game smoke run also checked deferred evaluation. Its scores were
verification data rather than full-catalogue findings; the smoke artifacts were
removed before the full experiment was restarted.
