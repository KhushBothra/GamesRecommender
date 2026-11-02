# Game embeddings and clustering

Run commands from the Git checkout root, containing clustering/ and Data Prep/.

## Official input files

Both inputs are tracked with Git LFS:

- Data Prep/games_cleaned.csv: AppID, name, description and popularity data.
- Data Prep/games_descriptors.csv: AppID and already-weighted categorical features.

    git lfs pull origin

The current files contain 117,344 games and 509 categorical features. The code
joins by AppID and validates matching populations, unique IDs and finite weights.
It reads both CSVs without editing them. The original games.csv and local
descriptor-generation code are no longer needed for embeddings.

## Create or reuse embeddings

Use Python 3.12 and the repository's .venv-clustering environment. Install
clustering/requirements.txt when setting up a fresh checkout.

    .\.venv-clustering\Scripts\python.exe -m clustering.create_embeddings --offline

Omit --offline if the MiniLM model needs downloading. The command:

1. Reads and validates both official CSVs.
2. Normalizes the supplied categorical values without recalculating tag weights.
3. Creates or reuses the shared MiniLM description vectors.
4. Creates or reuses A/B/C/D matrices with the original weighting rules.
5. Exports full CSVs, or reuses existing CSVs after checking their checksums.

Use --cleaned PATH or --descriptors PATH to select alternative input files.
The command never writes to either input. It does not train K-means.

To validate inputs without encoding or writing outputs:

    .\.venv-clustering\Scripts\python.exe -m clustering.create_embeddings --prepare-only

## Unchanged feature logic

- The official descriptor values already include positional tag weights and the
  genre/category merging rules. Values are used as supplied, with no second decay.
- Categorical rows are L2-normalized.
- Descriptions use pretrained all-MiniLM-L6-v2: 384 dimensions, a 256-token limit,
  and normalized vectors. The encoder is not fine-tuned.
- Existing seven popularity bins map to 0/6 through 6/6; raw CCU is not binned again.
- Blocks receive square-root weight multipliers before concatenation.
  There is no final whole-vector normalization.

| Combination | Categorical | Description | Popularity | Dimensions |
|---|---:|---:|---:|---:|
| A | 1 | 0 | 0 | 509 |
| B | 6/7 | 0 | 1/7 | 510 |
| C | 2/3 | 1/3 | 0 | 893 |
| D | 0.6 | 0.3 | 0.1 | 894 |

Dimensions reflect the current official vocabulary and default MiniLM model.

## Output files

For each combination, for example D:

    clustering/outputs/D/features.npy
    clustering/outputs/D/features.json
    clustering/outputs/D/embeddings.csv
    clustering/outputs/D/embeddings_preview.csv
    clustering/outputs/D/embeddings_csv_manifest.json

CSV rows start with AppID and Name, followed by weighted feature values. The
full file contains every game; the preview contains ten games. Prefixes identify
categorical__, description__, and popularity__weighted columns.

The validation report is clustering/validation/embeddings_validation.json.
Data manifests and feature fingerprints reference the two official source
checksums. Existing outputs are reused only when their provenance agrees.
CSV reuse also checks the complete file checksum and the feature-matrix checksum.

The switch from locally generated descriptors to Khush's official CSV was
verified against every saved vector coordinate. Only source metadata changed;
the embedding and CSV payloads were retained unchanged. Its local audit record
is clustering/outputs/official_descriptor_adoption.json. Obsolete migration
backups were removed after verification; current checksums remain recorded.

Generated outputs, caches and the environment are ignored by Git. The official
Data Prep CSVs remain tracked by LFS.

## Clustering and evaluation

After embeddings are ready, train the original 28 K-means configurations:

    .\.venv-clustering\Scripts\python.exe -m clustering.run_all --offline

This uses the same two official files. It fits k = 2, 3, 5, 8, 13, 21, 34 for
A/B/C/D, using seed 42 and n_init=10, and reuses compatible feature matrices.
Models, cluster labels, centres and manifests go under k_XX/seed_42 folders.

Evaluate saved models later without retraining:

    .\.venv-clustering\Scripts\python.exe -m clustering.evaluate --output-dir clustering/outputs

This calculates full-data Davies-Bouldin index and silhouette on a shared
sample of 5,000 games by default. It does not select an optimal k automatically.
Mapper and the LLM/UI/backend recommender are separate work.

## Tests and earlier results

    .\.venv-clustering\Scripts\python.exe -m unittest discover -s clustering/tests -v

The 28 tests cover source alignment, preweighted-value preservation, read-only
input handling, LFS pointer rejection, feature weighting, description cache
validation and safe CSV reuse.

Obsolete dataset copies, old-run archives, download logs and one-time migration
helpers were removed. Current official inputs, embeddings, description cache,
research notebooks and the Python environment are retained.
The original games.csv remains local for the data-cleaning notebook.
