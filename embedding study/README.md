# Embedding study

Start with [results.md](results.md) for the methodology and complete study results.
The current recommendation is **A as a provisional baseline**, with B still a
candidate. The report explains the mixed metrics and weak cluster separation.

## Folder layout

```text
embedding study/
    make_embeddings.py       Build the four vectors and export CSVs
    read_data.py             Read the two source files and align AppIDs
    feature_vectors.py       Normalize and combine the feature blocks
    description_vectors.py   Clean text and run/reuse pretrained MiniLM
    save_csv.py              Write and verify readable embedding CSVs
    start_comparison.py      Edit the study settings here; start the study
    compare_embeddings.py    Loop over embeddings, cells and cluster counts
    region_functions.py      Lens, cover, KMeans, scores and stability
    make_comparison_table.py Average scores at each fixed k
    file_helpers.py          Small shared file/checksum helpers
    tests/                   Small checks, including a complete example run
    tables/                  Small shareable final result CSVs and run manifest
    outputs/                 Local full embeddings A/B/C/D (ignored by Git)
    cache/                   Local saved description vectors/model (ignored)
    region_results/          Local full experiment artifacts (ignored)
    validation/              Embedding-stage validation evidence
```

Other project work remains in `../Data Prep`, `../Tag Weight Study` and
`../system prompt`. The root raw `games.csv` is still needed by data-cleaning
work. The existing `.venv-clustering` environment keeps its name so its installed
paths continue working. There is no need to recreate it for this folder rename.

## Commands

Run these from the GamesRecommender project root. The folder name has a space,
so quote script paths. No package imports or `python -m clustering` are needed.

Validate inputs without changing or encoding anything:

```powershell
.\.venv-clustering\Scripts\python.exe ".\embedding study\make_embeddings.py" --prepare-only
```

Build embeddings, or reuse checked saved vectors/CSVs:

```powershell
.\.venv-clustering\Scripts\python.exe ".\embedding study\make_embeddings.py" --offline
```

The model must already be downloaded for offline encoding. Remove `--offline`
when downloading MiniLM for the first time. Source data are read-only. If Git
gave LFS pointers instead of CSV data, fetch them using `git lfs pull` first.

Start the study or read an already completed study:

```powershell
.\.venv-clustering\Scripts\python.exe ".\embedding study\start_comparison.py"
```

The existing full experiment is complete. Its folder is protected from retraining
over the original evidence. To run a new experiment, change RESULTS_FOLDER to a
new name in start_comparison.py. A small trial can use SAMPLE_GAMES=1000. The
completed historical run is not mixed with new code or parameter settings.

Refresh the local comparison tables from saved results without training:

```powershell
.\.venv-clustering\Scripts\python.exe ".\embedding study\make_comparison_table.py"
```

The small `tables/` CSVs document the published completed run. Refreshing a new
local run does not silently replace those published tables or results.md.

Check the code:

```powershell
.\.venv-clustering\Scripts\python.exe -B -m unittest discover -s ".\embedding study\tests" -v
```

Install dependencies only for a new environment:

```powershell
.\.venv-clustering\Scripts\python.exe -m pip install -r ".\embedding study\requirements.txt"
```

## Understand the code in three steps

1. `make_embeddings.py`: read AppIDs, normalize existing descriptors, get cached
   or new 384-number description vectors, encode popularity, apply square-root
   weights, join A/B/C/D, export and check the CSV numbers. Here "categorical"
   in older metadata means tags + genres + categories.
2. `compare_embeddings.py`: for each embedding, calculate its lens and nine
   overlapping cells. For each cell and k, cluster the **full original vectors**.
   Compute local Silhouette/DB scores, describe the clusters, and test stability.
3. `make_comparison_table.py`: average valid cells at the **same k**. Keep
   Silhouette, DB and stability separate, with coverage and pending-cell counts.
   Do not choose a different best k per cell before averaging.

The default description recipe is truncation at 256 tokens, which was used for
the reported study. Unused external-vector and long-description chunking modes,
the superseded global-clustering launchers and duplicate guides were removed.
Original finished data and results are preserved. Old absolute paths in historical
manifests describe where that run originally happened, not where to run code now.
