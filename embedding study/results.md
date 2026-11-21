# Embedding study: methodology and results

Report prepared: 21 November 2026. This documents the completed full-data study,
not the earlier 1,000-game trial. No new full-data models were trained during
the folder cleanup.

## Main finding

**A (tags + genres + categories) is our provisional choice.** It has the highest
local-average Silhouette at 7 of 8 tested cluster counts. B is a close candidate:
it has the lowest local-average Davies–Bouldin score at 5 of 8 counts. Against B,
A has higher resampling stability at 5 counts. These are comparisons of measured
values, not statistically significant wins.

There is **no winner across every metric**. Across all four embeddings, D has
the highest stability at 5 counts, but weaker separation than A/B. The most
stable individual setting is B at k=2 (mean ARI 0.999545). That setting creates
only two broad groups per nonempty cell, not two groups in the whole catalogue.

Silhouette values are low, so the experiment does not demonstrate strongly
separated natural game categories. A is a useful baseline for retrieval testing,
not a proven best recommendation method. No final best k has been selected.

## 1. Data and alignment

- Official inputs: `../Data Prep/games_cleaned.csv` and `../Data Prep/games_descriptors.csv`.
- Same 117,344 games across A/B/C/D; 509 descriptor columns.
- The team confirmed this population as the approved English-only corpus.
- Cleaned data provide AppID, Name, Description, Peak CCU and Popularity Bin
  (also developer, publisher and price fields, which are not embedded here).
- Descriptors combine tags, genres and categories. Supplied tag-position weights
  are already applied; absent terms have value 0. No second tag decay is added.
- Reject duplicate/missing AppIDs and non-finite feature values. Align the two
  files by AppID, preserve descriptor column order, and match saved vector rows
  to the game index by ID and checksum. CSV row order is not assumed to match.

## 2. The four embeddings

| Embedding | Descriptors | Description | Popularity | Vector length |
| --- | --- | --- | --- | --- |
| A | 100% | 0% | 0% | 509 |
| B | 85.714% (6/7) | 0% | 14.286% (1/7) | 510 |
| C | 66.667% (2/3) | 33.333% (1/3) | 0% | 893 |
| D | 60% | 30% | 10% | 894 |

1. Normalize each descriptor row to length 1. A zero descriptor row stays zero.
2. Clean description HTML, remove script/style text and collapse whitespace.
3. Encode descriptions with pretrained `sentence-transformers/all-MiniLM-L6-v2`:
   384 values, CPU batches of 64, maximum 256 tokens, truncate longer descriptions,
   and normalize each output row to length 1. We did not train or fine-tune MiniLM.
   Empty descriptions are rejected, not silently replaced with invented text.
4. Map the existing seven popularity bins to 0/6, 1/6, ..., 6/6. We do not recompute
   bins from raw Peak CCU. This is an ordinal scalar representation.
5. Multiply each block by the square root of its stated weight, then concatenate
   descriptors, description if present, and popularity if present. Do not apply
   another normalization after concatenation. Weights apply to squared Euclidean
   distance contributions; they are not accuracy percentages.

Each full CSV contains AppID and Name followed by the vector coordinates.
The numeric matrices and all coordinates in their CSV exports were checked.

## 3. Mapper-inspired overlapping cover

We borrow the filter, overlapping-cover and local-clustering stages from
[Games Mapper](https://arxiv.org/html/2606.14376v2). We do **not** build its graph,
edges, connectivity analysis or graph-assisted recommendations. The paper uses
release year in its case study; our distance/alignment lens is a project choice.

For each representation separately:

1. Compute the mean full vector and each game's displacement from it.
2. Horizontal coordinate = Euclidean length of that displacement (raw radius).
3. Fit PC1 on centered full vectors. Make its largest-magnitude component positive
   to remove arbitrary sign flips. PCA random seed is 42.
4. Vertical coordinate = cosine alignment of the displacement with PC1. This is
   alignment with a reference direction, not pairwise cosine similarity of games.
   Exact-mean games get alignment 0 and are counted.
5. Scale radius by its 99th percentile and clip the **lens coordinate only** to
   [0,1]. Keep alignment in [-1,1] and save the raw radius. Full vectors are untouched.
6. Create three equal-width intervals on each axis, with overlap equal to 30% of
   interval width. Their Cartesian product gives nine inclusive rectangles.
7. Keep every region/AppID membership. A game can belong to multiple cells;
   those memberships are not collapsed into a global label.

The same scaling and cover construction rules are used for A/B/C/D. The actual
members of a named cell can differ between representations because their fitted
lenses differ. The 2D lens loses information; the full vectors are still used
for clustering and all distance-based scores.

## 4. Local clustering and evaluation

For each representation and cell, try k = **2, 3, 5, 8, 13, 21, 34, 55**.
K means the requested number of clusters inside that cell. It is not the total
number of clusters in the catalogue. These are eight selected settings, not every
integer from 2 to 55 and not a proof of the globally optimal count.

| Setting | Value |
| --- | --- |
| Clustering | scikit-learn KMeans, Euclidean geometry |
| Initialization | k-means++, n_init=10 |
| Main random seed | 42 |
| Maximum iterations / tolerance | 300 / 0.0001 |
| Algorithm / CPU threads | Lloyd / 8 |
| Silhouette | Euclidean; fixed sample of up to 5,000 games per cell |
| Davies–Bouldin | All games in the cell |
| Sampling rule | One seed-42 game priority, shared across k/seeds and representations |

Metrics require at least two nonempty labels and fewer labels than games. Empty
or too-small cells are recorded as invalid; k is never silently reduced. A sampled
Silhouette is invalid if the sample lacks a valid partition, even if DB is valid.

For **each fixed k**, take the unweighted mean of valid cell scores separately
for each metric. Save their standard deviation and valid-cell counts. We do not
choose a different winning k inside each cell and average those winners. These
are local-cell macro averages, **not conventional global clustering scores**.

Profiles use descriptor presence: cluster prevalence minus prevalence in the
whole experiment population. Save up to five positive standout terms and up to
three games nearest the full-vector cluster centre. Profiles are descriptive;
they are not ground-truth game-category labels.

## 5. Stability procedure

Keep the full vectors, fitted mean/PC1 and cover fixed. For each eligible cell/k:

1. Draw three 80% game subsamples without replacement, using seeds 42, 43, 44.
2. Fit the same KMeans settings on each sample, always with clustering seed 42.
3. Compare each of the three pairs on their shared AppIDs using Adjusted Rand
   Index (ARI), which does not depend on numeric cluster-label names.
4. Average eligible pair scores within a cell, then average across cells.

High ARI means repeatable grouping under these samples, not correct semantics.
This does not measure sensitivity to refitting the lens or to multiple independent
KMeans random seeds. Stability SD below is across cell means, not across pairs.

## 6. Completion and cover coverage

4 embeddings x 9 cells x 8 k values = **288 requested settings**.
All 288 are recorded: **192 valid primary fits and 96 skipped empty-cell settings**.
There are 576 extra subsample fits and 576 valid pairwise ARI comparisons.
Each representation/k has six valid cells out of nine (66.7%). All three cells
in the first radius column, x1_y1/x1_y2/x1_y3, are empty in every representation.
These are skips, not software failures. The nonempty cells still cover all games.

### Number of games in each cell

Counts include repeated memberships across overlapping cells, not extra games.

| Cell | A | B | C | D |
| --- | --- | --- | --- | --- |
| x1_y1 | 0 | 0 | 0 | 0 |
| x1_y2 | 0 | 0 | 0 | 0 |
| x1_y3 | 0 | 0 | 0 | 0 |
| x2_y1 | 6699 | 6638 | 432 | 426 |
| x2_y2 | 66868 | 66919 | 21955 | 22015 |
| x2_y3 | 33523 | 33512 | 12649 | 12716 |
| x3_y1 | 27458 | 27416 | 25061 | 25043 |
| x3_y2 | 93106 | 93048 | 109543 | 109556 |
| x3_y3 | 25627 | 25503 | 31064 | 31066 |

## 7. All fixed-k aggregate results

Each entry is **mean +/- population SD across valid cells**. Higher Silhouette
and ARI are preferred; lower DB is preferred. Every row has six valid metric
cells and six stability cells, with 18 ARI pairs. The attached CSVs retain the
full precision; displayed values are rounded to six decimals.

### Embedding A

| k | Local Silhouette | Local Davies–Bouldin | Local stability ARI | Valid cells |
| --- | --- | --- | --- | --- |
| 2 | 0.079991 +/- 0.022254 | 3.544262 +/- 0.521786 | 0.777227 +/- 0.314023 | 6 / 9 |
| 3 | 0.072488 +/- 0.020632 | 3.293970 +/- 0.426749 | 0.910328 +/- 0.138437 | 6 / 9 |
| 5 | 0.079068 +/- 0.025405 | 3.058182 +/- 0.423672 | 0.742592 +/- 0.183505 | 6 / 9 |
| 8 | 0.079346 +/- 0.019106 | 2.986717 +/- 0.255937 | 0.632934 +/- 0.233536 | 6 / 9 |
| 13 | 0.076283 +/- 0.013518 | 2.962605 +/- 0.188756 | 0.629795 +/- 0.118250 | 6 / 9 |
| 21 | 0.073307 +/- 0.014854 | 3.003535 +/- 0.205439 | 0.596768 +/- 0.094221 | 6 / 9 |
| 34 | 0.075213 +/- 0.019300 | 2.941285 +/- 0.249097 | 0.542355 +/- 0.051793 | 6 / 9 |
| 55 | 0.076554 +/- 0.021993 | 2.872798 +/- 0.238899 | 0.540404 +/- 0.052909 | 6 / 9 |

### Embedding B

| k | Local Silhouette | Local Davies–Bouldin | Local stability ARI | Valid cells |
| --- | --- | --- | --- | --- |
| 2 | 0.079819 +/- 0.022446 | 3.548006 +/- 0.525985 | 0.999545 +/- 0.000497 | 6 / 9 |
| 3 | 0.071736 +/- 0.021980 | 3.306060 +/- 0.454208 | 0.887942 +/- 0.141924 | 6 / 9 |
| 5 | 0.076814 +/- 0.026288 | 2.979311 +/- 0.441332 | 0.569997 +/- 0.216974 | 6 / 9 |
| 8 | 0.076912 +/- 0.018222 | 2.968489 +/- 0.174296 | 0.577988 +/- 0.191377 | 6 / 9 |
| 13 | 0.076112 +/- 0.010858 | 2.930335 +/- 0.243207 | 0.648067 +/- 0.108890 | 6 / 9 |
| 21 | 0.073030 +/- 0.011970 | 2.947817 +/- 0.250594 | 0.586793 +/- 0.099838 | 6 / 9 |
| 34 | 0.073983 +/- 0.016836 | 2.933103 +/- 0.255348 | 0.544094 +/- 0.085366 | 6 / 9 |
| 55 | 0.077913 +/- 0.025559 | 2.885160 +/- 0.295420 | 0.537674 +/- 0.033804 | 6 / 9 |

### Embedding C

| k | Local Silhouette | Local Davies–Bouldin | Local stability ARI | Valid cells |
| --- | --- | --- | --- | --- |
| 2 | 0.055449 +/- 0.015420 | 4.189921 +/- 0.698300 | 0.991106 +/- 0.008743 | 6 / 9 |
| 3 | 0.049865 +/- 0.014536 | 3.975024 +/- 0.441345 | 0.781720 +/- 0.217490 | 6 / 9 |
| 5 | 0.049249 +/- 0.013447 | 3.786429 +/- 0.353375 | 0.780688 +/- 0.135415 | 6 / 9 |
| 8 | 0.043003 +/- 0.009604 | 3.661876 +/- 0.265852 | 0.666525 +/- 0.161509 | 6 / 9 |
| 13 | 0.042789 +/- 0.006310 | 3.665594 +/- 0.185547 | 0.621821 +/- 0.105933 | 6 / 9 |
| 21 | 0.035577 +/- 0.002745 | 3.655799 +/- 0.268350 | 0.625423 +/- 0.104773 | 6 / 9 |
| 34 | 0.033263 +/- 0.003748 | 3.604938 +/- 0.323440 | 0.537313 +/- 0.110255 | 6 / 9 |
| 55 | 0.030736 +/- 0.006899 | 3.631370 +/- 0.540075 | 0.545724 +/- 0.073661 | 6 / 9 |

### Embedding D

| k | Local Silhouette | Local Davies–Bouldin | Local stability ARI | Valid cells |
| --- | --- | --- | --- | --- |
| 2 | 0.055384 +/- 0.015481 | 4.192440 +/- 0.701430 | 0.922049 +/- 0.160805 | 6 / 9 |
| 3 | 0.048075 +/- 0.014335 | 3.885924 +/- 0.507101 | 0.920408 +/- 0.140946 | 6 / 9 |
| 5 | 0.049383 +/- 0.014022 | 3.780294 +/- 0.393454 | 0.828965 +/- 0.170803 | 6 / 9 |
| 8 | 0.045737 +/- 0.009400 | 3.768017 +/- 0.364370 | 0.670679 +/- 0.154041 | 6 / 9 |
| 13 | 0.037152 +/- 0.005334 | 3.856349 +/- 0.162178 | 0.648859 +/- 0.088690 | 6 / 9 |
| 21 | 0.037838 +/- 0.005863 | 3.639392 +/- 0.234374 | 0.584294 +/- 0.094103 | 6 / 9 |
| 34 | 0.031931 +/- 0.007241 | 3.647802 +/- 0.412373 | 0.572208 +/- 0.104884 | 6 / 9 |
| 55 | 0.030667 +/- 0.006337 | 3.595938 +/- 0.539541 | 0.529792 +/- 0.069279 | 6 / 9 |

## 8. Comparison at the same k

These are numerical leaders; small differences are not claims of statistical significance.

| k | Highest Silhouette | Lowest DB | Highest stability |
| --- | --- | --- | --- |
| 2 | A | A | B |
| 3 | A | A | D |
| 5 | A | B | D |
| 8 | A | B | D |
| 13 | A | B | D |
| 21 | A | B | C |
| 34 | A | B | D |
| 55 | B | A | C |

## 9. Interpretation and decision

- A leads Silhouette at 7/8 k values; B leads at k=55.
- B leads DB at 5/8 values; A leads at k=2, 3 and 55.
- Across all representations, D leads stability at 5/8 values, C at 2/8,
  and B at 1/8. A is not the most stable representation on every comparison.
- In the direct A-versus-B comparison, A has higher ARI at k=3, 5, 8, 21, 55;
  B has higher ARI at k=2, 13, 34.
- B at k=2 has the highest individual local-mean ARI, 0.999545. A and B have
  almost identical separation scores at k=2; high repeatability alone does not
  establish useful recommendations.
- Descriptions, at the tested model/weights, do not improve these separation
  measures. This does not prove descriptions contain no useful relevance signal.

**Provisional choice: A.** This is a judgement based on its generally higher
Silhouette, A/B stability comparison, and the lack of a consistent improvement
from popularity. No primary metric or significance threshold was predeclared
to establish a formal winner. B remains a serious candidate. No single combined
score or best k is invented after looking at these results.

## 10. Limits and next work

Silhouette ranges are approximately 0.072-0.080 for A, 0.072-0.080 for B,
0.031-0.055 for C and 0.031-0.055 for D. These are weak separation scores.
Lens fitting, dimensionality, block weights, ordinal popularity and truncating
descriptions at 256 tokens can affect the conclusions. Scores from different
representation geometries are descriptive comparisons, not ground-truth accuracy.
Only the chosen cover, eight k values, one clustering seed and three subsamples
were studied. Overlap makes cell results dependent; do not interpret them as
independent statistical replicates.

No expert-labeled relevance set or user-query retrieval test was used. Next,
compare A/B retrieval on the same agreed requests and inspect relevance before
making a final recommendation-quality claim. Full Mapper graph construction is
outside this study.

## 11. Evidence and files to share

Small files in `tables/` can be committed alongside this report:

- [All 32 aggregate rows](tables/comparison_table.csv): full-precision means,
  SDs, coverage, stability pair counts and completion flags.
- [A/B/C/D beside each other](tables/side_by_side.csv).
- [All 288 cell/k results and skip reasons](tables/cell_scores.csv).
- [All 576 stability pair results](tables/stability.csv).
- [Stability pair summary](tables/stability_summary.csv): this older summary's
  SD is across individual pairs; use comparison_table.csv for SD across cell means.
- [Cell populations](tables/cell_game_counts.csv).
- [Original run manifest](tables/original_run_manifest.json): exact settings,
  source hashes, selected-ID checksum, software versions and original code hashes.
- [Full original environment versions](tables/environment_versions.txt).

Full local artifacts remain in `region_results/`: lens CSVs, fitted mean/PC1,
exact bounds, memberships, per-cell AppID labels, centres and profiles. Full
embedding matrices/CSVs remain in `outputs/A` through `outputs/D`. Large outputs,
the text-model cache and environment are ignored by Git.

The folder was renamed from `clustering` to `embedding study` on 21 November
2026. Original run manifests deliberately retain their original absolute paths
and code hashes as historical evidence; those paths are not current commands.
The reported data and numerical results were preserved, not regenerated by the
cleanup. The simpler code uses direct script commands listed in README.md.

## 12. Source checksums

| Source | SHA256 |
| --- | --- |
| games_cleaned.csv | ab8ebf9a45103693be7a9fb88b15bed6f2b4c57334bcc9d01ded868e48414e13 |
| games_descriptors.csv | aa0f9c6a79a295b9f624cd77fe740f066ad3da74b5c3552075028e0d46e42c26 |
| Ordered selected AppIDs | 1cf751a0a710783d510629261f04f2144ad9a0d14bb09789a70d646fcbbb21bd |

Exact software versions used for the measurements:

| Software | Version |
| --- | --- |
| python | 3.12.14 |
| numpy | 2.5.4 |
| pandas | 3.1.0 |
| scikit-learn | 1.9.1 |
| scipy | 1.18.1 |
| joblib | 1.6.0 |
| sentence-transformers | 5.7.0 |
| torch | 2.14.1 |
