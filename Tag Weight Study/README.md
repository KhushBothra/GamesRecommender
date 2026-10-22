# Steam tag weight decay experiment

## Purpose

This study uses the Nik Davis SteamSpy tag-count dataset to estimate how tag importance changes with tag position. It then compares candidate decay curves against the empirical vectors calculated from that dataset.

The empirical vectors are the experiment's target values. They are formed by max-normalizing each game's tag vote counts and averaging those normalized values by rank among games with the same number of tags. The averaging step does not itself select a decay function. The experiment's goal is to find which proposed smooth curve best fits these empirical target points.

This is a calibration experiment for tag position weights. It does not make SteamSpy vote counts the recommender's tag weights directly, and it does not evaluate recommendation quality or user satisfaction.

## Latest: length-aware exponential and holdout errors

The follow-up experiment is implemented in `generalized_decay.py`; see
[`results/generalization/RESULTS.md`](results/generalization/RESULTS.md).
It compares nine candidates on the same game splits: 60% training, 20% validation,
and the remainder test within each tag-list length, using seed `20261023`.
Models fit training means only; validation selects the smooth rate form, and test
games are excluded from both fitting and selection. All ranks from a game stay
together. There are 16,605 training, 5,530 validation, and 5,553 test games.

The selected smooth form is a **saturating exponential rate**:

```text
x = (r-1)/(k-1)
lambda(k) = a * (1 - exp(-b*(k-1)))
w(r,k) = exp(-lambda(k)*x)
```

Validation selected this form over affine and power rate functions. Its all-data
refit for reuse has `a=2.249150856857026`, `b=0.0946333263878718`. Its training-only
evaluation fit has `a=2.2072768579717916`, `b=0.0969780937258659`; the holdout scores
below evaluate that training-only fit, not the all-data refit.

| Model | Parameters | Test mean-curve macro RMSE | Test individual-game macro RMSE |
|---|---:|---:|---:|
| Separate per-length exponential | 20 | **0.042258** | **0.187755** |
| Smooth saturating-rate exponential | 2 | 0.045610 | 0.188538 |
| Absolute-rank exponential | 1 | 0.075244 | 0.197807 |
| Original fixed shared power | 1 | 0.165070 | 0.246402 |

Separate rates have the lowest validation and test error. The smooth rule is a
compact compromise with close errors and can predict lengths lacking a fitted
lookup entry. A paired bootstrap conditional on these fitted models estimates
smooth-minus-separate macro MSE difference `+0.00029458`, with 95% interval
`[+0.00005653, +0.00054576]`: the small measured advantage for separate rates should
not be described as an exact tie. The same ordering holds when rescoring without
the sparse 21-tag group. A leave-length-out stress test of the selected smooth
form has mean-curve macro RMSE `0.046568`.

Mean-curve errors measure predictions of held-out group averages. Individual-game
errors also contain within-length variation; both exclude rank 1. Sample-weighted
errors, per-length errors, the split assignments, training parameters, predictions,
the paired bootstrap, and input hashes are exported in `results/generalization/`.
The split is retrospective within this historical dataset. Bootstrap intervals
exclude fitting/selection uncertainty and do not establish recommendation quality.
The leave-length-out check fixes the validation-selected form, so it is a stress
test rather than nested selection for entirely unseen lengths.

Run both stages and their checks from this folder:

```bash
.venv/bin/python fit_decay.py
.venv/bin/python generalized_decay.py
.venv/bin/python -m unittest discover -s . -p 'test_*.py' -v
.venv/bin/python execute_notebook.py
```

`generalized_decay.py` accepts `--study-dir`, `--output-dir`, and `--seed`.
`selected_evaluation_model.json` contains the training-only smooth model used in
the test tables; `selected_length_aware_model.json` contains its all-data refit for
reuse. `training_models.json` contains every evaluated model. The original
`results/selected_model.json` remains the winner of the earlier fixed shared-curve
comparison; it is not the new length-aware model.

```python
import json
from pathlib import Path
from generalized_decay import predict_weights

model = json.loads(Path("results/generalization/selected_length_aware_model.json").read_text())
weights = predict_weights(10, model)
```

`predict_weights` returns an empty vector for zero tags and `[1]` for one tag.
Smooth models use the same formula above length 21, explicitly as extrapolation.
The per-length lookup baseline raises `ValueError` for lengths without a fitted
rate. No main-catalogue features are changed.

## Original comparison of four fixed families

The four-family experiment is implemented and its results are in
[`results/RESULTS.md`](results/RESULTS.md). The selected **shared** rule is:

`w(r,k) = (1 + x)^(-1.458500902)`, where `x = (r - 1) / (k - 1)` for `k > 1`.

Its group-balanced RMSE is **0.165817**. The best constrained straight line is
`w = 1 - 0.7098613871*x`, with RMSE **0.193664**. Excluding the 12-game 21-tag group
does not change the shared winner. Sample-size-weighted scoring also favors power.

With **separate parameters for each length**, exponential decay is best, with
group-balanced RMSE **0.040460**. A shared normalized-position curve is a substantial
compromise: it imposes the same last-rank weight at every length, whereas the
empirical means vary with length. Use the separate fits for descriptive accuracy
at observed lengths, or the selected shared model when one compact rule is needed.
These scores describe fits to the same empirical means, not held-out performance.

### Exact fitted candidates and scoring

| Family | Formula | Parameter range |
|---|---|---|
| Linear | `1 - a*x` | `0 <= a <= 1` |
| Exponential | `exp(-lambda*x)` | `lambda >= 0` |
| Logarithmic | `1 - a*ln(1+x)/ln(2)` | `0 <= a <= 1` |
| Power | `(1+x)^(-p)` | `p >= 0` |

Each family has one parameter per fit, equals 1 at rank 1, stays in `[0,1]`, and
never increases. Fit all 20 informative lengths (`k=2...21`) together, then fit
each length independently. The constrained linear parameter is solved
analytically. Other parameters use one-dimensional numerical optimization,
checking boundaries and refining minima found on linear/logarithmic search grids.

For each length, `MSE_k = sum((prediction-target)^2)/(k-1)` over ranks `2...k`.
The fitting objective and primary ranking metric are
`group_balanced_MSE = sum(MSE_k)/number_of_groups`. Report
`group_balanced_RMSE = sqrt(group_balanced_MSE)` for readability.
The sensitivity metric is `sample_weighted_MSE = sum(n_games_k*MSE_k)/sum(n_games_k)`;
its RMSE is also the square root of the aggregated MSE. These are not arithmetic
averages of per-group RMSEs. Sample-weighted scoring evaluates the same fitted
parameters; it does not refit them. The `without_k21` scenario refits shared models
on the other 19 lengths and reports both fitting strategies again.

Only existing target cells are used. Rank 1, empty vectors, and singleton vectors
do not contribute to any score. No extra probability or L2 normalization is used.

### Run the experiment

The isolated `.venv` in this study folder is already prepared in the current
workspace. From this folder:

```bash
.venv/bin/python fit_decay.py
.venv/bin/python -m unittest -v test_fit_decay.py
.venv/bin/python execute_notebook.py
```

For a fresh environment, use **Python 3.10 or later** to create `.venv`, then
install this folder's dependencies (the main preprocessing requirements are
separate):

```bash
python3 -m venv .venv
.venv/bin/python -m pip install -r requirements.txt
```

The CLI resolves inputs relative to its own script, so it also works from other
directories. Optional arguments are `--study-dir PATH` and `--output-dir PATH`;
the default output folder is `results/` under the selected study folder.

`execute_notebook.py` runs every notebook cell with the calling Python interpreter,
preserves the source and both empirical CSVs, and checks that all 18 numerical
exports from both stages match prior CLI runs byte for byte. It saves the executed notebook and
`results/notebook_validation.json`. The notebook can also be run interactively
using the study environment as its kernel. If a source change makes existing
empirical CSVs stale, the notebook stops rather than overwriting them; archive
those two generated files before deliberately regenerating them.

### Outputs and reuse

| Output in `results/` | Contents |
|---|---|
| `RESULTS.md` | Selected formula, best straight line, scores, sensitivity, and limitations |
| `model_scores.csv` | All families, both fitting strategies, both scenarios, both aggregate metrics |
| `fitted_parameters.csv` | Shared and per-length parameters, including sensitivity refits |
| `per_group_errors.csv` | MSE/RMSE, sample sizes, and parameters for every fitted group |
| `predictions.csv` | Targets, predictions, residuals, and score-inclusion flags by rank |
| `empirical_targets.csv` | Original observed target values in long form, including the singleton |
| `selected_model.json` | Full-data shared winner and reuse metadata |
| `run_manifest.json` | Input/script hashes, source audit, group sizes, Python and library versions |
| `model_comparison.png` / `.svg` | Shared versus separate errors, with and without `k=21` |
| `shared_fits.png` / `.svg` | Empirical targets and all shared curves for every length |
| `per_length_fits.png` / `.svg` | Empirical targets and all independently fitted curves for every length |

From this folder, reuse the selected shared rule as follows:

```python
import json
from pathlib import Path
from fit_decay import weight_vector

model = json.loads(Path("results/selected_model.json").read_text())
weights = weight_vector(10, model["family"], model["parameter"])
```

`weight_vector(n_tags, family, parameter)` returns a NumPy array in tag-rank order.
For zero tags it returns an empty vector; for one tag it returns `[1]`. Lengths
above 21 use the same formula without clamping or truncation and are explicitly
extrapolations beyond this calibration data. Invalid lengths, parameters, or
family names raise `ValueError`. The study does not alter main-catalog features.

The tests cover parameter recovery, boundary fits, function constraints,
extrapolation, missing-cell handling, group-balanced scoring, source consistency,
selection, and sensitivity refitting. No uncertainty intervals or recommendation
evaluation are included. The historical calibration cannot confirm that ordered
tags in a different catalogue are ordered by vote count or importance.

## Relationship to the project proposal

The formal proposal asks how tag importance should decrease as a tag's position increases, and proposes comparing different rates and forms of positional decay. It does not specify the exact curve equations. The internal evaluation document emphasizes controlling feature scaling when comparing game representations. Keep the selected curve and its implementation documented so it can be applied consistently in the recommender experiments.

## Files in this folder

| File | Purpose |
| --- | --- |
| `Tag Weights DB/steamspy_tag_data.csv` | Source SteamSpy table. It has one `appid` column and one column per tag; each tag cell is a vote count. |
| `Empirical Tag Weight Decay.ipynb` | Audits the source and creates the indexed tag-count data and empirical average vectors. |
| `steamspy_tag_data_indexed.csv` | One row per source game, with `appid`, `n_tags`, and a JSON object of nonzero tag-count pairs, sorted by descending count when written. |
| `steamspy_average_decay_by_n_tags.csv` | Empirical target vectors. Each row has an `n_tags` group, its `n_games` sample size, and `weight_rank_1` through `weight_rank_21`. |

The other CSV files in `Tag Weights DB` are not inputs to this experiment unless the experiment is deliberately expanded later.

## Current source-data audit

The current `steamspy_tag_data.csv` has **29,022 rows**, **371 tag columns**, and **372 columns total** including `appid`. The audit found no missing AppIDs, duplicate AppIDs, missing tag cells, negative counts, or non-integer counts. There are **575 games with no nonzero tag counts**; these have `n_tags = 0` and cannot contribute a normalized decay vector.

The current number of games in each group is:

| `n_tags` | Games | `n_tags` | Games | `n_tags` | Games | `n_tags` | Games |
|---:|---:|---:|---:|---:|---:|---:|---:|
| 0 | 575 | 6 | 2,386 | 12 | 677 | 18 | 343 |
| 1 | 759 | 7 | 1,791 | 13 | 535 | 19 | 514 |
| 2 | 2,312 | 8 | 1,391 | 14 | 486 | 20 | 2,225 |
| 3 | 4,100 | 9 | 1,097 | 15 | 419 | 21 | 12 |
| 4 | 3,893 | 10 | 924 | 16 | 417 |  |  |
| 5 | 3,037 | 11 | 771 | 17 | 358 |  |  |

All groups from 0 through 21 are retained in the current output. The 21-tag group has only 12 games, so its mean vector is much less stable than the larger groups. The 20-tag group is unusually large at 2,225 games; keep it, but note its sample size when interpreting a group-balanced fit. The `n_tags = 0` row has no weights. The `n_tags = 1` vector contains only rank 1, which is always 1 after normalization and provides no information about decay after the first tag.

These are snapshots of the files currently in the folder. Rerun the notebook if the source file changes, and keep generated outputs together with the source version used to make them.

## How the empirical targets are made

For game `i`, let its positive SteamSpy vote counts be sorted in descending order:

`c[i,1] >= c[i,2] >= ... >= c[i,k] > 0`, where `k = n_tags`.

Each game's ranked counts are max-scaled by its own rank-1 count:

`w[i,r] = c[i,r] / c[i,1]`.

Thus every game with at least one tag has rank-1 weight 1, and all weights lie between 0 and 1. For each tag-count group `k`, average each rank across the games in that group:

`mu[k,r] = (1 / N[k]) * sum(w[i,r] for games i with n_tags = k)`, for `r = 1 ... k`.

`mu[k, r]` is the empirical target for rank `r` when a game has `k` nonzero tags. The individual game is normalized before averaging, so a very popular game with large raw SteamSpy counts does not dominate the group's average simply because its vote totals are larger. These are max-scaled vote ratios, not probabilities and not counts of users who prefer a tag in the recommender.

Sorting uses descending vote count. If two tags have the same count, the notebook's stable sort preserves their original source-column order. Zero-count cells are omitted. Missing cells are distinct from zero, though the current source audit found no missing tag cells.

## Curve-fitting experiment

### Target and candidate curves

Use the empirical `mu[k,r]` vectors as the target points. For each candidate curve family, define a function `f(r, k; theta)` that returns the weight at rank `r` for a game with `k` tags. Compare its output with `mu[k,r]` for every available `k` and rank.

The proposal names different decay rates and forms but leaves the formulas open. The teammate implementing the experiment should write down each exact formula, parameter range, and interpretation before fitting it. Candidate families to consider include:

- **Gentle and harsh decay:** shallow and steep forms. These can be fixed curves or different parameter values in one family; record which meaning is used.
- **Linear decay:** a straight-line decrease from weight 1, with parameters constrained so weights stay in `[0, 1]`.
- **Exponential decay:** a smooth, monotone decrease controlled by a rate parameter.
- **Logarithmic decay:** a monotone curve that drops more quickly near the first ranks and then flattens, with parameters constrained to keep weights nonnegative.
- **Power or reciprocal decay:** smooth alternatives with a controllable tail.

These are candidate families for the experiment, not equations already fixed by the proposal. Avoid comparing labels such as “gentle” and “harsh” without specifying their equations and parameters. For a candidate function, rank 1 should be fixed at 1 to match the normalization. Require non-increasing weights and keep weights in `[0, 1]` unless there is a specific, justified reason to test otherwise.

For functions intended to work for different tag-list lengths, one useful position coordinate is `x = (r - 1) / (k - 1)` for `k > 1`. This maps the first and last positions to 0 and 1 for every vector. Decide and document whether a candidate is fit separately for each `k` or uses shared parameters across all `k`. A separate fit for every `k` can describe each empirical vector closely but produces a family of per-length curves; a shared fit yields one more compact rule and may fit some lengths less closely. Report per-`k` errors either way.

### Recommended comparison procedure

1. Load the empirical average CSV and use only rank cells that exist for each `n_tags` value. Do not treat blank cells beyond the vector's length as zeros.
2. For each candidate family, fit its parameter or parameters to the empirical target points. Keep the normalization and monotonicity constraints consistent across candidates.
3. Calculate error separately for each `k`. Since all candidates are constrained to equal 1 at rank 1, report the decay fit error over ranks `2 ... k`; rank 1 can be shown but contributes no discrimination between candidates.
4. Use the **macro-average of per-`k` errors** as the primary summary so the large groups do not drown out smaller tag-count groups. Also report a sample-size-weighted summary as a sensitivity check. State the exact metrics and aggregation formula in the results.
5. Report the fitted parameters, per-`k` errors, overall primary score, weighted sensitivity score, and a plot showing empirical target points and fitted curves. Include the `n_games` sample size in tables or plot annotations, especially for `k = 21`.
6. Include the empirical target vectors in the results as the reference points. They are not a competing theoretical curve and should not be replaced by the fitted values in the source output.
7. If two candidate families fit similarly, prefer the simpler curve only if it remains easy to explain and implement. Record the fit difference and the reason for the final choice.

An optional robustness analysis is to bootstrap games within each `n_tags` group and recompute the empirical means. This would show how uncertain each target vector is, particularly for the 12-game 21-tag group. It should be reported as uncertainty around the empirical targets, not as a new source of labels.

This is a fit to historical aggregate observations. A low curve-fit error means the function matches these empirical averages; it does not prove that the resulting recommendations are better. Recommendation quality must be evaluated in the downstream recommender experiment.

## Applying the selected curve to the main game catalogue

The main dataset remains a separate catalogue. For each game's ordered Tags list, assign the selected decay weight according to the tag's position and the curve's `n_tags` input. Genres and Categories keep weight 1 under the current project plan. Keep these sources separate until the runtime feature construction step so the tag weights do not reduce the genre or category weights.

The source used here has observed `n_tags` values only from 0 to 21. If the main dataset contains games with more than 21 tags, document how the selected formula handles them. A formula that takes normalized position may extend to longer lists, but such weights are an extrapolation beyond this calibration data. Do not silently clamp or truncate ranks.

The final recommender should apply weights during vector construction. It should not expect a text embedder to infer numeric importance from a phrase such as “tag weight 0.6.” The selected weight must affect the descriptor feature's numeric contribution. The exact representation method belongs to the downstream embedding/vector-construction work.

## Reproducing the current outputs

Open and run `Empirical Tag Weight Decay.ipynb` from top to bottom. The notebook reads the source CSV without modifying it and writes:

- `steamspy_tag_data_indexed.csv`
- `steamspy_average_decay_by_n_tags.csv`

Both outputs are written to this folder only when absent; existing outputs are verified without overwriting them. The notebook's source-path search supports launching Jupyter from the project root or from the study folder. The notebook now continues through candidate curve fitting, scoring, sensitivity analysis, plots, and final shared-curve selection using the same implementation as the CLI.

## Handoff checklist

- [x] Confirm the source CSV and generated outputs correspond to the same dataset version.
- [x] Specify every candidate curve equation and allowed parameter range.
- [x] Compare shared and per-`n_tags` parameters; document the accuracy/complexity tradeoff.
- [x] Fit all candidate families to the empirical target vectors and report per-group errors.
- [x] Use a group-balanced primary score and include a sample-size-weighted sensitivity score.
- [x] Plot targets and fits; show sample sizes and call out the small 21-tag group.
- [x] Choose the best-fitting shared curve based on the stated metric, and record fit quality and limitations.
- [x] Document how the selected curve handles tag-list lengths above 21.
- [x] Export the fitted curve, parameters, and callable function for downstream handoff.

Applying these weights to the main catalogue remains downstream work; this study
does not change the preprocessing pipeline or recommender implementation.
