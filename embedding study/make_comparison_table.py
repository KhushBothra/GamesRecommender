"""Average cell scores at the SAME k. Never pick a best k inside each cell.

Can run during an experiment: pending cells stay visible in the table.
Run: python "embedding study/make_comparison_table.py"
"""

import argparse
import math
from pathlib import Path

import pandas as pd

from file_helpers import STUDY_ROOT, read_json, write_csv, file_sha256


def mean_and_spread(values):
    """No valid scores means blank output, not a score of zero."""
    if not values:
        return None, None
    average = sum(values) / len(values)
    spread = math.sqrt(sum((value - average) ** 2 for value in values) / len(values))
    return average, spread


def make_table(results):
    manifest = read_json(results / 'experiment.json')
    settings = manifest['settings']
    rows = []

    # 1. Treat each embedding, k and seed as a separate comparison setting.
    for embedding in settings['combinations']:
        for k in settings['clusters']:
            for seed in settings['seeds']:
                silhouette_scores = []
                db_scores = []
                cell_stability = []
                completed = valid = invalid = partial = valid_pairs = 0

                # 2. Read the nine cells for this SAME k.
                for x in range(1, 4):
                    for y in range(1, 4):
                        cell = f'x{x}_y{y}'
                        path = results / embedding / cell / f'k_{k:02d}_seed_{seed}' / 'result.json'
                        if not path.exists():
                            continue  # Still pending; do not count it as invalid.
                        report = read_json(path)
                        score = report['score']
                        identity = (score['combination'], score['region'], score['k'], score['seed'])
                        if identity != (embedding, cell, k, seed):
                            raise ValueError(f'Result identity does not match its folder: {path}')
                        for filename, digest in report['files'].items():
                            if file_sha256(path.parent / filename) != digest:
                                raise ValueError(f'Changed result file: {path.parent / filename}')
                        completed += 1
                        if score['status'] == 'ok':
                            valid += 1
                        elif score['status'] == 'partial':
                            partial += 1
                        elif score['status'] == 'invalid':
                            invalid += 1
                        else:
                            raise ValueError(f'Unknown result status in {path}')

                        # Each metric has its own valid-cell count.
                        for name, values in [('silhouette', silhouette_scores),
                                             ('davies_bouldin', db_scores)]:
                            value = score[name]
                            if value is not None:
                                if not math.isfinite(value):
                                    raise ValueError(f'Non-finite score in {path}')
                                values.append(value)

                        # Average ARI pairs within each cell first, then across cells.
                        pairs = [pair['ari'] for pair in report['stability'] if pair['ari'] is not None]
                        if pairs:
                            cell_stability.append(sum(pairs) / len(pairs))
                            valid_pairs += len(pairs)

                # 3. Average valid cell scores. Keep missing/invalid cells visible.
                sil_mean, sil_spread = mean_and_spread(silhouette_scores)
                db_mean, db_spread = mean_and_spread(db_scores)
                ari_mean, ari_spread = mean_and_spread(cell_stability)
                rows.append({
                    'embedding': embedding, 'k': k, 'seed': seed,
                    'completed_cells': completed, 'pending_cells': 9 - completed,
                    'valid_cells': valid, 'partial_cells': partial, 'invalid_cells': invalid,
                    'valid_fraction': valid / 9,
                    'silhouette_cells': len(silhouette_scores),
                    'silhouette_mean': sil_mean, 'silhouette_std': sil_spread,
                    'db_cells': len(db_scores), 'db_mean': db_mean, 'db_std': db_spread,
                    'stability_cells': len(cell_stability), 'stability_pairs': valid_pairs,
                    'stability_mean': ari_mean, 'stability_std': ari_spread,
                    'table_status': 'complete' if completed == 9 else 'PROVISIONAL',
                    'smoke_test': manifest['smoke_test'],
                })

    return pd.DataFrame(rows)


def save_tables(results):
    results = Path(results)
    table = make_table(results)
    write_csv(results / 'comparison_table.csv', table)

    # 4. Put A/B/C/D next to each other, with one row for each k and seed.
    columns = ['silhouette_mean', 'db_mean', 'valid_cells', 'pending_cells', 'stability_mean']
    side_by_side = table.pivot(index=['k', 'seed'], columns='embedding', values=columns)
    side_by_side.columns = [f'{embedding}_{metric}' for metric, embedding in side_by_side.columns]
    write_csv(results / 'side_by_side.csv', side_by_side.reset_index())

    pending = int(table.pending_cells.sum())
    if pending:
        print(f'PROVISIONAL: {pending} cell/k/seed settings are still pending.')
    print(f'Comparison table: {results / "comparison_table.csv"}')
    print('These are averages of local-cell scores, not global clustering scores.')
    print('Higher silhouette, lower DB, and higher stability are desirable.')
    print('Compare coverage and example games too. No automatic winner is selected.')
    return table


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--results-dir', type=Path,
                        default=STUDY_ROOT / 'region_results')
    args = parser.parse_args()
    save_tables(args.results_dir)
