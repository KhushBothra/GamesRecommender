"""Start here: these are the settings for our embedding comparison.

From the project folder run: python "embedding study/start_comparison.py"
"""

from compare_embeddings import main as run_experiment
from file_helpers import STUDY_ROOT


# 1. Choose which embeddings and cluster counts to compare.
EMBEDDINGS = ['A', 'B', 'C', 'D']
CLUSTER_COUNTS = [2, 3, 5, 8, 13, 21, 34, 55]

# 2. Every embedding uses these same settings.
SEED = 42
OVERLAP = 0.30                 # 30% overlap between neighbouring intervals.
RADIUS_PERCENTILE = 99         # Scales only the lens, not the full vectors.
KMEANS_STARTS = 10
SILHOUETTE_SAMPLE = 5000       # DB index still uses every game in the cell.
STABILITY_SAMPLES = 3          # Repeat on three 80% samples of each cell.
CPU_THREADS = 8

# 3. Full run: leave SAMPLE_GAMES as None.
# For a quick trial, use 1000 and change RESULTS_FOLDER to 'region_smoke'.
# Changed settings require a new results folder; old results stay untouched.
SAMPLE_GAMES = None
RESULTS_FOLDER = 'region_results'


def main():
    results = STUDY_ROOT / RESULTS_FOLDER
    arguments = [
        '--combinations', *EMBEDDINGS,
        '--clusters', *[str(k) for k in CLUSTER_COUNTS],
        '--seeds', str(SEED),
        '--overlap', str(OVERLAP),
        '--n-init', str(KMEANS_STARTS),
        '--sample-size', str(SILHOUETTE_SAMPLE),
        '--resamples', str(STABILITY_SAMPLES),
        '--threads', str(CPU_THREADS),
        '--output-dir', str(results),
        '--english-confirmed',  # Khush has confirmed the current population.
    ]
    if SAMPLE_GAMES is not None:
        arguments.extend(['--limit', str(SAMPLE_GAMES)])
    if RADIUS_PERCENTILE != 99:
        arguments.extend(['--radius-percentile', str(RADIUS_PERCENTILE)])

    # The existing runner performs the numbered steps and reuses completed fits.
    finished = run_experiment(arguments)
    if not finished:
        return  # An existing completed study was opened; leave its files alone.

    # Only after the experiment completes, create the easy-to-read tables.
    from make_comparison_table import save_tables
    save_tables(results)


if __name__ == '__main__':
    main()
