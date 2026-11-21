"""Compare A/B/C/D using a 2D lens, nine overlapping cells and local KMeans.

Run from the project root: python "embedding study/compare_embeddings.py" --limit 1000
Read main() below for the complete control flow.
"""

import argparse
from pathlib import Path

import numpy as np
import pandas as pd
from threadpoolctl import threadpool_limits

from feature_vectors import read_features, COMBINATIONS
from file_helpers import (PROJECT_ROOT, STUDY_ROOT, file_sha256, read_json,
                          write_json, write_csv, package_versions, fingerprint, app_ids_sha256)
from read_data import load_inputs
from region_functions import (make_lens, make_regions, fit_groups, score_groups,
                               cluster_profiles, check_stability)


def summarize(scores):
    """Keep coverage visible alongside macro means of LOCAL cell scores."""
    rows = []
    for (combination, k, seed), group in scores.groupby(['combination', 'k', 'seed']):
        row = {'combination': combination, 'k': int(k), 'seed': int(seed),
               'regions': len(group), 'valid_regions': int((group.status == 'ok').sum())}
        row['valid_fraction'] = row['valid_regions'] / row['regions']
        for metric in ['silhouette', 'davies_bouldin']:
            values = pd.to_numeric(group[metric], errors='coerce').dropna()
            row[metric + '_valid_regions'] = len(values)
            row[metric + '_local_macro_mean'] = values.mean() if len(values) else None
            row[metric + '_local_std'] = values.std(ddof=0) if len(values) else None
        rows.append(row)
    return pd.DataFrame(rows)


def read_settings(argv):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--features-dir', type=Path, default=STUDY_ROOT / 'outputs')
    parser.add_argument('--cleaned', type=Path, default=PROJECT_ROOT / 'Data Prep/games_cleaned.csv')
    parser.add_argument('--descriptors', type=Path, default=PROJECT_ROOT / 'Data Prep/games_descriptors.csv')
    parser.add_argument('--output-dir', type=Path, default=STUDY_ROOT / 'region_results')
    parser.add_argument('--combinations', nargs='+', choices=list(COMBINATIONS), default=list(COMBINATIONS))
    parser.add_argument('--clusters', nargs='+', type=int, default=[2, 3, 5, 8, 13, 21, 34, 55])
    parser.add_argument('--seeds', nargs='+', type=int, default=[42])
    parser.add_argument('--overlap', type=float, default=0.30)
    parser.add_argument('--radius-percentile', type=float, default=99)
    parser.add_argument('--sample-size', type=int, default=5000)
    parser.add_argument('--limit', type=int, help='Fixed random subset for a smoke test only.')
    parser.add_argument('--threads', type=int, default=8)
    parser.add_argument('--n-init', type=int, default=10)
    parser.add_argument('--resamples', type=int, default=0, help='0 disables stability; otherwise at least 2.')
    parser.add_argument('--sample-fraction', type=float, default=0.8)
    parser.add_argument('--english-confirmed', action='store_true', help='Team has approved input population as English-only.')
    args = parser.parse_args(argv)
    if not (0 < args.overlap < 1 and 0 < args.radius_percentile <= 100
            and args.sample_size >= 2 and args.threads >= 1 and args.n_init >= 1
            and 0 < args.sample_fraction < 1 and (args.resamples == 0 or args.resamples >= 2)
            and all(k >= 2 for k in args.clusters) and all(s >= 0 for s in args.seeds)
            and (args.limit is None or args.limit >= 3)):
        parser.error('Invalid experiment settings.')
    for values in [args.combinations, args.clusters, args.seeds]:
        if len(values) != len(set(values)):
            parser.error('Do not repeat combinations, cluster counts or seeds.')
    return args


def main(argv=None):
    args = read_settings(argv)
    if (args.output_dir / 'experiment.json').exists():
        existing = read_json(args.output_dir / 'experiment.json')
        if existing.get('status') == 'complete':
            print('This results folder contains a completed study. No training was started.')
            print('Read results.md, or use --output-dir with a NEW folder for another experiment.')
            return False
    # 1. Check source data and match saved vectors by AppID, never by guesswork.
    data = load_inputs(args.cleaned, args.descriptors)
    index = pd.read_csv(args.features_dir / 'game_index.csv')
    if index.AppID.duplicated().any() or set(index.AppID) != set(data['frame'].AppID):
        raise ValueError('Saved embeddings and source files have different AppIDs.')
    positions = data['frame'].set_index('AppID').index.get_indexer(index.AppID)
    games = data['frame'].iloc[positions].reset_index(drop=True)
    terms = data['categorical'][positions] > 0
    if args.limit is not None and args.limit > len(index):
        raise ValueError('--limit exceeds available games.')
    selected = np.arange(len(index))
    if args.limit is not None:
        selected = np.sort(np.random.default_rng(42).choice(len(index), args.limit, replace=False))
    games, terms = games.iloc[selected].reset_index(drop=True), terms[selected]
    population_prevalence = terms.mean(axis=0)
    # One random game priority is shared across configurations, k values and seeds.
    priority = np.random.default_rng(42).permutation(len(games))
    settings = {key: str(value) if isinstance(value, Path) else value
                for key, value in vars(args).items()}
    settings.update({'source_signature': data['source_signature'],
                     'selected_appids_sha256': app_ids_sha256(games.AppID),
                     'code_sha256': {name: file_sha256(Path(__file__).parent / name)
                                     for name in ['region_functions.py', 'compare_embeddings.py']},
                     'versions': package_versions(), 'lens_seed': 42,
                     'sample_seed': 42, 'max_iter': 300, 'tol': 1e-4,
                     'algorithm': 'lloyd', 'silhouette': 'fixed sample; Euclidean',
                     'profile_method': 'descriptor prevalence minus experiment population prevalence'})
    settings_hash = fingerprint(settings)
    folder = args.output_dir
    folder.mkdir(parents=True, exist_ok=True)
    manifest_path = folder / 'experiment.json'
    if manifest_path.exists() and read_json(manifest_path)['fingerprint'] != settings_hash:
        raise ValueError('Output folder has different settings or code. Choose a new --output-dir.')
    write_json(manifest_path, {'fingerprint': settings_hash, 'settings': settings,
                              'status': 'running', 'scope': 'local cover evaluation; no graph',
                              'population_status': 'team confirmed' if args.english_confirmed else 'English-only population UNCONFIRMED',
                              'smoke_test': args.limit is not None})
    write_csv(folder / 'selected_games.csv', games[['AppID', 'Name']])
    scores, stability = [], []
    with threadpool_limits(limits=args.threads):
        for combination in args.combinations:
            # 2. Load a verified embedding, fit its lens and create nine cells.
            saved, metadata = read_features(args.features_dir, combination, index.AppID.astype(str).tolist())
            for key in ['cleaned_sha256', 'descriptors_sha256']:
                if metadata['source_signature'][key] != data['source_signature'][key]:
                    raise ValueError(f'{combination}: inputs changed; rebuild embeddings first.')
            vectors = np.array(saved[selected], dtype=np.float32)
            target = folder / combination
            target.mkdir(exist_ok=True)
            lens, mean, direction, details = make_lens(vectors, args.radius_percentile)
            np.savez(target / 'lens_fit.npz', mean=mean, direction=direction)
            lens_table = games[['AppID', 'Name']].copy()
            lens_table[['raw_radius', 'scaled_radius', 'alignment']] = lens
            write_csv(target / 'lens.csv', lens_table)
            regions = make_regions(lens, args.overlap)
            write_json(target / 'lens_and_cover.json', {**details, 'feature_metadata': metadata,
                       'regions': [{k: v for k, v in region.items() if k != 'indices'} for region in regions]})
            memberships = []
            for region in regions:
                ids = region['indices']
                memberships.extend({'region': region['region'], 'AppID': int(games.AppID.iloc[i])} for i in ids)
                local = vectors[ids]
                local_games = games.iloc[ids].reset_index(drop=True)
                sample = np.sort(np.argsort(priority[ids])[:args.sample_size])
                # 3. Try each requested k on FULL vectors inside this cell.
                for k in args.clusters:
                    for seed in args.seeds:
                        run_folder = target / region['region'] / f'k_{k:02d}_seed_{seed}'
                        run_folder.mkdir(parents=True, exist_ok=True)
                        report = run_folder / 'result.json'
                        if report.exists():
                            old = read_json(report)
                            if all((run_folder / name).exists() and file_sha256(run_folder / name) == digest
                                   for name, digest in old['files'].items()):
                                scores.append(old['score'])
                                stability.extend(old['stability'])
                                print(f'{combination} {region["region"]} k={k}: reused', flush=True)
                                continue
                        score = {'combination': combination, 'region': region['region'],
                                 'k': k, 'seed': seed, 'games': len(ids), 'actual_clusters': 0,
                                 'silhouette': None, 'davies_bouldin': None,
                                 'status': 'invalid', 'reason': '', 'sample_rows': len(sample)}
                        files, pairs = {}, []
                        if len(ids) <= k:
                            score['reason'] = 'Too few games: metrics require games > requested k.'
                        else:
                            model = fit_groups(local, k, seed, args.n_init)
                            score.update(score_groups(local, model.labels_, sample))
                            labels = local_games[['AppID', 'Name']].copy()
                            labels['cluster'] = model.labels_
                            write_csv(run_folder / 'labels.csv', labels)
                            np.save(run_folder / 'centers.npy', model.cluster_centers_)
                            profiles = cluster_profiles(local, model.labels_, model.cluster_centers_,
                                                        local_games, terms[ids], data['descriptor_columns'], population_prevalence)
                            write_json(run_folder / 'profiles.json', {'clusters': profiles})
                            pairs = check_stability(local, k, seed, args.resamples,
                                                    args.sample_fraction, args.n_init)
                            pairs = [{**p, 'combination': combination, 'region': region['region'],
                                      'k': k, 'seed': seed} for p in pairs]
                            files = {name: file_sha256(run_folder / name)
                                     for name in ['labels.csv', 'centers.npy', 'profiles.json']}
                        write_json(report, {'score': score, 'files': files, 'stability': pairs})
                        scores.append(score)
                        stability.extend(pairs)
                        print(f'{combination} {region["region"]} k={k}: {score["status"]}', flush=True)
                # Save progress after every cell so an interrupted run is useful.
                write_csv(folder / 'cell_scores.csv', pd.DataFrame(scores))
            write_csv(target / 'memberships.csv', pd.DataFrame(memberships, columns=['region', 'AppID']))
    # 4. Report local scores with coverage, without claiming a global score/winner.
    frame = pd.DataFrame(scores)
    write_csv(folder / 'comparison.csv', summarize(frame))
    write_csv(folder / 'stability.csv', pd.DataFrame(stability, columns=[
        'combination', 'region', 'k', 'seed', 'repeat_a', 'repeat_b', 'shared_games', 'ari', 'reason']))
    if stability:
        stability_rows = []
        for keys, group in pd.DataFrame(stability).groupby(['combination', 'k', 'seed']):
            valid = group.dropna(subset=['ari'])
            stability_rows.append({'combination': keys[0], 'k': int(keys[1]), 'seed': int(keys[2]),
                                   'attempted_pairs': len(group), 'valid_pairs': len(valid),
                                   'eligible_regions': valid.region.nunique(), 'regions': 9,
                                   'ari_mean': valid.ari.mean(), 'ari_std': valid.ari.std(ddof=0)})
        write_csv(folder / 'stability_summary.csv', pd.DataFrame(stability_rows))
    manifest = read_json(manifest_path)
    manifest['status'] = 'complete'
    manifest['attempted_local_fits'] = len(scores)
    manifest['stability_status'] = 'completed' if args.resamples else 'not requested; no stability claim'
    write_json(manifest_path, manifest)
    print(f'Completed. Comparison: {folder / "comparison.csv"}', flush=True)
    return True


if __name__ == '__main__':
    main()
