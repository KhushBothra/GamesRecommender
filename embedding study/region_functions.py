"""Small steps for the overlapping-region experiment. No Mapper graph is built."""

import numpy as np
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from sklearn.metrics import adjusted_rand_score, davies_bouldin_score, silhouette_score


def make_lens(vectors, percentile=99, seed=42):
    """Return distance/alignment coordinates and the fitted mean/PC1."""
    mean = vectors.mean(axis=0, dtype=np.float64)
    centered = np.asarray(vectors - mean, dtype=np.float32)
    radius = np.sqrt(np.einsum('ij,ij->i', centered, centered, dtype=np.float64))
    if np.any(radius > 0):
        pca = PCA(n_components=1, svd_solver='randomized', random_state=seed)
        direction = pca.fit(centered).components_[0].astype(np.float64)
        direction /= np.linalg.norm(direction)
        # PCA can return either sign. Always make its largest component positive.
        if direction[np.argmax(np.abs(direction))] < 0:
            direction *= -1
    else:
        direction = np.zeros(vectors.shape[1])
        direction[0] = 1
    alignment = np.zeros(len(vectors))
    np.divide(centered @ direction, radius, out=alignment, where=radius > 0)
    alignment = np.clip(alignment, -1, 1)
    scale = float(np.percentile(radius, percentile))
    if scale == 0:
        scale = float(radius.max()) or 1.0
    scaled_radius = np.clip(radius / scale, 0, 1)
    details = {'radius_scale': scale, 'exact_mean_games': int((radius == 0).sum()),
               'zero_vectors': int((np.linalg.norm(vectors, axis=1) == 0).sum()),
               'radius_clipped_games': int((radius > scale).sum()),
               'pc1_sign': 'largest absolute component is positive'}
    return np.column_stack([radius, scaled_radius, alignment]), mean, direction, details


def intervals(low, high, overlap):
    """Three equal-width intervals; overlap is a fraction of interval width."""
    width = (high - low) / (3 - 2 * overlap)
    step = width * (1 - overlap)
    return [(low + i * step, high if i == 2 else low + i * step + width)
            for i in range(3)]


def make_regions(lens, overlap=0.30):
    """Each game can appear in several rectangles. Bounds are inclusive."""
    regions = []
    for x, (left, right) in enumerate(intervals(0, 1, overlap)):
        for y, (bottom, top) in enumerate(intervals(-1, 1, overlap)):
            mask = ((lens[:, 1] >= left) & (lens[:, 1] <= right)
                    & (lens[:, 2] >= bottom) & (lens[:, 2] <= top))
            regions.append({'region': f'x{x + 1}_y{y + 1}',
                            'bounds': [left, right, bottom, top],
                            'indices': np.flatnonzero(mask)})
    return regions


def fit_groups(vectors, k, seed, n_init=10):
    return KMeans(n_clusters=k, random_state=seed, n_init=n_init,
                  max_iter=300, tol=1e-4, algorithm='lloyd').fit(vectors)


def score_groups(vectors, labels, sample):
    """DB uses the complete cell; silhouette uses a fixed cell sample."""
    result = {'actual_clusters': int(np.unique(labels).size),
              'davies_bouldin': None, 'silhouette': None,
              'sample_rows': len(sample), 'status': 'invalid', 'reason': ''}
    if not 2 <= result['actual_clusters'] < len(vectors):
        result['reason'] = 'Need at least two non-empty clusters and fewer clusters than games.'
        return result
    result['davies_bouldin'] = float(davies_bouldin_score(vectors, labels))
    sample_labels = labels[sample]
    result['sample_clusters'] = int(np.unique(sample_labels).size)
    if not 2 <= result['sample_clusters'] < len(sample):
        result['status'] = 'partial'
        result['reason'] = 'Silhouette sample does not contain a valid partition.'
        return result
    result['silhouette'] = float(silhouette_score(vectors[sample], sample_labels,
                                                metric='euclidean'))
    result['status'] = 'ok'
    return result


def cluster_profiles(vectors, labels, centers, games, terms, term_names, population):
    """Describe clusters by positive prevalence difference and nearest games."""
    profiles = []
    for label in np.unique(labels):
        members = np.flatnonzero(labels == label)
        prevalence = terms[members].mean(axis=0)
        difference = prevalence - population
        top = [int(i) for i in np.argsort(-difference, kind='stable')[:5]
               if difference[i] > 0]
        distances = np.sum((vectors[members] - centers[label]) ** 2, axis=1)
        examples = members[np.argsort(distances, kind='stable')[:3]]
        profiles.append({'cluster': int(label), 'games': len(members),
                         'terms': [{'term': term_names[i],
                                    'cluster_prevalence': float(prevalence[i]),
                                    'population_prevalence': float(population[i]),
                                    'difference': float(difference[i])} for i in top],
                         'examples': games.iloc[examples][['AppID', 'Name']].to_dict('records')})
    return profiles


def check_stability(vectors, k, seed, repeats, fraction, n_init):
    """Subsample a fixed cell. Compare shared rows using label-invariant ARI."""
    runs, results = [], []
    count = int(len(vectors) * fraction)
    if repeats < 2:
        return results
    for repeat in range(repeats):
        rows = np.sort(np.random.default_rng(seed + repeat).choice(len(vectors), count, replace=False))
        labels = None
        if k < count:
            labels = fit_groups(vectors[rows], k, seed, n_init).labels_
        runs.append((rows, labels))
    for a in range(repeats):
        for b in range(a + 1, repeats):
            common, ia, ib = np.intersect1d(runs[a][0], runs[b][0], return_indices=True)
            record = {'repeat_a': a, 'repeat_b': b, 'shared_games': len(common),
                      'ari': None, 'reason': ''}
            if runs[a][1] is None or runs[b][1] is None:
                record['reason'] = 'Subsample too small for requested k.'
            else:
                la, lb = runs[a][1][ia], runs[b][1][ib]
                if not (2 <= len(np.unique(la)) < len(common)
                        and 2 <= len(np.unique(lb)) < len(common)):
                    record['reason'] = 'Shared games do not contain valid partitions.'
                else:
                    record['ari'] = float(adjusted_rand_score(la, lb))
            results.append(record)
    return results
