"""Export saved full feature matrices to CSV without recalculating embeddings.

Run: python -m clustering.export_embeddings --output-dir clustering/outputs
The CSV coordinates are the exact weighted feature space used by K-means.
"""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
import json
from pathlib import Path

import numpy as np
import pandas as pd

from .pipeline import PROJECT_ROOT, app_ids_sha256, file_sha256, write_json


def export_one(root: Path, combination: str, index: pd.DataFrame, overwrite=False):
    directory = root / combination
    metadata = json.loads((directory / 'features.json').read_text(encoding='utf-8'))
    matrix_path = directory / 'features.npy'
    if file_sha256(matrix_path) != metadata['feature_sha256']:
        raise ValueError(f'{combination}: feature checksum mismatch.')
    values = np.load(matrix_path, mmap_mode='r', allow_pickle=False)
    if values.shape != (len(index), metadata['dimension']):
        raise ValueError(f'{combination}: feature dimensions disagree with the game index.')
    if app_ids_sha256(index['AppID']) != metadata['source_signature']['app_ids_sha256']:
        raise ValueError(f'{combination}: AppID order disagrees with feature provenance.')
    columns = ['categorical__' + label for label in metadata['source_signature']['descriptor_columns']]
    if metadata['weights']['description']:
        columns += [f'description__{i:03d}' for i in range(metadata['description']['dimension'])]
    if metadata['weights']['popularity']:
        columns += ['popularity__weighted']
    if len(columns) != values.shape[1] or len(set(columns)) != len(columns):
        raise ValueError(f'{combination}: feature header mismatch.')
    target = directory / 'embeddings.csv'
    if target.exists() and not overwrite:
        raise ValueError(f'{target} already exists; use --overwrite to replace it.')
    temporary = target.with_name(target.name + '.partial')
    chunk_size = 5000
    print(f'{combination}: exporting {len(index):,} rows x {values.shape[1]} feature columns', flush=True)
    with temporary.open('w', encoding='utf-8-sig', newline='') as stream:
        for start in range(0, len(index), chunk_size):
            block = np.asarray(values[start:start + chunk_size])
            if not np.isfinite(block).all():
                raise ValueError(f'{combination}: nonfinite feature values.')
            frame = pd.DataFrame(block, columns=columns)
            frame.insert(0, 'Name', index['Name'].iloc[start:start + len(block)].to_numpy())
            frame.insert(0, 'AppID', index['AppID'].iloc[start:start + len(block)].to_numpy())
            # Nine significant decimal digits preserve float32 values on reload.
            frame.to_csv(stream, index=False, header=(start == 0), float_format='%.9g', lineterminator='\n')
            if start == 0:
                frame.head(10).to_csv(directory/'embeddings_preview.csv', index=False,
                                      encoding='utf-8-sig', float_format='%.9g', lineterminator='\n')
            if (start // chunk_size) % 5 == 0 or start + len(block) == len(index):
                print(f'{combination}: wrote {min(start + chunk_size, len(index)):,}/{len(index):,} rows', flush=True)
    print(f'{combination}: checking every exported row and coordinate against features.npy', flush=True)
    rows_checked = 0
    dtype = {column: np.float32 for column in columns}
    dtype.update({'AppID': np.int64, 'Name': str})
    for frame in pd.read_csv(temporary, chunksize=chunk_size, dtype=dtype,
                             keep_default_na=False, encoding='utf-8-sig', float_precision='round_trip'):
        if frame.columns.tolist() != ['AppID', 'Name'] + columns:
            raise ValueError(f'{combination}: exported column order mismatch.')
        stop = rows_checked + len(frame)
        expected_index = index.iloc[rows_checked:stop]
        if not np.array_equal(frame['AppID'].to_numpy(), expected_index['AppID'].to_numpy()):
            raise ValueError(f'{combination}: exported AppID mismatch.')
        if frame['Name'].tolist() != expected_index['Name'].tolist():
            raise ValueError(f'{combination}: exported game name mismatch.')
        if not np.array_equal(frame[columns].to_numpy(dtype=np.float32), values[rows_checked:stop]):
            raise ValueError(f'{combination}: exported numeric values do not round-trip.')
        rows_checked = stop
    if rows_checked != len(index):
        raise ValueError(f'{combination}: exported row count mismatch.')
    temporary.replace(target)
    result = {'combination': combination, 'file': str(target.resolve()),
              'preview_file': str((directory/'embeddings_preview.csv').resolve()),
              'rows': rows_checked, 'feature_columns': len(columns), 'total_columns': len(columns)+2,
              'bytes': target.stat().st_size, 'csv_sha256': file_sha256(target),
              'source_feature_sha256': metadata['feature_sha256'],
              'every_coordinate_matches_float32_source': True,
              'format': 'UTF-8 with BOM, comma-separated, nine significant decimal digits',
              'column_order': 'AppID, Name, categorical block, description block if present, popularity if present'}
    write_json(directory/'embeddings_csv_manifest.json', result)
    print(f'{combination}: verified and saved {target} ({target.stat().st_size / 1024**2:.1f} MiB)', flush=True)
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output-dir', type=Path, default=PROJECT_ROOT/'clustering'/'outputs')
    parser.add_argument('--combinations', nargs='+', choices=list('ABCD'), default=list('ABCD'))
    parser.add_argument('--overwrite', action='store_true')
    args = parser.parse_args()
    root = args.output_dir.resolve()
    index = pd.read_csv(root/'game_index.csv', dtype={'AppID': np.int64, 'Name': str}, keep_default_na=False)
    if index['AppID'].duplicated().any():
        parser.error('Duplicate AppIDs in the saved game index.')
    records = []
    try:
        for combination in dict.fromkeys(args.combinations):
            records.append(export_one(root, combination, index, args.overwrite))
        write_json(root/'embedding_csv_exports.json', {
            'exported_at_utc': datetime.now(timezone.utc).isoformat(), 'exports': records,
            'meaning': 'Saved weighted embeddings used by K-means, not the original unnormalized source descriptors.'})
    except (ValueError, OSError) as exc:
        parser.exit(2, f'CSV export failed: {exc}\n')
    print(f'Completed and verified {len(records)} full CSV exports.', flush=True)


if __name__ == '__main__':
    main()
