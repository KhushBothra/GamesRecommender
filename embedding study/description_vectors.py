"""Clean descriptions, reuse checked vectors, or encode batches with MiniLM.

The study uses truncation at 256 tokens. No chunking or external-vector mode.
"""
import hashlib
import json
from html.parser import HTMLParser
from pathlib import Path
import re

import numpy as np
from file_helpers import file_sha256, fingerprint, write_json, package_versions

DEFAULT_MODEL = 'sentence-transformers/all-MiniLM-L6-v2'
CLEANING_VERSION = 'html_text_without_script_style_whitespace_v1'


class HTMLText(HTMLParser):
    """Keep text outside script/style tags and put spaces between HTML tags."""
    def __init__(self):
        super().__init__(convert_charrefs=True)
        self.parts = []
        self.hidden = []

    def handle_starttag(self, tag, attrs):
        if tag in {'script', 'style'}:
            self.hidden.append(tag)
        elif not self.hidden:
            self.parts.append(' ')

    def handle_startendtag(self, tag, attrs):
        if not self.hidden:
            self.parts.append(' ')

    def handle_endtag(self, tag):
        if self.hidden:
            if tag == self.hidden[-1]:
                self.hidden.pop()
        else:
            self.parts.append(' ')

    def handle_data(self, data):
        if not self.hidden:
            self.parts.append(data)


def clean_description(value):
    if not isinstance(value, str):
        raise ValueError('Descriptions must be non-missing strings.')
    parser = HTMLText()
    parser.feed(value)
    parser.close()
    return re.sub(r'\s+', ' ', ''.join(parser.parts)).strip()


def prepare_text(frame):
    raw_ids = list(frame['AppID'])
    ids = np.asarray(raw_ids, dtype='<i8')
    if any(isinstance(value, (bool, np.bool_)) or int(value) != value for value in raw_ids):
        raise ValueError('Description AppIDs must be integers.')
    if len(ids) == 0 or len(set(ids)) != len(ids) or (ids <= 0).any():
        raise ValueError('Description AppIDs must be positive and unique.')
    texts = [clean_description(value) for value in frame['Description']]
    if len(texts) != len(ids) or any(not text for text in texts):
        raise ValueError('Missing or empty cleaned description.')
    digest = hashlib.sha256()
    for text in texts:
        encoded = text.encode('utf-8')
        digest.update(len(encoded).to_bytes(8, 'little'))
        digest.update(encoded)
    return ids, texts, hashlib.sha256(ids.tobytes()).hexdigest(), digest.hexdigest()


def load_cache(path, expected):
    """Never accept cached descriptions for different games, texts or settings."""
    info = json.loads(path.with_suffix('.json').read_text(encoding='utf-8'))
    for key, value in expected.items():
        if key not in info or info[key] != value:
            raise ValueError(f'Description cache mismatch: {key}')
    if file_sha256(path) != info['embedding_sha256']:
        raise ValueError('Description cache checksum mismatch.')
    vectors = np.load(path, allow_pickle=False)
    if vectors.shape != (expected['rows'], info['dimension']) or vectors.dtype != np.float32:
        raise ValueError('Description cache has the wrong shape or type.')
    if not np.isfinite(vectors).all():
        raise ValueError('Description cache contains NaN or infinite values.')
    norms = np.linalg.norm(vectors.astype(np.float64), axis=1)
    if not np.allclose(norms, 1, rtol=1e-4, atol=1e-5):
        raise ValueError('Description vectors must have length 1.')
    return vectors, info


def prepare_descriptions(frame, args):
    # 1. Build the same text/settings identity used by our saved descriptions.
    ids, texts, ids_hash, text_hash = prepare_text(frame)
    expected = {
        'schema_version': 1, 'app_ids_sha256': ids_hash,
        'description_text_sha256': text_hash, 'model_id': DEFAULT_MODEL,
        'model_revision': None, 'max_seq_length': 256, 'pooling': 'truncate',
        'rows': len(texts), 'cleaning_version': CLEANING_VERSION, 'normalized': True,
    }
    cache = Path(args.cache_dir)
    path = cache / f'descriptions_{fingerprint(expected)}.npy'
    if path.exists() or path.with_suffix('.json').exists():
        return load_cache(path, expected)

    # 2. Load only when the correct saved description vectors are unavailable.
    from sentence_transformers import SentenceTransformer
    import torch
    torch.set_num_threads(args.threads)
    model = SentenceTransformer(DEFAULT_MODEL, revision=None, device='cpu',
                               cache_folder=str(cache / 'huggingface'),
                               local_files_only=args.offline)
    model.max_seq_length = 256
    cache.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix('.partial.npy')
    vectors = None
    try:
        # 3. Encode a small batch, normalize it and write it directly to disk.
        for start in range(0, len(texts), args.batch_size):
            batch = texts[start:start + args.batch_size]
            numbers = np.asarray(model.encode(batch, batch_size=args.batch_size,
                show_progress_bar=False, convert_to_numpy=True,
                normalize_embeddings=False), dtype=np.float32)
            if numbers.ndim != 2 or len(numbers) != len(batch) or not np.isfinite(numbers).all():
                raise ValueError('Encoder returned an invalid matrix.')
            norms = np.linalg.norm(numbers.astype(np.float64), axis=1)
            if (norms <= 0).any():
                raise ValueError('Encoder returned an empty vector.')
            numbers = np.asarray(numbers / norms[:, None], dtype=np.float32)
            if vectors is None:
                vectors = np.lib.format.open_memmap(temporary, mode='w+', dtype=np.float32,
                                                   shape=(len(texts), numbers.shape[1]))
            vectors[start:start + len(batch)] = numbers
            print(f'Descriptions: {start + len(batch):,}/{len(texts):,}', flush=True)
        dimension = vectors.shape[1]
        vectors.flush()
        vectors._mmap.close()
        vectors = None
        temporary.replace(path)
        # 4. Save metadata last. An incomplete cache must not be reused.
        try:
            revision = model[0].auto_model.config._commit_hash
        except (AttributeError, IndexError, TypeError, KeyError):
            revision = None
        info = {**expected, 'identity_fingerprint': fingerprint(expected),
                'dimension': dimension, 'embedding_sha256': file_sha256(path),
                'resolved_model_revision': revision, 'dtype': 'float32',
                'torch_threads': args.threads, 'versions': package_versions()}
        write_json(path.with_suffix('.json'), info)
    finally:
        if vectors is not None:
            vectors._mmap.close()
        temporary.unlink(missing_ok=True)
    return load_cache(path, expected)
