"""Offline, resumable SQLite FTS5 candidate retrieval. No ground truth in retrieval."""
from pathlib import Path
from functools import lru_cache
import hashlib
import json
import sqlite3
import time
import polars as pl

VERSION = '1.0.0'
COLUMNS = ['entity_id', 'country_key', 'name_match', 'address_match', 'name_core']


def cache_parts(cache_root, split, source):
    folder = Path(cache_root) / split / f'{split}_source{source}'
    manifest = json.loads((folder / 'manifest.json').read_text())
    if not manifest['complete']:
        raise ValueError(f'Incomplete normalization: {folder}')
    parts = [folder / name for name in sorted(manifest['parts'])]
    if not all(p.is_file() for p in parts):
        raise FileNotFoundError(f'Missing normalized parts: {folder}')
    return parts, manifest


def name_grams(text):
    # Unicode-aware Python slicing; no external language dictionaries.
    return sorted({'g' + word[i:i+3] for word in text.split() if len(word) >= 4
                   for i in range(len(word)-2)})


def connect_index(path):
    con = sqlite3.connect(path, timeout=60)
    con.execute('PRAGMA journal_mode=WAL')
    con.execute('PRAGMA synchronous=NORMAL')
    con.execute('PRAGMA cache_size=-65536')  # 64 MiB SQLite page cache, not 64M pages.
    con.execute('PRAGMA temp_store=FILE')
    return con


def build_index(cache_root, split, index_path, progress=True):
    """Each source shard and its checkpoint commit in the same transaction."""
    sources = [cache_parts(cache_root, split, s) for s in (2, 3)]
    signature = hashlib.sha256(json.dumps({'version': VERSION, 'sqlite': sqlite3.sqlite_version,
        'manifests': [m for _, m in sources]}, sort_keys=True).encode()).hexdigest()
    path = Path(index_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    con = connect_index(path)
    try:
        con.execute('CREATE TABLE IF NOT EXISTS meta(key TEXT PRIMARY KEY, value TEXT NOT NULL)')
        previous = con.execute("SELECT value FROM meta WHERE key='signature'").fetchone()
        if previous and previous[0] != signature:
            raise ValueError('Inputs/index version changed; choose a new index directory.')
        if previous is None:
            with con:
                con.execute('CREATE TABLE IF NOT EXISTS docs(entity_id TEXT NOT NULL)')
                con.execute("CREATE VIRTUAL TABLE IF NOT EXISTS search USING fts5(country, names, address, grams, content='', tokenize='unicode61 remove_diacritics 2')")
                con.execute("CREATE VIRTUAL TABLE IF NOT EXISTS vocab USING fts5vocab(search, 'col')")
                con.execute('CREATE TABLE IF NOT EXISTS shards(key TEXT PRIMARY KEY, rows INTEGER NOT NULL)')
                con.execute("INSERT INTO meta VALUES('signature',?)", (signature,))
                con.execute("INSERT INTO meta VALUES('complete','false')")
        total = sum(m['total_rows'] for _, m in sources)
        done = dict(con.execute('SELECT key, rows FROM shards'))
        expected_shards = {f'{s}/{p.name}' for s, (parts, _) in zip((2, 3), sources) for p in parts}
        if not set(done).issubset(expected_shards):
            raise ValueError('Index has unexpected checkpoints.')
        if set(done) == expected_shards and sum(done.values()) == total:
            if con.execute('SELECT COUNT(*) FROM docs').fetchone()[0] != total:
                raise ValueError('Index document count disagrees with checkpoints.')
            with con:
                con.execute("UPDATE meta SET value='true' WHERE key='complete'")
            if progress: print(f'{split} index already complete: {total:,} records', flush=True)
            return {'rows': total, 'seconds_this_run': 0, 'path': str(path)}
        started = time.perf_counter()
        written = 0
        next_rowid = con.execute('SELECT COALESCE(MAX(rowid),0)+1 FROM docs').fetchone()[0]
        for source, (parts, manifest) in zip((2, 3), sources):
            for part in parts:
                key = f'{source}/{part.name}'
                if key in done:
                    continue
                frame = pl.read_parquet(part, columns=COLUMNS)
                expected = manifest['parts'][part.name]['rows']
                if frame.height != expected:
                    raise ValueError(f'Normalization row mismatch: {part}')
                # Keep one shard in RAM; SQLite transaction rollback handles interrupts.
                with con:
                    for batch in frame.iter_slices(10_000):
                        docs, terms = [], []
                        for record in batch.iter_rows(named=True):
                            rid = next_rowid
                            next_rowid += 1
                            docs.append((rid, record['entity_id']))
                            terms.append((rid, record['country_key'], record['name_match'],
                                          record['address_match'], ' '.join(name_grams(record['name_core']))))
                        con.executemany('INSERT INTO docs(rowid,entity_id) VALUES(?,?)', docs)
                        con.executemany('INSERT INTO search(rowid,country,names,address,grams) VALUES(?,?,?,?,?)', terms)
                    con.execute('INSERT INTO shards VALUES(?,?)', (key, frame.height))
                done[key] = frame.height
                written += frame.height
                elapsed = time.perf_counter() - started
                rate = written / max(elapsed, .001)
                if progress:
                    print(f'{split} index: {sum(done.values()):,}/{total:,} | {rate:,.0f} rows/s | ETA {(total-sum(done.values()))/max(rate,1)/60:.1f} min', flush=True)
                con.execute('PRAGMA wal_checkpoint(PASSIVE)')
        if sum(done.values()) != total:
            raise ValueError('Index count mismatch.')
        if con.execute('SELECT COUNT(*) FROM docs').fetchone()[0] != total:
            raise ValueError('Index document count mismatch.')
        with con:
            con.execute("UPDATE meta SET value='true' WHERE key='complete'")
        con.execute('PRAGMA wal_checkpoint(TRUNCATE)')
        return {'rows': total, 'seconds_this_run': time.perf_counter()-started, 'path': str(path)}
    finally:
        con.close()


class Retriever:
    """Union BM25 name/address/name-trigram/joint searches within country."""
    def __init__(self, path, name_k=20, address_k=20, gram_k=10, joint_k=10,
                 max_postings=100_000):
        self.con = sqlite3.connect(f'{Path(path).resolve().as_uri()}?mode=ro', uri=True)
        if self.con.execute("SELECT value FROM meta WHERE key='complete'").fetchone() != ('true',):
            self.con.close()
            raise ValueError('Complete the index before querying it.')
        self.config = dict(name_k=name_k, address_k=address_k, gram_k=gram_k,
                           joint_k=joint_k, max_postings=max_postings)
        # Per-instance bounded cache; clear on close, so old connections are not retained.
        self.frequency = lru_cache(maxsize=150_000)(self._frequency)

    def _frequency(self, token, field):
        row = self.con.execute('SELECT doc FROM vocab WHERE term=? AND col=?', (token, field)).fetchone()
        return row[0] if row else 0

    def rare_terms(self, tokens, field, count):
        scored = []
        for token in set(tokens):
            # Single-character address/name terms are weak; gram terms have a g prefix.
            if len(token) < 2:
                continue
            freq = self.frequency(token, field)
            if 0 < freq <= self.config['max_postings']:
                scored.append((freq, token))
        return [token for _, token in sorted(scored)[:count]]

    @staticmethod
    def expression(field, tokens):
        if not tokens:
            return None
        escaped = ['"' + t.replace('"', '""') + '"' for t in tokens]
        return field + ' : (' + ' OR '.join(escaped) + ')'

    def retrieve(self, record):
        names = self.expression('names', self.rare_terms(record['name_match'].split(), 'names', 4))
        address = self.expression('address', self.rare_terms(record['address_match'].split(), 'address', 4))
        grams = self.expression('grams', self.rare_terms(name_grams(record['name_core']), 'grams', 6))
        joint = f'({names}) AND ({address})' if names and address else None
        country = 'country : "' + record['country_key'].replace('"','""') + '"'
        found = {}
        for bit, clause, k in [(1,names,self.config['name_k']), (2,address,self.config['address_k']),
                               (4,grams,self.config['gram_k']), (8,joint,self.config['joint_k'])]:
            if not clause or k <= 0:
                continue
            query = f'({country}) AND ({clause})'
            rows = self.con.execute('SELECT rowid, rank FROM search WHERE search MATCH ? ORDER BY rank, rowid LIMIT ?', (query,k))
            for rank, (rid, score) in enumerate(rows, start=1):
                value = found.setdefault(rid, {'routes': 0, 'best_rank': rank, 'best_bm25': score})
                value['routes'] |= bit
                value['best_rank'] = min(rank, value['best_rank'])
                value['best_bm25'] = min(score, value['best_bm25'])
        if not found:
            return []
        marks = ','.join('?' for _ in found)
        ids = dict(self.con.execute(f'SELECT rowid,entity_id FROM docs WHERE rowid IN ({marks})', list(found)))
        return [{'candidate_entity_id': ids[rid], **features} for rid, features in sorted(
            found.items(), key=lambda item: (item[1]['best_rank'], item[1]['best_bm25'], ids[item[0]]))]

    def close(self):
        self.frequency.cache_clear()
        self.con.close()


def sample_anchors(cache_root, count=5000):
    parts, _ = cache_parts(cache_root, 'train', 1)
    # Reserved validation anchors: 10% by stable Polars hash in the pinned environment.
    return (pl.scan_parquet(parts).select(COLUMNS)
        .filter(pl.col('entity_id').hash(seed=42) % 10 == 0)
        .with_columns(pl.col('entity_id').hash(seed=123).alias('_sample'))
        .sort('_sample').head(count).drop('_sample').collect(engine='streaming'))


def evaluate_pilot(cache_root, retriever, anchors, out_dir):
    """Labels are consulted ONLY after retrieval. No true matches are injected."""
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    started = time.perf_counter()
    results = {}
    edges = []
    for index, record in enumerate(anchors.iter_rows(named=True), start=1):
        pairs = retriever.retrieve(record)
        results[record['entity_id']] = {p['candidate_entity_id'] for p in pairs}
        edges.extend({'source1_entity_id': record['entity_id'], **pair} for pair in pairs)
        if index % 250 == 0:
            print(f'Retrieved {index:,}/{anchors.height:,} anchors in {time.perf_counter()-started:.1f}s', flush=True)
    retrieval_seconds = time.perf_counter()-started
    folder = Path(cache_root) / 'train/train_ground_truth'
    manifest = json.loads((folder / 'manifest.json').read_text())
    if not manifest['complete']:
        raise ValueError('Ground truth cache is incomplete.')
    labels = pl.scan_parquet([folder / k for k in sorted(manifest['parts'])]).join(
        anchors.select(pl.col('entity_id').alias('source1_entity_id')).lazy(),
        on='source1_entity_id', how='semi').select('source1_entity_id','matched_ids').collect()
    if labels.height != anchors.height:
        raise ValueError('Pilot anchors are missing ground truth rows.')
    truth = dict(labels.iter_rows())
    rows = []
    for anchor in anchors.iter_rows(named=True):
        aid = anchor['entity_id']
        actual, candidates = set(truth[aid]), results[aid]
        hits = actual & candidates
        n, h = len(actual), len(hits)
        rows.append({'source1_entity_id': aid, 'country': anchor['country_key'],
            'true_matches': n, 'retrieved_matches': h, 'candidates': len(candidates),
            'recall': h/n if n else None,
            'oracle_macro_f05': 1.25*h/(.25*n+h) if n else 1.0,
            'missing_true_ids': sorted(actual-candidates)})
    details = pl.DataFrame(rows)
    positives = details['true_matches'].sum()
    stats = {'anchors': anchors.height, 'true_pairs': positives,
        'retrieved_true_pairs': details['retrieved_matches'].sum(),
        'pair_recall': details['retrieved_matches'].sum()/positives if positives else None,
        'macro_recall_non_singletons': details['recall'].mean(),
        'oracle_macro_f05_ceiling': details['oracle_macro_f05'].mean(),
        'candidate_pairs': len(edges), 'mean_candidates': details['candidates'].mean(),
        'zero_candidate_anchors': (details['candidates']==0).sum(),
        'retrieval_seconds': retrieval_seconds, 'retriever_config': retriever.config,
        'note': 'Oracle ceiling assumes a perfect classifier; this is NOT a trained-model score.'}
    schema = {'source1_entity_id': pl.String, 'candidate_entity_id': pl.String,
              'routes': pl.Int64, 'best_rank': pl.Int64, 'best_bm25': pl.Float64}
    candidates = pl.DataFrame(edges, schema=schema) if edges else pl.DataFrame(schema=schema)
    for frame, name in [(candidates, 'pilot_candidates.parquet'),(details,'pilot_recall.parquet')]:
        temp = out_dir / (name+'.tmp')
        frame.write_parquet(temp)
        temp.replace(out_dir / name)
    temp = out_dir / 'pilot_metrics.json.tmp'
    temp.write_text(json.dumps(stats,indent=2)+'\n')
    temp.replace(out_dir/'pilot_metrics.json')
    return stats, details
