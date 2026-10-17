"""Pair features and macro-F0.5 evaluation for the XGBoost notebook."""
from pathlib import Path
import hashlib
import json
import sqlite3
import time
import numpy as np
import polars as pl
from rapidfuzz import fuzz
from candidate_retrieval import cache_parts, Retriever, COLUMNS

VERSION = '1.0.0'
PAIR_SCHEMA = {'source1_entity_id': pl.String, 'candidate_entity_id': pl.String,
               'routes': pl.Int64, 'best_rank': pl.Int64, 'best_bm25': pl.Float64}
TEXT_COLUMNS = ['entity_id', 'country_key', 'name_norm', 'name_match', 'name_core',
                'address_norm', 'address_match', 'address_numbers', 'address_digit_tokens',
                'postal_candidates', 'name_has_nonlatin', 'address_has_nonlatin']
FEATURE_NAMES = [
    'name_ratio', 'name_token_sort', 'name_token_set', 'name_wratio', 'core_ratio',
    'core_token_sort', 'original_name_ratio', 'address_ratio', 'address_token_sort',
    'address_token_set', 'address_wratio', 'original_address_ratio',
    'name_jaccard', 'name_containment', 'address_jaccard', 'address_containment',
    'number_jaccard', 'number_overlap', 'number_conflict', 'digit_jaccard',
    'postal_overlap', 'postal_conflict', 'name_exact', 'address_exact', 'core_exact',
    'name_length_ratio', 'address_length_ratio', 'name_missing_a', 'name_missing_b',
    'address_missing_a', 'address_missing_b', 'name_nonlatin_a', 'name_nonlatin_b',
    'address_nonlatin_a', 'address_nonlatin_b', 'country_equal', 'target_is_s3',
    'route_name', 'route_address', 'route_gram', 'route_joint', 'reciprocal_rank',
    'log_bm25_magnitude',
]


def atomic_json(path, value):
    path = Path(path)
    temp = path.with_suffix(path.suffix+'.tmp')
    temp.write_text(json.dumps(value, indent=2)+'\n')
    temp.replace(path)


def digest(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True).encode()).hexdigest()


def file_digest(path):
    h = hashlib.sha256()
    with open(path,'rb') as f:
        for block in iter(lambda:f.read(1024*1024), b''): h.update(block)
    return h.hexdigest()


def select_training_anchors(cache_root, count=3000):
    paths, _ = cache_parts(cache_root,'train',1)
    return (pl.scan_parquet(paths).select(COLUMNS)
        .filter(pl.col('entity_id').hash(seed=42)%10 != 0)
        .with_columns(pl.col('entity_id').hash(seed=456).alias('_sample'))
        .sort('_sample').head(count).drop('_sample').collect(engine='streaming'))


def retrieve_training(anchors, index_path, config, out_dir, batch_size=100):
    """Label-blind retrieval with atomic per-anchor-batch checkpoints."""
    out_dir = Path(out_dir); out_dir.mkdir(parents=True,exist_ok=True)
    con=sqlite3.connect(f'{Path(index_path).resolve().as_uri()}?mode=ro',uri=True)
    try: index_signature=con.execute("SELECT value FROM meta WHERE key='signature'").fetchone()[0]
    finally: con.close()
    signature=digest({'anchors':anchors.to_dicts(),'config':config,'index':index_signature,
                      'batch':batch_size, 'retrieval_code':file_digest(Path(__file__).with_name('candidate_retrieval.py'))})
    manifest_path=out_dir/'manifest.json'
    if manifest_path.exists():
        m=json.loads(manifest_path.read_text())
        if m['signature'] != signature: raise ValueError('Training retrieval inputs changed; choose a new TRAIN_ROOT.')
    else:
        m={'signature':signature,'parts':{},'complete':False};atomic_json(manifest_path,m)
    r=Retriever(index_path,**config)
    started=time.perf_counter(); newly_done=0
    try:
        for index,batch in enumerate(anchors.iter_slices(batch_size)):
            key=f'part-{index:05d}.parquet';path=out_dir/key
            if key in m['parts'] and path.exists() and file_digest(path)==m['parts'][key]['sha256']:
                continue
            edges=[]
            for row in batch.iter_rows(named=True):
                edges.extend({'source1_entity_id':row['entity_id'],**p} for p in r.retrieve(row))
            frame=pl.DataFrame(edges,schema=PAIR_SCHEMA) if edges else pl.DataFrame(schema=PAIR_SCHEMA)
            temp=out_dir/(key+'.tmp');frame.write_parquet(temp);temp.replace(path)
            m['parts'][key]={'rows':frame.height,'anchors':batch.height,'sha256':file_digest(path)}
            atomic_json(manifest_path,m)
            newly_done+=batch.height
            done=sum(p['anchors'] for p in m['parts'].values())
            elapsed=time.perf_counter()-started
            print(f'Training retrieval: {done:,}/{anchors.height:,} anchors | ETA {(anchors.height-done)*elapsed/max(newly_done,1)/60:.1f} min',flush=True)
        assert sum(p['anchors'] for p in m['parts'].values())==anchors.height
        m['complete']=True;atomic_json(manifest_path,m)
    finally:r.close()
    return pl.read_parquet([out_dir/k for k in sorted(m['parts'])])


def load_labels(cache_root, anchor_ids):
    folder=Path(cache_root)/'train/train_ground_truth'
    m=json.loads((folder/'manifest.json').read_text());assert m['complete']
    ids=pl.DataFrame({'source1_entity_id':anchor_ids})
    labels=(pl.scan_parquet([folder/k for k in sorted(m['parts'])])
        .join(ids.lazy(),on='source1_entity_id',how='semi')
        .select('source1_entity_id','matched_ids','match_count').collect(engine='streaming'))
    assert labels.height==len(anchor_ids) and labels['source1_entity_id'].n_unique()==len(anchor_ids)
    return labels


def label_pairs(pairs, labels):
    assert pairs.select(pl.struct('source1_entity_id','candidate_entity_id').n_unique()).item()==pairs.height
    positives=(labels.select('source1_entity_id',pl.col('matched_ids').alias('candidate_entity_id'))
        .explode('candidate_entity_id',empty_as_null=True).drop_nulls('candidate_entity_id')
        .with_columns(pl.lit(1,dtype=pl.UInt8).alias('label')))
    return pairs.join(positives,on=['source1_entity_id','candidate_entity_id'],how='left').with_columns(pl.col('label').fill_null(0))


def overlap(a,b):
    a,b=set(a),set(b)
    if not a or not b:return (0.,0.,0.)
    h=len(a&b)
    return h/len(a|b), h/min(len(a),len(b)), float(h==0)


def similarity(fn,a,b):return fn(a,b)/100 if a and b else 0.
def exact(a,b):return float(bool(a) and a==b)
def length_ratio(a,b):return min(len(a),len(b))/max(len(a),len(b)) if a and b else 0.


def features_for_pair(r):
    n,m=r['name_match'],r['name_match_b'];a,b=r['address_match'],r['address_match_b']
    c,d=r['name_core'],r['name_core_b']
    nj,nc,_=overlap(n.split(),m.split());aj,ac,_=overlap(a.split(),b.split())
    numj,numo,numc=overlap(r['address_numbers'],r['address_numbers_b'])
    dij,_,_=overlap(r['address_digit_tokens'],r['address_digit_tokens_b'])
    _,po,pc=overlap(r['postal_candidates'],r['postal_candidates_b'])
    routes=r['routes']
    values=[similarity(fuzz.ratio,n,m),similarity(fuzz.token_sort_ratio,n,m),
        similarity(fuzz.token_set_ratio,n,m),similarity(fuzz.WRatio,n,m),
        similarity(fuzz.ratio,c,d),similarity(fuzz.token_sort_ratio,c,d),
        similarity(fuzz.ratio,r['name_norm'],r['name_norm_b']),
        similarity(fuzz.ratio,a,b),similarity(fuzz.token_sort_ratio,a,b),
        similarity(fuzz.token_set_ratio,a,b),similarity(fuzz.WRatio,a,b),
        similarity(fuzz.ratio,r['address_norm'],r['address_norm_b']),
        nj,nc,aj,ac,numj,float(numo>0),numc,dij,float(po>0),pc,
        exact(n,m),exact(a,b),exact(c,d),length_ratio(n,m),length_ratio(a,b),
        float(not n),float(not m),float(not a),float(not b),
        float(r['name_has_nonlatin']),float(r['name_has_nonlatin_b']),
        float(r['address_has_nonlatin']),float(r['address_has_nonlatin_b']),
        float(r['country_key']==r['country_key_b']),float(r['candidate_entity_id'].startswith('S3-')),
        *[float(bool(routes&bit)) for bit in (1,2,4,8)],
        1/max(r['best_rank'],1),float(np.log1p(abs(r['best_bm25'])))]
    assert len(values)==len(FEATURE_NAMES)
    return values


def feature_cache(cache_root,pairs,out_dir,batch_rows=25000):
    """Same label-free feature extractor is reusable for later inference."""
    out_dir=Path(out_dir);out_dir.mkdir(parents=True,exist_ok=True)
    manifests=[cache_parts(cache_root,'train',s)[1] for s in (1,2,3)]
    signature=digest({'code':file_digest(__file__),'manifests':manifests,
        'pairs_hash':hashlib.sha256(pairs.select(list(PAIR_SCHEMA)).hash_rows(seed=7).to_numpy().tobytes()).hexdigest(),
        'batch_rows':batch_rows,'polars':pl.__version__})
    mp=out_dir/'manifest.json'
    if mp.exists():
        state=json.loads(mp.read_text())
        if state['signature']!=signature:raise ValueError('Feature inputs changed; use a new TRAIN_ROOT.')
    else:state={'signature':signature,'parts':{},'complete':False};atomic_json(mp,state)
    needed=[f'part-{i:05d}.parquet' for i in range((pairs.height+batch_rows-1)//batch_rows)]
    if state['complete'] and set(needed)==set(state['parts']) and all(
        (out_dir/k).exists() and file_digest(out_dir/k)==state['parts'][k]['sha256'] for k in needed):
        return pl.read_parquet([out_dir/k for k in needed])
    ids=pairs.select(pl.col('source1_entity_id').alias('entity_id')).unique()
    parts,_=cache_parts(cache_root,'train',1)
    left=pl.scan_parquet(parts).select(TEXT_COLUMNS).join(ids.lazy(),on='entity_id',how='semi').collect(engine='streaming')
    target_ids=pairs.select(pl.col('candidate_entity_id').alias('entity_id')).unique()
    target_parts=sum([cache_parts(cache_root,'train',s)[0] for s in (2,3)],[])
    right=(pl.scan_parquet(target_parts).select(TEXT_COLUMNS)
           .join(target_ids.lazy(),on='entity_id',how='semi').collect(engine='streaming'))
    assert left.height==ids.height and right.height==target_ids.height,'Missing normalized records'
    for i,batch in enumerate(pairs.iter_slices(batch_rows)):
        key=needed[i];path=out_dir/key
        if key in state['parts'] and path.exists() and file_digest(path)==state['parts'][key]['sha256']:continue
        joined=(batch.join(left,left_on='source1_entity_id',right_on='entity_id',how='left',maintain_order='left')
            .join(right,left_on='candidate_entity_id',right_on='entity_id',how='left',suffix='_b',maintain_order='left'))
        assert joined.height==batch.height
        array=np.asarray([features_for_pair(r) for r in joined.iter_rows(named=True)],dtype=np.float32)
        assert np.isfinite(array).all()
        result=joined.select('source1_entity_id','candidate_entity_id').hstack(pl.DataFrame(array,schema=FEATURE_NAMES,orient='row'))
        temp=out_dir/(key+'.tmp');result.write_parquet(temp);temp.replace(path)
        state['parts'][key]={'rows':result.height,'sha256':file_digest(path)};atomic_json(mp,state)
        print(f'Features: {min((i+1)*batch_rows,pairs.height):,}/{pairs.height:,} pairs',flush=True)
    state['complete']=True;atomic_json(mp,state)
    return pl.read_parquet([out_dir/k for k in needed])


def entity_scores(pair_ids,labels,probabilities,threshold):
    # Missing candidates remain in true_count. Correct singletons score one.
    selected=pair_ids.with_columns(pl.Series('_score',probabilities)).filter(pl.col('_score')>=threshold)
    aggregated=selected.group_by('source1_entity_id').agg(pl.len().alias('predicted'),pl.col('label').sum().alias('tp'))
    counts=(labels.select('source1_entity_id',pl.col('match_count').alias('true_count'))
        .join(aggregated,on='source1_entity_id',how='left').with_columns(pl.col('predicted','tp').fill_null(0)))
    return counts.with_columns(pl.when((pl.col('true_count')==0)&(pl.col('predicted')==0)).then(1.)
        .otherwise(1.25*pl.col('tp')/(.25*pl.col('true_count')+pl.col('predicted'))).alias('f05'))


def tune_threshold(pair_ids,labels,probabilities):
    # Include a value above 1, so the all-empty baseline is part of the search.
    thresholds=np.unique(np.concatenate([np.linspace(.01,.99,99),[.995,.999,1.000001]]))
    rows=[]
    for threshold in thresholds:
        scores=entity_scores(pair_ids,labels,probabilities,float(threshold))
        rows.append({'threshold':float(threshold),'macro_f05':scores['f05'].mean()})
    table=pl.DataFrame(rows).sort(['macro_f05','threshold'],descending=[True,True])
    return table.row(0,named=True)['threshold'],table
