"""Batched test scoring, resumable output, and packaging helpers for notebook 04."""
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor
import csv
import hashlib
import json
import shutil
import sqlite3
import threading
import time
import zlib
import numpy as np
import polars as pl
import xgboost as xgb
from candidate_retrieval import cache_parts
from fast_retrieval import FastRetriever
from train_matcher import TEXT_COLUMNS, FEATURE_NAMES, features_for_pair, file_digest, atomic_json, digest

VERSION='1.0.0'


def build_target_store(cache_root,split,path):
    path=Path(path);path.parent.mkdir(parents=True,exist_ok=True)
    sources=[cache_parts(cache_root,split,s) for s in (2,3)]
    signature=digest({'version':VERSION,'columns':TEXT_COLUMNS,'manifests':[m for _,m in sources]})
    c=sqlite3.connect(path)
    try:
        c.execute('PRAGMA journal_mode=WAL');c.execute('PRAGMA synchronous=NORMAL');c.execute('PRAGMA cache_size=-65536')
        c.execute('CREATE TABLE IF NOT EXISTS meta(key TEXT PRIMARY KEY,value TEXT)')
        old=c.execute("SELECT value FROM meta WHERE key='signature'").fetchone()
        if old and old!=(signature,):raise ValueError('Target store inputs changed; choose a new prediction directory.')
        c.execute('CREATE TABLE IF NOT EXISTS records(entity_id TEXT PRIMARY KEY,payload BLOB NOT NULL) WITHOUT ROWID')
        c.execute('CREATE TABLE IF NOT EXISTS shards(key TEXT PRIMARY KEY,rows INTEGER)')
        if old is None:
            with c:c.executemany('INSERT INTO meta VALUES(?,?)',[('signature',signature),('complete','false')])
        if c.execute("SELECT value FROM meta WHERE key='complete'").fetchone()==('true',):return path
        done=dict(c.execute('SELECT key,rows FROM shards'));total=sum(m['total_rows'] for _,m in sources)
        started=time.perf_counter();new_rows=0
        for s,(parts,manifest) in zip((2,3),sources):
            for part in parts:
                key=f'{s}/{part.name}'
                if key in done:continue
                frame=pl.read_parquet(part,columns=TEXT_COLUMNS)
                assert frame.height==manifest['parts'][part.name]['rows']
                with c:
                    for batch in frame.iter_slices(5000):
                        rows=[(r[0],zlib.compress(json.dumps(r,separators=(',',':'),ensure_ascii=False).encode(),1)) for r in batch.iter_rows()]
                        c.executemany('INSERT INTO records VALUES(?,?)',rows)
                    c.execute('INSERT INTO shards VALUES(?,?)',(key,frame.height))
                done[key]=frame.height;new_rows+=frame.height
                elapsed=time.perf_counter()-started
                print(f'Target feature store: {sum(done.values()):,}/{total:,} | ETA {(total-sum(done.values()))*elapsed/max(new_rows,1)/60:.1f} min',flush=True)
                c.execute('PRAGMA wal_checkpoint(PASSIVE)')
        assert sum(done.values())==total and c.execute('SELECT COUNT(*) FROM records').fetchone()[0]==total
        with c:c.execute("UPDATE meta SET value='true' WHERE key='complete'")
        c.execute('PRAGMA wal_checkpoint(TRUNCATE)')
        return path
    finally:c.close()


class TargetStore:
    def __init__(self,path):
        self.con=sqlite3.connect(f'{Path(path).resolve().as_uri()}?mode=ro',uri=True)
        assert self.con.execute("SELECT value FROM meta WHERE key='complete'").fetchone()==('true',)
        self.signature=self.con.execute("SELECT value FROM meta WHERE key='signature'").fetchone()[0]
        self.con.execute('PRAGMA mmap_size=4294967296');self.con.execute('PRAGMA cache_size=-65536')
    def get(self,ids):
        ids=sorted(set(ids));out={}
        for start in range(0,len(ids),500):
            group=ids[start:start+500];marks=','.join('?' for _ in group)
            for eid,blob in self.con.execute(f'SELECT entity_id,payload FROM records WHERE entity_id IN ({marks})',group):
                out[eid]=dict(zip(TEXT_COLUMNS,json.loads(zlib.decompress(blob))))
        if len(out)!=len(ids):raise ValueError('A candidate ID is missing from the test feature store.')
        return out
    def close(self):self.con.close()


class RetrievalPool:
    def __init__(self,index,aux,config,workers=4):
        self.index,self.aux,self.config=index,aux,config
        self.local=threading.local();self.all=[];self.lock=threading.Lock()
        self.executor=ThreadPoolExecutor(max_workers=workers)
    def _retrieve(self,record):
        if not hasattr(self.local,'retriever'):
            self.local.retriever=FastRetriever(self.index,self.aux,**self.config)
            with self.lock:self.all.append(self.local.retriever)
        return self.local.retriever.retrieve(record)
    def retrieve(self,records):return list(self.executor.map(self._retrieve,records))
    def close(self):
        self.executor.shutdown(wait=True,cancel_futures=True)
        for r in self.all:r.close()


class ModelScorer:
    def __init__(self,model_dir,device='cuda'):
        model_dir=Path(model_dir);self.meta=json.loads((model_dir/'model_metadata.json').read_text())
        assert self.meta['feature_names']==FEATURE_NAMES,'Feature order mismatch'
        src=Path(__file__).parent
        assert file_digest(src/'train_matcher.py')==self.meta['feature_code_sha256'],'Trained feature code changed'
        assert file_digest(src/'candidate_retrieval.py')==self.meta['retrieval_code_sha256'],'Original retrieval code changed'
        self.booster=xgb.Booster();self.booster.load_model(model_dir/'xgboost_model.json')
        self.booster.set_param({'device':device,'nthread':8})
        self.end=int(self.meta['best_iteration'])+1
        assert self.end<=self.booster.num_boosted_rounds()
        self.threshold=float(self.meta['threshold'])
        self.signature={'model':file_digest(model_dir/'xgboost_model.json'),'metadata':file_digest(model_dir/'model_metadata.json'),'device':device}
        self.predict(np.zeros((1,len(FEATURE_NAMES)),dtype=np.float32))
        actual=json.loads(self.booster.save_config())['learner']['generic_param']['device']
        if device.startswith('cuda') and not actual.startswith('cuda'):raise RuntimeError('GPU inference fell back to CPU. Fix CUDA access before continuing.')
    def predict(self,array):
        if not len(array):return np.array([],dtype=np.float32)
        return self.booster.predict(xgb.DMatrix(array),iteration_range=(0,self.end))


def score_batch(frame,pool,store,scorer):
    records=frame.select(TEXT_COLUMNS).to_dicts()
    lists=pool.retrieve(records)
    targets=store.get(p['candidate_entity_id'] for items in lists for p in items)
    vectors=[];counts=[]
    for record,items in zip(records,lists):
        counts.append(len(items))
        if len({p['candidate_entity_id'] for p in items})!=len(items):raise ValueError('Duplicate candidate')
        for pair in items:
            target=targets[pair['candidate_entity_id']]
            row={**record,**pair,**{k+'_b':v for k,v in target.items() if k!='entity_id'}}
            vectors.append(features_for_pair(row))
    array=np.asarray(vectors,dtype=np.float32).reshape(-1,len(FEATURE_NAMES))
    if not np.isfinite(array).all():raise ValueError('Nonfinite feature values')
    probabilities=scorer.predict(array)
    assert len(probabilities)==sum(counts)
    output=[];offset=0
    for record,items,n in zip(records,lists,counts):
        candidates=[p['candidate_entity_id'] for p in items]
        matches=[eid for eid,prob in zip(candidates,probabilities[offset:offset+n]) if prob>=scorer.threshold]
        output.append({'source1_entity_id':record['entity_id'],'candidate_entity_ids':candidates,'matched_entity_ids':matches})
        offset+=n
    return pl.DataFrame(output,schema={'source1_entity_id':pl.String,'candidate_entity_ids':pl.List(pl.String),'matched_entity_ids':pl.List(pl.String)})


def run_predictions(cache_root,index,aux,store_path,scorer,work_dir,batch_rows=250,workers=4,split='test'):
    work_dir=Path(work_dir);parts_dir=work_dir/'parts';parts_dir.mkdir(parents=True,exist_ok=True)
    paths,manifest=cache_parts(cache_root,split,1)
    store=TargetStore(store_path)
    signature=digest({'version':VERSION,'scorer':scorer.signature,'store':store.signature,'references':manifest,
        'aux':json.loads((Path(aux)/'stats.json').read_text()),'batch_rows':batch_rows,
        'ranker_code':file_digest(Path(__file__).with_name('rank_postings.c')),
        'code':file_digest(__file__),'fast_code':file_digest(Path(__file__).with_name('fast_retrieval.py'))})
    mp=work_dir/'manifest.json'
    if mp.exists():
        state=json.loads(mp.read_text())
        if state['signature']!=signature:store.close();raise ValueError('Prediction inputs changed; use a new ER_PREDICT_ROOT.')
    else:state={'signature':signature,'parts':{},'complete':False,'total_rows':manifest['total_rows']};atomic_json(mp,state)
    pool=RetrievalPool(index,aux,scorer.meta['retrieval_config'],workers)
    started=time.perf_counter();new_rows=0;seen=[];processed=0
    try:
        for source_index,path in enumerate(paths):
            frame=pl.read_parquet(path,columns=TEXT_COLUMNS)
            for batch_index,batch in enumerate(frame.iter_slices(batch_rows)):
                key=f'part-{source_index:05d}-{batch_index:05d}.parquet';target=parts_dir/key;seen.append(key)
                if key in state['parts'] and target.exists() and file_digest(target)==state['parts'][key]['sha256']:
                    info=state['parts'][key]
                    if (info['first'],info['last'],info['rows'])!=(batch['entity_id'][0],batch['entity_id'][-1],batch.height):raise ValueError('Reference order changed')
                    processed+=batch.height;continue
                scored=score_batch(batch,pool,store,scorer)
                temp=target.with_suffix('.tmp.parquet');scored.write_parquet(temp);temp.replace(target)
                state['parts'][key]={'rows':batch.height,'first':batch['entity_id'][0],'last':batch['entity_id'][-1],'sha256':file_digest(target),
                    'candidates':scored['candidate_entity_ids'].list.len().sum(),'matches':scored['matched_entity_ids'].list.len().sum()}
                processed+=batch.height;new_rows+=batch.height;atomic_json(mp,state)
                elapsed=time.perf_counter()-started;remaining=state['total_rows']-processed
                print(f'Predicted {processed:,}/{state["total_rows"]:,} references | {new_rows/max(elapsed,.001):.1f} rows/s | ETA {remaining*elapsed/max(new_rows,1)/3600:.2f} h',flush=True)
        assert processed==state['total_rows'] and set(seen)==set(state['parts'])
        state['complete']=True;atomic_json(mp,state)
        return state
    finally:pool.close();store.close()


def export_outputs(work_dir,output_dir):
    work_dir=Path(work_dir);output_dir=Path(output_dir);output_dir.mkdir(parents=True,exist_ok=True)
    state=json.loads((work_dir/'manifest.json').read_text())
    if not state['complete']:raise ValueError('Predictions are incomplete; refusing a partial submission.')
    mtmp=output_dir/'matching_results.tsv.tmp';ctmp=output_dir/'candidate_pairs.tsv.tmp'
    rows=0
    with mtmp.open('w',encoding='utf-8',newline='') as m,ctmp.open('w',encoding='utf-8',newline='') as c:
        m.write('source1_entity_id\tmatched_entity_ids\n');c.write('source1_entity_id\tcandidate_entity_ids\n')
        for key in sorted(state['parts']):
            path=work_dir/'parts'/key
            if file_digest(path)!=state['parts'][key]['sha256']:raise ValueError(f'Prediction part changed: {path}')
            for eid,candidates,matches in pl.read_parquet(path).select('source1_entity_id','candidate_entity_ids','matched_entity_ids').iter_rows():
                if len(set(candidates))!=len(candidates) or len(set(matches))!=len(matches) or not set(matches)<=set(candidates):raise ValueError('Invalid candidate/match lists')
                m.write(eid+'\t'+','.join(matches)+'\n');c.write(eid+'\t'+','.join(candidates)+'\n');rows+=1
    assert rows==state['total_rows']
    mtmp.replace(output_dir/'matching_results.tsv');ctmp.replace(output_dir/'candidate_pairs.tsv')
    return {'rows':rows,'candidate_pairs':sum(p['candidates'] for p in state['parts'].values()),'matches':sum(p['matches'] for p in state['parts'].values())}


def verify_fast_retrieval(cache_root,index,aux,pilot_dir,report_path,config,workers=4):
    """Check every saved pilot candidate, route, rank and score, with no labels."""
    pilot_dir=Path(pilot_dir);report_path=Path(report_path)
    src=Path(__file__).parent
    fingerprint={'fast_code_sha256':file_digest(src/'fast_retrieval.py'),
        'ranker_code_sha256':file_digest(src/'rank_postings.c'),
        'original_retrieval_sha256':file_digest(src/'candidate_retrieval.py'),
        'reference_candidate_sha256':file_digest(pilot_dir/'pilot_candidates.parquet'),
        'retrieval_config':config}
    if report_path.exists():
        previous=json.loads(report_path.read_text())
        if previous.get('fingerprint')==fingerprint and previous.get('status')=='PASS':return previous
    pilot=pl.read_parquet(pilot_dir/'pilot_candidates.parquet')
    ids=pl.read_parquet(pilot_dir/'pilot_recall.parquet',columns=['source1_entity_id'])
    paths,_=cache_parts(cache_root,'train',1)
    anchors=(pl.scan_parquet(paths).select(TEXT_COLUMNS).join(ids.lazy(),left_on='entity_id',right_on='source1_entity_id',how='semi').collect(engine='streaming'))
    assert anchors.height==ids.height
    expected={eid:{} for eid in ids['source1_entity_id']}
    for row in pilot.iter_rows(named=True):expected[row['source1_entity_id']][row['candidate_entity_id']]=row
    pool=RetrievalPool(index,aux,config,workers);started=time.perf_counter();checked=0;error=0.
    try:
        for batch in anchors.iter_slices(250):
            for record,items in zip(batch.iter_rows(named=True),pool.retrieve(batch.to_dicts())):
                old=expected[record['entity_id']];new={p['candidate_entity_id']:p for p in items}
                if set(old)!=set(new):raise ValueError('Optimized retrieval changed candidate IDs; refusing full inference.')
                for eid,a in new.items():
                    b=old[eid];error=max(error,abs(a['best_bm25']-b['best_bm25']))
                    if a['routes']!=b['routes'] or a['best_rank']!=b['best_rank']:raise ValueError('Optimized retrieval changed ranking evidence.')
                checked+=1
            print(f'Exact retrieval parity: {checked:,}/{anchors.height:,}',flush=True)
        if error>1e-10:raise ValueError('Optimized BM25 scores differ from the trained retrieval scores.')
    finally:pool.close()
    report={'status':'PASS','anchors':checked,'candidate_pairs':pilot.height,'max_score_error':error,
            'seconds':time.perf_counter()-started,'workers':workers,'fingerprint':fingerprint}
    atomic_json(report_path,report);return report


def package_submission(package_root,model_dir,validation_report,parity_report,run_manifest,team_name='',team_members=''):
    """Stage verified model artifacts; optionally zip when team metadata is supplied."""
    import re,zipfile
    package_root=Path(package_root);code=package_root/'code/business_entity_resolution'
    if validation_report.get('status')!='PASS':raise ValueError('Output validation has not passed.')
    for filename,key in [('matching_results.tsv','matching_sha256'),('candidate_pairs.tsv','candidate_sha256')]:
        if file_digest(package_root/'output'/filename)!=validation_report[key]:raise ValueError('Validated outputs changed before packaging.')
    if parity_report.get('status')!='PASS':raise ValueError('Retrieval parity has not passed.')
    models=code/'models';models.mkdir(parents=True,exist_ok=True)
    for name in ('xgboost_model.json','model_metadata.json','validation_metrics.json'):
        source=Path(model_dir)/name;target=models/name
        if source.resolve()!=target.resolve():shutil.copy2(source,target)
    atomic_json(models/'retrieval_parity.json',parity_report)
    summary={'validation':validation_report,'retrieval_parity':parity_report,
        'references':run_manifest['total_rows'],'prediction_parts':len(run_manifest['parts']),
        'test_score':'Unknown: no test labels are available.'}
    atomic_json(code/'run_summary.json',summary)
    doc=package_root/'Documentation_template.md'
    text=doc.read_text()
    marker='\n## Completed prediction run (generated by notebook 04)\n'
    text=text.split(marker)[0]
    if team_name:text=re.sub(r'^\*\*Team Name:\*\*.*$',lambda _: '**Team Name:** '+team_name, text,flags=re.M)
    if team_members:text=re.sub(r'^\*\*Team Members:\*\*.*$',lambda _: '**Team Members:** '+team_members,text,flags=re.M)
    text=re.sub(r'^\*\*Status:\*\*.*?(?=\n## )','**Status:** Full test inference and output validation are complete. Test quality is unknown.\n',text,count=1,flags=re.S)
    text+=marker+f'\nFull test inference completed for {validation_report["rows"]:,} reference entities. '
    text+=f'The model scored {validation_report["candidate_pairs"]:,} candidate pairs and emitted {validation_report["matches"]:,} matches. '
    text+='Both TSVs passed bounded-memory validation against raw test IDs, including France, empty lists, uniqueness and candidate/match subset checks. '
    text+='Test/leaderboard quality is unknown. The saved model and reproduction metadata are included under `code/business_entity_resolution/models/`.\n'
    doc.write_text(text)
    if not team_name.strip() or not team_members.strip():
        print('Validated TSVs and model artifacts are staged. Fill TEAM_NAME and TEAM_MEMBERS, then rerun this packaging cell to create the final zip.')
        return None
    from datetime import date
    text=doc.read_text().replace('[To be filled at final submission]',date.today().isoformat())
    if '[To be filled' in text:raise ValueError('Complete the remaining methodology placeholders before packaging.')
    doc.write_text(text)
    safe=re.sub(r'[^A-Za-z0-9_-]+','_',team_name).strip('_') or 'team'
    archive=package_root.parent/f'{safe}_submission.zip';temp=archive.with_suffix('.zip.tmp')
    files=[package_root/'Documentation_template.md',package_root/'output/matching_results.tsv',package_root/'output/candidate_pairs.tsv']
    files.extend(p for p in code.rglob('*') if p.is_file() and '__pycache__' not in p.parts and '.ipynb_checkpoints' not in p.parts and p.suffix not in ('.pyc','.so'))
    with zipfile.ZipFile(temp,'w',compression=zipfile.ZIP_DEFLATED,compresslevel=3,allowZip64=True) as z:
        for path in files:z.write(path,path.relative_to(package_root).as_posix())
    temp.replace(archive)
    return archive
