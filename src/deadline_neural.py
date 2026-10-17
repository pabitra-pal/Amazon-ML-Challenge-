"""Disk-backed 06 training and complete hybrid inference. No external identity lookup."""
from pathlib import Path
import os,json,time,sqlite3,zlib,gc,shutil,zipfile,random
import numpy as np
import polars as pl
from candidate_retrieval import cache_parts,build_index
from fast_retrieval import prepare_fast_index
from train_matcher import (TEXT_COLUMNS,FEATURE_NAMES,PAIR_SCHEMA,load_labels,label_pairs,
    features_for_pair,file_digest,digest,atomic_json,entity_scores,tune_threshold)
from predict_submission import RetrievalPool,export_outputs
from neural_pilot import serialize,purge_shared_positives,load_encoder,pool_embeddings,contrastive_loss
from validate_streaming import validate_outputs
KEYS=['source1_entity_id','candidate_entity_id']
CONFIG={'name_k':20,'address_k':20,'gram_k':10,'joint_k':10,'max_postings':100000}
FEATURES=FEATURE_NAMES+['embedding_cosine']


def hours_left(deadline):return (deadline-time.time())/3600


def manifest(folder,signature):
    folder=Path(folder);folder.mkdir(parents=True,exist_ok=True);p=folder/'manifest.json'
    if p.exists():
        m=json.loads(p.read_text())
        if m['signature']!=signature:raise ValueError(f'Inputs/configuration changed: use a new work directory ({folder})')
    else:m={'signature':signature,'parts':{},'complete':False};atomic_json(p,m)
    return m


def committed(folder,m,key):
    p=Path(folder)/key
    return key in m['parts'] and p.is_file() and file_digest(p)==m['parts'][key]['sha256']


def put_frame(folder,m,key,frame,**extra):
    folder=Path(folder);p=folder/key;temp=p.with_suffix('.tmp');frame.write_parquet(temp);temp.replace(p)
    m['parts'][key]={'sha256':file_digest(p),'rows':frame.height,**extra};atomic_json(folder/'manifest.json',m)


def build_records(cache,split,path):
    """Original-script text plus existing features; bounded shard/batch memory."""
    path=Path(path);path.parent.mkdir(parents=True,exist_ok=True)
    sources=[cache_parts(cache,split,s) for s in (1,2,3)]
    sig=digest({'sources':[m for _,m in sources],'serializer':file_digest(Path(__file__).with_name('neural_pilot.py')),'version':1})
    c=sqlite3.connect(path);c.execute('PRAGMA journal_mode=WAL');c.execute('PRAGMA synchronous=NORMAL')
    try:
        c.execute('CREATE TABLE IF NOT EXISTS meta(k TEXT PRIMARY KEY,v TEXT)')
        old=c.execute("SELECT v FROM meta WHERE k='signature'").fetchone()
        if old and old!=(sig,):raise ValueError('Record store inputs changed')
        c.execute('CREATE TABLE IF NOT EXISTS records(id TEXT PRIMARY KEY,payload BLOB) WITHOUT ROWID')
        c.execute('CREATE TABLE IF NOT EXISTS shards(k TEXT PRIMARY KEY,n INTEGER)')
        if not old:
            with c:c.executemany('INSERT INTO meta VALUES(?,?)',[('signature',sig),('complete','false')])
        if c.execute("SELECT v FROM meta WHERE k='complete'").fetchone()==('true',):return path
        done=dict(c.execute('SELECT k,n FROM shards'));total=sum(m['total_rows'] for _,m in sources);start=time.time();new=0
        for s,(parts,_) in enumerate(sources,1):
            for part in parts:
                key=f'{s}/{part.name}'
                if key in done:continue
                frame=pl.read_parquet(part,columns=TEXT_COLUMNS+['business_name','business_address'])
                with c:
                    for batch in frame.iter_slices(2000):
                        rows=[]
                        for r in batch.to_dicts():
                            r['text']=serialize(r);r.pop('business_name');r.pop('business_address')
                            rows.append((r['entity_id'],zlib.compress(json.dumps(r,ensure_ascii=False,separators=(',',':')).encode(),1)))
                        c.executemany('INSERT INTO records VALUES(?,?)',rows)
                    c.execute('INSERT INTO shards VALUES(?,?)',(key,frame.height))
                done[key]=frame.height;new+=frame.height
                print(f'{split} record store: {sum(done.values()):,}/{total:,}, ETA {(total-sum(done.values()))*(time.time()-start)/max(new,1)/60:.1f} min',flush=True)
                c.execute('PRAGMA wal_checkpoint(PASSIVE)')
        assert sum(done.values())==total
        with c:c.execute("UPDATE meta SET v='true' WHERE k='complete'")
        c.execute('PRAGMA wal_checkpoint(TRUNCATE)')
        return path
    finally:c.close()


class Records:
    def __init__(self,path):
        self.c=sqlite3.connect(f'{Path(path).resolve().as_uri()}?mode=ro',uri=True)
        assert self.c.execute("SELECT v FROM meta WHERE k='complete'").fetchone()==('true',)
        self.signature=self.c.execute("SELECT v FROM meta WHERE k='signature'").fetchone()[0]
        self.c.execute('PRAGMA cache_size=-65536')
    def get(self,ids):
        ids=sorted(set(ids));out={}
        for i in range(0,len(ids),400):
            group=ids[i:i+400]
            for eid,blob in self.c.execute('SELECT id,payload FROM records WHERE id IN ('+','.join('?'*len(group))+')',group):out[eid]=json.loads(zlib.decompress(blob))
        if len(ids)!=len(out):raise ValueError('Missing record IDs')
        return out
    def close(self):self.c.close()


def choose_splits(cache,old_pilot,old_neural,work,count=250000,holdout=10000):
    work=Path(work);folder=work/'split'
    signature=digest({'count':count,'holdout':holdout,'source':cache_parts(cache,'train',1)[1],
        'labels':json.loads((Path(cache)/'train/train_ground_truth/manifest.json').read_text()),
        'oldpilot':file_digest(Path(old_pilot)/'pilot_recall.parquet'),'oldroles':file_digest(Path(old_neural)/'roles.parquet'),'code':file_digest(__file__)})
    m=manifest(folder,signature)
    if m['complete'] and all(committed(folder,m,k) for k in m['parts']):
        return pl.read_parquet(folder/'roles.parquet'),pl.read_parquet(folder/'labels.parquet')
    oldroles=pl.read_parquet(Path(old_neural)/'roles.parquet')
    used=set(oldroles['entity_id'])|set(pl.read_parquet(Path(old_pilot)/'pilot_recall.parquet')['source1_entity_id'])
    refs=pl.scan_parquet(cache_parts(cache,'train',1)[0]).select('entity_id','country_key')
    reserved=refs.filter((pl.col('entity_id').hash(seed=42)%10==0)&~pl.col('entity_id').is_in(sorted(used)))
    def take(frame,n,role):return frame.with_columns(pl.col('entity_id').hash(seed=606).alias('_sample')).sort('_sample').head(n).drop('_sample').collect().with_columns(pl.lit(role).alias('role'))
    roles=pl.concat([take(refs.filter(pl.col('entity_id').hash(seed=42)%10!=0),count,'train'),
        take(reserved.filter(pl.col('entity_id').hash(seed=607)%2==0),holdout,'tune'),
        take(reserved.filter(pl.col('entity_id').hash(seed=607)%2==1),holdout,'audit')])
    labels=load_labels(cache,roles['entity_id'].to_list())
    oldlabels=pl.read_parquet(Path(old_neural)/'labels.parquet');oldtrain=set(oldroles.filter(pl.col('role')=='train')['entity_id'])
    oldtargets={t for eid,ts in oldlabels.select('source1_entity_id','matched_ids').iter_rows() if eid in oldtrain for t in ts}
    truth=dict(labels.select('source1_entity_id','matched_ids').iter_rows())
    retain=[eid for eid,role in roles.select('entity_id','role').iter_rows() if role=='train' or not set(truth[eid])&oldtargets]
    roles=purge_shared_positives(roles.filter(pl.col('entity_id').is_in(retain)),labels)
    labels=labels.filter(pl.col('source1_entity_id').is_in(roles['entity_id'].implode()))
    assert set(roles['role'])=={'train','tune','audit'}
    put_frame(folder,m,'roles.parquet',roles);put_frame(folder,m,'labels.parquet',labels)
    m['complete']=True;atomic_json(folder/'manifest.json',m)
    return roles,labels


def feature_rows(pairs,records):
    result=[]
    for p in pairs:
        a=records[p['source1_entity_id']];b=records[p['candidate_entity_id']]
        result.append(features_for_pair({**a,**p,**{k+'_b':v for k,v in b.items() if k!='entity_id'}}))
    return np.asarray(result,dtype=np.float32).reshape(-1,len(FEATURE_NAMES))


def prepare_pairs(roles,labels,index,aux,records,work,deadline,negative_cap=8,max_prepare_hours=1.5):
    folder=Path(work)/'prepared'
    sig=digest({'roles':roles.to_dicts(),'labels':labels.to_dicts(),'records':records.signature,'negative_cap':negative_cap,'code':file_digest(__file__)})
    m=manifest(folder,sig)
    if m['complete']:return m
    truth=dict(labels.select('source1_entity_id','matched_ids').iter_rows());held={t for eid in roles.filter(pl.col('role')!='train')['entity_id'] for t in truth[eid]}
    pool=RetrievalPool(index,aux,CONFIG,4);started=time.time();new_refs=0;train_done=0
    try:
        for role in ('tune','audit','train'):
            for i,batch in enumerate(roles.filter(pl.col('role')==role).iter_slices(500)):
                key=f'{role}-{i:05d}.parquet';groupkey=f'groups-{i:05d}.parquet'
                if committed(folder,m,key) and (role!='train' or committed(folder,m,groupkey)):
                    if role=='train':train_done+=m['parts'][key]['anchors']
                    continue
                if role=='train' and ((time.time()-started)/3600>max_prepare_hours or hours_left(deadline)<8):
                    print(f'Preparation cutoff: using {train_done:,} completed training references.',flush=True);break
                source=records.get(batch['entity_id']);lists=pool.retrieve([source[e] for e in batch['entity_id']]);rows=[];groups=[]
                for eid,items in zip(batch['entity_id'],lists):
                    positives=set(truth[eid]);negative=[p for p in sorted(items,key=lambda x:(x['best_rank'],x['candidate_entity_id'])) if p['candidate_entity_id'] not in positives and p['candidate_entity_id'] not in held]
                    selected=items if role!='train' else [p for p in items if p['candidate_entity_id'] in positives]+negative[:negative_cap]
                    for p in selected:rows.append({'source1_entity_id':eid,**p,'label':int(p['candidate_entity_id'] in positives)})
                    if role=='train' and positives and negative:
                        groups.append({'entity_id':eid,'positives':sorted(positives)[:3],'negatives':[p['candidate_entity_id'] for p in negative[:3]]})
                pairframe=pl.DataFrame(rows,schema={**PAIR_SCHEMA,'label':pl.UInt8})
                targets=records.get(pairframe['candidate_entity_id']);arr=feature_rows(rows,{**source,**targets})
                out=pairframe.select(*KEYS,'label').hstack(pl.DataFrame(arr,schema=FEATURE_NAMES,orient='row'))
                put_frame(folder,m,key,out,anchors=batch.height,role=role)
                if role=='train':
                    g=pl.DataFrame(groups,schema={'entity_id':pl.String,'positives':pl.List(pl.String),'negatives':pl.List(pl.String)})
                    put_frame(folder,m,groupkey,g,examples=sum(len(x['positives']) for x in groups),role='groups');train_done+=batch.height
                new_refs+=batch.height
                if i%10==0:print(f'Prepared {role}: {min((i+1)*500,roles.filter(pl.col("role")==role).height):,}; {new_refs/(time.time()-started):.1f} refs/s; {hours_left(deadline):.1f} h remaining',flush=True)
        m['complete']=True;m['training_references']=train_done;atomic_json(folder/'manifest.json',m);return m
    finally:pool.close()


def train_large(records,work,initial_encoder,deadline,batch_size=4,max_length=192,device="cuda",min_references=1000):
    import torch
    work=Path(work);prepared=work/'prepared';prep=json.loads((prepared/'manifest.json').read_text());out=work/'encoder';ck=work/'optimizer.pt'
    config={'initial':file_digest(Path(initial_encoder)/'model.safetensors'),'prepared':file_digest(prepared/'manifest.json'),
        'batch':batch_size,'length':max_length,'device':device,'lr':1e-5,'seed':606,'code':file_digest(__file__)}
    sig=digest(config)
    if (out/'run.json').exists():
        assert json.loads((out/'run.json').read_text())['signature']==sig
        return out
    if prep.get('training_references',0)<min_references or hours_left(deadline)<7.5:return None
    tokenizer,model=load_encoder(initial_encoder,device);model.gradient_checkpointing_enable();model.train()
    torch.manual_seed(606)
    if device=='cuda':torch.cuda.manual_seed_all(606)
    optimizer=torch.optim.AdamW(model.parameters(),lr=config['lr']);scaler=torch.amp.GradScaler('cuda',enabled=device=='cuda')
    start_step=0
    if ck.exists():
        state=torch.load(ck,map_location='cpu',weights_only=True)
        if state['signature']!=sig:raise ValueError('Training checkpoint mismatch')
        model.load_state_dict(state['model']);optimizer.load_state_dict(state['optimizer']);scaler.load_state_dict(state['scaler'])
        start_step=state['step'];torch.set_rng_state(state['rng'])
        if device=='cuda':torch.cuda.set_rng_state_all(state['cuda_rng'])
        del state
    keys=sorted(k for k in prep['parts'] if k.startswith('groups-'));random.Random(606).shuffle(keys)
    steps=0;updates=0;examples_seen=0;start=time.time();cutoff=False
    def checkpoint():
        temp=ck.with_suffix('.tmp');torch.save({'signature':sig,'step':steps,'model':model.state_dict(),'optimizer':optimizer.state_dict(),
            'scaler':scaler.state_dict(),'rng':torch.get_rng_state(),'cuda_rng':torch.cuda.get_rng_state_all() if device=='cuda' else []},temp);temp.replace(ck)
    try:
        for ki,key in enumerate(keys):
            assert committed(prepared,prep,key)
            groups=pl.read_parquet(prepared/key).to_dicts();examples=[(g['entity_id'],pos,g['negatives']) for g in groups for pos in g['positives']]
            random.Random(606+ki).shuffle(examples)
            for offset in range(0,len(examples),batch_size):
                steps+=1
                if steps<=start_step:continue
                if hours_left(deadline)<7:
                    steps-=1;cutoff=True;break
                batch=examples[offset:offset+batch_size];width=1+max(len(x[2]) for x in batch)
                ids=[x[0] for x in batch];valid=[]
                for _,p,neg in batch:
                    items=[p]+neg;ids.extend(items+[p]*(width-len(items)));valid.append([True]*len(items)+[False]*(width-len(items)))
                texts=records.get(ids);tokens=tokenizer([texts[e]['text'] for e in ids],padding=True,truncation=True,max_length=max_length,return_tensors='pt').to(device)
                optimizer.zero_grad(set_to_none=True)
                with torch.autocast(device,dtype=torch.float16,enabled=device=='cuda'):
                    emb=pool_embeddings(model(**tokens),tokens['attention_mask'])
                    loss=contrastive_loss(emb[:len(batch)],emb[len(batch):].reshape(len(batch),width,-1),torch.tensor(valid,device=device))
                if not torch.isfinite(loss):raise ValueError('Nonfinite loss')
                scaler.scale(loss).backward();scaler.unscale_(optimizer);torch.nn.utils.clip_grad_norm_(model.parameters(),1.)
                scaler.step(optimizer);scaler.update();updates+=1;examples_seen+=len(batch)
                if updates%100==0:
                    rate=examples_seen/(time.time()-start)
                    print(f'Encoder step {steps:,}; {rate:.1f} examples/s; loss {loss.item():.4f}; {hours_left(deadline):.2f} h remaining',flush=True)
                if updates%1000==0:checkpoint()
            if cutoff:break
        checkpoint();out.mkdir(exist_ok=True);model.save_pretrained(out,safe_serialization=True);tokenizer.save_pretrained(out)
        atomic_json(out/'run.json',{'signature':sig,'config':config,'steps':steps,'deadline_cutoff':cutoff,'completed_one_pass':not cutoff,'seconds_this_run':time.time()-start})
        return out
    finally:
        del model,optimizer;gc.collect()
        if device=='cuda':torch.cuda.empty_cache()


class Vectors:
    """Persistent float32 embeddings, fetched/encoded in bounded batches."""
    def __init__(self,encoder,path,records,device='cuda',batch_size=64,max_length=192):
        self.encoder=Path(encoder);self.records=records;self.device=device;self.batch=batch_size;self.length=max_length
        self.tokenizer,self.model=load_encoder(encoder,device);self.model.eval()
        path=Path(path);path.parent.mkdir(parents=True,exist_ok=True);self.c=sqlite3.connect(path)
        self.c.execute('PRAGMA journal_mode=WAL');self.c.execute('PRAGMA synchronous=NORMAL');self.c.execute('PRAGMA cache_size=-32768')
        self.signature=digest({'encoder':file_digest(self.encoder/'model.safetensors'),'records':records.signature,'length':max_length,'dtype':'float32'})
        self.c.execute('CREATE TABLE IF NOT EXISTS meta(k TEXT PRIMARY KEY,v TEXT)');old=self.c.execute("SELECT v FROM meta WHERE k='signature'").fetchone()
        if old and old!=(self.signature,):raise ValueError('Embedding cache mismatch')
        self.c.execute('CREATE TABLE IF NOT EXISTS vectors(id TEXT PRIMARY KEY,vector BLOB) WITHOUT ROWID')
        if not old:
            with self.c:self.c.execute('INSERT INTO meta VALUES(?,?)',('signature',self.signature))
        self.encoded=0;self.started=time.time()
    def get(self,ids):
        import torch
        ids=sorted(set(ids));out={}
        for i in range(0,len(ids),400):
            group=ids[i:i+400]
            for eid,blob in self.c.execute('SELECT id,vector FROM vectors WHERE id IN ('+','.join('?'*len(group))+')',group):out[eid]=np.frombuffer(blob,dtype=np.float32)
        missing=[e for e in ids if e not in out]
        for i in range(0,len(missing),self.batch):
            group=missing[i:i+self.batch];rows=self.records.get(group)
            tokens=self.tokenizer([rows[e]['text'] for e in group],padding=True,truncation=True,max_length=self.length,return_tensors='pt').to(self.device)
            with torch.inference_mode(),torch.autocast(self.device,dtype=torch.float16,enabled=self.device=='cuda'):
                array=pool_embeddings(self.model(**tokens),tokens['attention_mask']).cpu().numpy()
            with self.c:self.c.executemany('INSERT INTO vectors VALUES(?,?)',[(eid,v.astype(np.float32).tobytes()) for eid,v in zip(group,array)])
            out.update(zip(group,array));self.encoded+=len(group)
        return out
    def close(self):
        import torch
        self.c.execute('PRAGMA wal_checkpoint(TRUNCATE)');self.c.close();del self.model;gc.collect()
        if self.device=='cuda':torch.cuda.empty_cache()


def add_cosine(frame,vectors):
    v=vectors.get(set(frame['source1_entity_id'])|set(frame['candidate_entity_id']))
    sims=np.array([np.dot(v[a],v[b]) for a,b in frame.select(KEYS).iter_rows()],dtype=np.float32)
    return frame.with_columns(pl.Series('embedding_cosine',sims))


def embed_prepared(work,encoder,records,deadline,roles=('train','tune','audit'),name='features',device='cuda'):
    work=Path(work);prepared=work/'prepared';prep=json.loads((prepared/'manifest.json').read_text());folder=work/name
    sig=digest({'prep':file_digest(prepared/'manifest.json'),'encoder':file_digest(Path(encoder)/'model.safetensors'),'roles':roles})
    m=manifest(folder,sig);vectors=Vectors(encoder,work/(name+'_vectors.sqlite'),records,device=device)
    try:
        keys=[k for k,p in prep['parts'].items() if p.get('role') in roles]
        for i,key in enumerate(keys):
            if committed(folder,m,key):continue
            if hours_left(deadline)<5.8 and 'train' in roles:raise TimeoutError('Reserve final-inference time; use completed 05 model.')
            assert committed(prepared,prep,key)
            frame=add_cosine(pl.read_parquet(prepared/key),vectors)
            put_frame(folder,m,key,frame,role=prep['parts'][key]['role'])
            if i%20==0:print(f'{name}: {i+1}/{len(keys)} chunks; newly encoded {vectors.encoded:,}; {hours_left(deadline):.1f} h remaining',flush=True)
        m['complete']=True;atomic_json(folder/'manifest.json',m)
        return folder
    finally:vectors.close()


def load_role(folder,role):
    folder=Path(folder);m=json.loads((folder/'manifest.json').read_text());assert m['complete']
    keys=[k for k,v in m['parts'].items() if v.get('role')==role]
    for k in keys:assert committed(folder,m,k)
    return pl.read_parquet([folder/k for k in keys])


def booster_predict(path,frame,end,device='cuda'):
    import xgboost as xgb
    b=xgb.Booster();b.load_model(path);b.set_param({'device':device,'nthread':8})
    p=b.predict(xgb.DMatrix(frame.select(FEATURES).to_numpy()),iteration_range=(0,int(end)+1))
    return p


def fit_hybrid(folder,labels,roles,work,deadline,device='cuda'):
    import xgboost as xgb
    work=Path(work);meta=work/'hybrid_metadata.json';modelpath=work/'hybrid.json'
    sig=digest({'features':file_digest(Path(folder)/'manifest.json'),'code':file_digest(__file__)})
    if meta.exists():
        saved=json.loads(meta.read_text());assert saved['signature']==sig
        return modelpath,saved
    if hours_left(deadline)<5.8:raise TimeoutError('Insufficient time for another classifier fit')
    train=load_role(folder,'train');tune=load_role(folder,'tune')
    x=train.select(FEATURES).to_numpy();y=train['label'].to_numpy();weights=train.with_columns((1/pl.len().over('source1_entity_id')).alias('w'))['w'].to_numpy().copy();weights*=len(weights)/weights.sum()
    del train;gc.collect()
    dtrain=xgb.QuantileDMatrix(x,label=y,weight=weights,max_bin=128);del x,y,weights;gc.collect()
    dtune=xgb.QuantileDMatrix(tune.select(FEATURES).to_numpy(),label=tune['label'].to_numpy(),ref=dtrain,max_bin=128)
    class Cutoff(xgb.callback.TrainingCallback):
        def after_iteration(self,model,epoch,evals_log):return hours_left(deadline)<5.3
    b=xgb.train({'objective':'binary:logistic','device':device,'tree_method':'hist','max_depth':6,'eta':.05,
        'min_child_weight':5,'subsample':.9,'colsample_bytree':.9,'lambda':2.,'nthread':8,'max_bin':128,'seed':42,'eval_metric':'logloss'},
        dtrain,num_boost_round=1200,evals=[(dtune,'tune')],early_stopping_rounds=60,verbose_eval=100,callbacks=[Cutoff()])
    if device=='cuda':assert json.loads(b.save_config())['learner']['generic_param']['device'].startswith('cuda')
    end=int(b.best_iteration);p=b.predict(dtune,iteration_range=(0,end+1))
    tl=labels.filter(pl.col('source1_entity_id').is_in(roles.filter(pl.col('role')=='tune')['entity_id'].implode()))
    threshold,sweep=tune_threshold(tune,tl,p);sweep.write_csv(work/'threshold_sweep.tsv',separator='\t')
    b.save_model(modelpath);saved={'signature':sig,'threshold':threshold,'best_iteration':end,'features':FEATURES,'tune_macro_f05':sweep['macro_f05'][0]}
    atomic_json(meta,saved);del b,dtrain,dtune;gc.collect();return modelpath,saved


def select_model(work,old_neural,records,roles,labels,new_encoder,new_model,new_meta,deadline):
    work=Path(work);path=work/'selected.json'
    if path.exists():return json.loads(path.read_text())
    old_neural=Path(old_neural);oldmeta=json.loads((old_neural/'lexical_plus_e5_metadata.json').read_text())
    fallback={'encoder':str((old_neural/'encoder').resolve()),'model':str((old_neural/'lexical_plus_e5.json').resolve()),
        'threshold':oldmeta['threshold'],'best_iteration':oldmeta['best_iteration'],'name':'05_hybrid'}
    # Selection is on tune only; the fresh audit is evaluated only after selection.
    if new_model is not None:
        folder=embed_prepared(work,old_neural/'encoder',records,deadline,roles=('tune',),name='fallback_tune')
        tune=load_role(folder,'tune');p=booster_predict(fallback['model'],tune,fallback['best_iteration'])
        tl=labels.filter(pl.col('source1_entity_id').is_in(roles.filter(pl.col('role')=='tune')['entity_id'].implode()))
        oldscore=entity_scores(tune,tl,p,fallback['threshold'])['f05'].mean()
        selected=({'encoder':str(Path(new_encoder).resolve()),'model':str(Path(new_model).resolve()),'threshold':new_meta['threshold'],
            'best_iteration':new_meta['best_iteration'],'name':'06_hybrid'} if new_meta['tune_macro_f05']>oldscore else fallback)
        selected['selection']={'05_tune':oldscore,'06_tune':new_meta['tune_macro_f05']}
    else:selected=fallback;selected['selection']={'reason':'Deadline fallback to completed 05 model'}
    selected['encoder_sha256']=file_digest(Path(selected['encoder'])/'model.safetensors');selected['model_sha256']=file_digest(selected['model'])
    atomic_json(path,selected);return selected


def audit_selected(work,selected,records,roles,labels,deadline):
    work=Path(work)
    if (work/'audit.json').exists():return json.loads((work/'audit.json').read_text())
    folder=embed_prepared(work,selected['encoder'],records,deadline,roles=('audit',),name='selected_audit')
    frame=load_role(folder,'audit');p=booster_predict(selected['model'],frame,selected['best_iteration'])
    al=labels.filter(pl.col('source1_entity_id').is_in(roles.filter(pl.col('role')=='audit')['entity_id'].implode()))
    scores=entity_scores(frame,al,p,selected['threshold']);scores.write_parquet(work/'audit_entities.parquet')
    tp=int(scores['tp'].sum());pred=int(scores['predicted'].sum());true=int(scores['true_count'].sum())
    report={'model':selected['name'],'audit_macro_f05':scores['f05'].mean(),'anchors':scores.height,
        'precision':tp/pred if pred else 0.,'recall':tp/true if true else 0.,'threshold':selected['threshold'],
        'note':'Selection and threshold use tune only. No France labels; not a leaderboard score.'}
    atomic_json(work/'audit.json',report);return report


def predict_all(cache,index,aux,records,selected,work,batch_rows=250,device="cuda"):
    import xgboost as xgb
    work=Path(work);folder=work/'predictions';partsdir=folder/'parts';partsdir.mkdir(parents=True,exist_ok=True)
    paths,refs=cache_parts(cache,'test',1)
    assert file_digest(selected['model'])==selected['model_sha256']
    assert file_digest(Path(selected['encoder'])/'model.safetensors')==selected['encoder_sha256']
    sig=digest({'references':refs,'records':records.signature,'selected':selected,'batch':batch_rows,'code':file_digest(__file__)})
    m=manifest(folder,sig);m['total_rows']=refs['total_rows'];pool=RetrievalPool(index,aux,CONFIG,4)
    vectors=Vectors(selected['encoder'],work/'test_vectors.sqlite',records,device=device)
    b=xgb.Booster();b.load_model(selected['model']);b.set_param({'device':device,'nthread':8})
    start=time.time();new=0;done=0
    try:
        for pi,path in enumerate(paths):
            frame=pl.read_parquet(path,columns=['entity_id'])
            for bi,batch in enumerate(frame.iter_slices(batch_rows)):
                key=f'part-{pi:05d}-{bi:05d}.parquet'
                if key in m['parts'] and (partsdir/key).exists() and file_digest(partsdir/key)==m['parts'][key]['sha256']:
                    done+=m['parts'][key]['rows'];continue
                sources=records.get(batch['entity_id']);lists=pool.retrieve([sources[e] for e in batch['entity_id']])
                pairs=[{'source1_entity_id':eid,**p} for eid,items in zip(batch['entity_id'],lists) for p in items]
                pf=pl.DataFrame(pairs,schema=PAIR_SCHEMA)
                if pairs:
                    targets=records.get(pf['candidate_entity_id']);arr=feature_rows(pairs,{**sources,**targets})
                    f=pf.select(KEYS).hstack(pl.DataFrame(arr,schema=FEATURE_NAMES,orient='row'));f=add_cosine(f,vectors)
                    probs=b.predict(xgb.DMatrix(f.select(FEATURES).to_numpy()),iteration_range=(0,int(selected['best_iteration'])+1))
                    if device=='cuda' and not json.loads(b.save_config())['learner']['generic_param']['device'].startswith('cuda'):raise RuntimeError('GPU inference fell back to CPU')
                else:probs=np.array([])
                out=[];offset=0
                for eid,items in zip(batch['entity_id'],lists):
                    candidates=[p['candidate_entity_id'] for p in items];n=len(items)
                    matches=[eid2 for eid2,p in zip(candidates,probs[offset:offset+n]) if p>=selected['threshold']];offset+=n
                    out.append({'source1_entity_id':eid,'candidate_entity_ids':candidates,'matched_entity_ids':matches})
                result=pl.DataFrame(out,schema={'source1_entity_id':pl.String,'candidate_entity_ids':pl.List(pl.String),'matched_entity_ids':pl.List(pl.String)})
                temp=(partsdir/key).with_suffix('.tmp');result.write_parquet(temp);temp.replace(partsdir/key)
                m['parts'][key]={'sha256':file_digest(partsdir/key),'rows':batch.height,'candidates':len(pairs),'matches':sum(len(r['matched_entity_ids']) for r in out)}
                atomic_json(folder/'manifest.json',m);done+=batch.height;new+=batch.height
                if bi%10==0:print(f'Hybrid prediction {done:,}/{refs["total_rows"]:,}; ETA {(refs["total_rows"]-done)*(time.time()-start)/max(new,1)/3600:.2f} h',flush=True)
        assert done==refs['total_rows'];m['complete']=True;atomic_json(folder/'manifest.json',m);return folder
    finally:pool.close();vectors.close();del b;gc.collect()


def package(work,package_root,source_code,selected,audit,report,team_name='',team_members=''):
    work=Path(work);root=Path(package_root);code=root/'code/business_entity_resolution'
    for name,key in [('matching_results.tsv','matching_sha256'),('candidate_pairs.tsv','candidate_sha256')]:
        assert report['status']=='PASS' and file_digest(root/'output'/name)==report[key]
    shutil.copytree(source_code,code,dirs_exist_ok=True,ignore=shutil.ignore_patterns('models','__pycache__','.ipynb_checkpoints','*.so'))
    models=code/'models';models.mkdir(exist_ok=True)
    for name in ('experiment.json','pip-freeze.txt','hybrid_metadata.json'):
        if (work/name).exists():shutil.copy2(work/name,code/name)
    shutil.copytree(selected['encoder'],models/'encoder',dirs_exist_ok=True)
    shutil.copy2(selected['model'],models/'hybrid.json')
    portable={**selected,'encoder':'models/encoder','model':'models/hybrid.json','features':FEATURES}
    atomic_json(models/'selected.json',portable);atomic_json(code/'run_summary.json',{'validation':report,'audit':audit,'selected':portable})
    (code/'README.md').write_text('''# Hybrid business entity resolution submission

Recreate the base environment from environment.yml, then run python src/setup_neural.py.
Use the resulting neural Python to run:

    python src/reproduce_neural.py --cache-root /path/to/normalized_v1 --work-root /path/to/new_work --dataset-root /path/to/dataset

First run notebook 01 with the supplied raw dataset to create the normalized cache.
The command rebuilds test lookup tables, scores every test reference with the shipped
encoder and XGBoost model, writes output/ TSVs and validates them against raw test IDs.
Keep work/cache directories outside this package. A CUDA GPU and C99 cc compiler are
required. CPU retrieval/text features and GPU E5/XGBoost scoring are batched. No external
business lookup occurs. All model/tokenizer weights for inference are included.
For training details see notebook 06 and the root methodology document.
''')
    (root/'Documentation_template.md').write_text(f'''# Business Entity Resolution Solution

**Team Name:** {team_name or '[To be filled by team]'}
**Team Members:** {team_members or '[To be filled by team]'}

## 1. Executive Summary
Lexical candidate retrieval plus an E5 multilingual shared encoder and XGBoost pair classifier.
Selected model: {selected['name']}. Outputs passed full raw-ID validation.

## 2. Methodology
Original multilingual text is preserved. Offline ICU normalization/transliteration creates
additional comparison views. E5 text includes original name/address and romanized views,
with query prefix, per-field character caps and 192-token truncation. No external entity
lookup, geocoding or external business data are used. ICU is a preprocessing dependency;
its use is our interpretation of the rules, not an asserted organizer approval.

## 3. Candidate Generation
Same-country name/address/trigram/joint lexical retrieval uses caps 20/20/10/10 and exact
BM25 scoring. No true matches are injected into evaluation or test candidates. The exported
candidate lists are exactly the pairs scored. Multiple matches and empty lists are supported.

## 4. Matching Model
MIT-licensed multilingual-e5-small (118M class of model) and Apache-licensed XGBoost are
below 8B parameters. E5 uses explicit-positive/hard-negative contrastive training, not
implicit in-batch negatives. XGBoost uses 43 lexical features plus embedding cosine similarity.
Larger-run classifier training samples hard negatives; tuning and audit retain all candidates.
The selected threshold is {selected['threshold']}, with {selected['best_iteration']+1} trees.
The shipped encoder and tokenizer define exact inference; no retraining is needed to reproduce.

## 5. Results and Error Analysis
{json.dumps(audit,indent=2)}

The audit does not measure France or leaderboard quality. Tune controls model selection;
the final audit is used for reporting. Inspect run_summary.json for output counts and hashes.

## 6. Reproduction
Run normalization notebook 01, then src/reproduce_neural.py as documented in code README.
Both required TSVs contain every test reference. Source code, environment definitions and
model weights are included. Raw data, work caches and optimizer checkpoints are excluded.
''')
    if not team_name.strip() or not team_members.strip():
        print('Validated submission staged. Fill TEAM_NAME and TEAM_MEMBERS and rerun packaging to create ZIP.');return None
    archive=root.parent/'neural_submission.zip';temp=archive.with_suffix('.tmp')
    with zipfile.ZipFile(temp,'w',zipfile.ZIP_DEFLATED,compresslevel=1,allowZip64=True) as z:
        for p in root.rglob('*'):
            if p.is_file() and '__pycache__' not in p.parts:z.write(p,p.relative_to(root).as_posix())
    temp.replace(archive);return archive


def test_preflight(cache,index,aux,records,selected,work,per_country=100):
    """Separate cached lexical work from unique-record encoding; no accuracy claim."""
    refs=pl.scan_parquet(cache_parts(cache,'test',1)[0]).select('entity_id','country_key')
    countries=refs.group_by('country_key').len().collect();pool=RetrievalPool(index,aux,CONFIG,4)
    vectors=Vectors(selected['encoder'],Path(work)/'test_vectors.sqlite',records);rows=[];encoding_seconds=0.;encoded=0
    try:
        for country,total in countries.iter_rows():
            ids=refs.filter(pl.col('country_key')==country).head(per_country).collect()['entity_id']
            started=time.time();sources=records.get(ids);lists=pool.retrieve([sources[e] for e in ids])
            pairs=[{'source1_entity_id':e,**p} for e,items in zip(ids,lists) for p in items]
            targets=records.get(p['candidate_entity_id'] for p in pairs);feature_rows(pairs,{**sources,**targets})
            lexical=time.time()-started;before=vectors.encoded;started=time.time()
            vectors.get(set(ids)|set(targets));elapsed=time.time()-started;count=vectors.encoded-before
            if count:encoding_seconds+=elapsed;encoded+=count
            rows.append({'country':country,'sample':len(ids),'lexical_seconds':lexical,'estimated_lexical_hours':lexical/len(ids)*total/3600})
        total_records=sum(cache_parts(cache,'test',s)[1]['total_rows'] for s in (1,2,3))
        # A warm resumed cache gives no encoding sample; use the measured 05 rate with conservative overhead.
        rate=encoded/encoding_seconds if encoded else 1500.
        estimate=sum(r['estimated_lexical_hours'] for r in rows)+total_records/rate/3600
        report={'countries':rows,'observed_new_encoding_records_per_second':rate,'projected_scoring_hours':estimate,
            'note':'Separate lexical and unique-record encoding projection, not guaranteed runtime; add validation/package buffer.'}
        atomic_json(Path(work)/'test_preflight.json',report);return report
    finally:pool.close();vectors.close()
