"""Multilingual explicit-negative contrastive pilot; never changes submission outputs."""
from pathlib import Path
import json, time, random, gc, os
import numpy as np
import polars as pl
from candidate_retrieval import cache_parts
from train_matcher import (TEXT_COLUMNS, FEATURE_NAMES, PAIR_SCHEMA, load_labels, label_pairs,
    feature_cache, tune_threshold, entity_scores, file_digest, atomic_json, digest)
from predict_submission import RetrievalPool
MODEL_ID='intfloat/multilingual-e5-small'
MODEL_REVISION='fd1525a9fd15316a2d503bf26ab031a61d056e98'
KEYS=['source1_entity_id','candidate_entity_id']


def serialize(r):
    fields=[('name',r.get('business_name') or r.get('name_norm') or ''),
        ('address',r.get('business_address') or r.get('address_norm') or ''),
        ('romanized name',r.get('name_match') or ''),('romanized address',r.get('address_match') or '')]
    return 'query: '+' | '.join(k+': '+str(v)[:300] for k,v in fields)


def purge_shared_positives(roles,labels):
    truth=dict(labels.select('source1_entity_id','matched_ids').iter_rows());held=set();keep=[]
    for role in ('audit','tune','train'):
        accepted=[eid for eid in roles.filter(pl.col('role')==role)['entity_id'] if not(set(truth[eid])&held)]
        keep.extend(accepted)
        for eid in accepted:held.update(truth[eid])
    return roles.filter(pl.col('entity_id').is_in(keep))


def prepare(cache,candidates_root,work,train_n=10000,tune_n=1000,audit_n=1000):
    cache,candidates_root,work=map(Path,(cache,candidates_root,work));work.mkdir(parents=True,exist_ok=True)
    signature=digest({'code':file_digest(__file__),'sources':[cache_parts(cache,'train',s)[1] for s in (1,2,3)],
        'labels':json.loads((cache/'train/train_ground_truth/manifest.json').read_text()),
        'counts':[train_n,tune_n,audit_n],'pilot':file_digest(candidates_root/'pilot/pilot_recall.parquet')})
    mp=work/'data_manifest.json'
    if mp.exists():
        state=json.loads(mp.read_text())
        if state['signature']!=signature:raise ValueError('Pilot inputs changed: choose a new NEURAL_ROOT')
        if state.get('complete') and all((work/k).exists() and file_digest(work/k)==v for k,v in state['files'].items()):
            return {k:pl.read_parquet(work/(k+'.parquet')) for k in ('roles','labels','pairs','records','features')}
    atomic_json(mp,{'signature':signature,'complete':False})
    refs=pl.scan_parquet(cache_parts(cache,'train',1)[0]).select(TEXT_COLUMNS)
    old=pl.read_parquet(candidates_root/'pilot/pilot_recall.parquet')['source1_entity_id']
    reserved=refs.filter((pl.col('entity_id').hash(seed=42)%10==0)&~pl.col('entity_id').is_in(old.implode()))
    def sample(frame,n,role):
        return frame.with_columns(pl.col('entity_id').hash(seed=505).alias('_h')).sort('_h').head(n).drop('_h').collect().with_columns(pl.lit(role).alias('role'))
    roles=pl.concat([sample(refs.filter(pl.col('entity_id').hash(seed=42)%10!=0),train_n,'train'),
        sample(reserved.filter(pl.col('entity_id').hash(seed=506)%2==0),tune_n,'tune'),
        sample(reserved.filter(pl.col('entity_id').hash(seed=506)%2==1),audit_n,'audit')])
    labels=load_labels(cache,roles['entity_id'].to_list());roles=purge_shared_positives(roles,labels)
    labels=labels.filter(pl.col('source1_entity_id').is_in(roles['entity_id'].implode()))
    assert set(roles['role'])=={'train','tune','audit'},'A split became empty'
    config={'name_k':20,'address_k':20,'gram_k':10,'joint_k':10,'max_postings':100000}
    pool=RetrievalPool(candidates_root/'train_search.sqlite',candidates_root/'train_fast',config,4);chunks=[]
    try:
        for i,batch in enumerate(roles.iter_slices(250)):
            lists=pool.retrieve(batch.to_dicts())
            rows=[{'source1_entity_id':eid,**pair} for eid,items in zip(batch['entity_id'],lists) for pair in items]
            chunks.append(pl.DataFrame(rows,schema=PAIR_SCHEMA))
            if i%8==0:print(f'Retrieved {min((i+1)*250,roles.height):,}/{roles.height:,} anchors',flush=True)
    finally:pool.close()
    pairs=label_pairs(pl.concat(chunks),labels).join(roles.select(pl.col('entity_id').alias('source1_entity_id'),'role'),on='source1_entity_id').sort(KEYS)
    features=feature_cache(cache,pairs.select(list(PAIR_SCHEMA)),work/'lexical_features')
    targets=set(pairs['candidate_entity_id']);train_ids=set(roles.filter(pl.col('role')=='train')['entity_id'])
    for eid,ids in labels.select('source1_entity_id','matched_ids').iter_rows():
        if eid in train_ids:targets.update(ids)
    ids=pl.DataFrame({'entity_id':sorted(targets|set(roles['entity_id']))})
    parts=sum([cache_parts(cache,'train',s)[0] for s in (1,2,3)],[])
    records=(pl.scan_parquet(parts).select('entity_id','business_name','business_address','name_match','address_match')
        .join(ids.lazy(),on='entity_id',how='semi').collect(engine='streaming'))
    assert records.height==ids.height,'Missing records'
    records=records.select('entity_id').with_columns(pl.Series('text',[serialize(r) for r in records.to_dicts()])).sort('entity_id')
    data={'roles':roles,'labels':labels,'pairs':pairs,'records':records,'features':features}
    for k,v in data.items():v.write_parquet(work/(k+'.parquet'))
    atomic_json(mp,{'signature':signature,'complete':True,'files':{k+'.parquet':file_digest(work/(k+'.parquet')) for k in data}})
    return data


def triplets(data,negatives=3):
    truth=dict(data['labels'].select('source1_entity_id','matched_ids').iter_rows())
    held={t for eid in data['roles'].filter(pl.col('role')!='train')['entity_id'] for t in truth[eid]}
    candidates={}
    for eid,target in data['pairs'].filter((pl.col('role')=='train')&(pl.col('label')==0)).sort(['best_rank','candidate_entity_id']).select(KEYS).iter_rows():
        if target not in held:candidates.setdefault(eid,[]).append(target)
    examples=[]
    for eid in data['roles'].filter(pl.col('role')=='train')['entity_id']:
        neg=candidates.get(eid,[])[:negatives]
        if not neg:continue
        for pos in sorted(truth[eid])[:3]:
            assert not set(neg)&set(truth[eid])
            examples.append((eid,pos,neg))
    if not examples:raise ValueError('No contrastive training examples')
    return examples


def pool_embeddings(output,mask):
    import torch.nn.functional as F
    values=output.last_hidden_state.float()
    return F.normalize((values*mask.unsqueeze(-1)).sum(1)/mask.sum(1,keepdim=True).clamp_min(1),p=2,dim=1)


def contrastive_loss(anchor,targets,valid,temperature=.07):
    import torch
    logits=torch.einsum('bd,bkd->bk',anchor,targets)/temperature
    logits=logits.masked_fill(~valid,float('-inf'))
    return torch.nn.functional.cross_entropy(logits,torch.zeros(len(anchor),device=anchor.device,dtype=torch.long))


def load_encoder(path,device='cuda'):
    import torch
    from transformers import AutoModel,AutoTokenizer
    if device=='cuda' and not torch.cuda.is_available():raise RuntimeError('CUDA unavailable in neural kernel')
    kwargs={'revision':MODEL_REVISION,'cache_dir':os.environ.get('HF_HOME')} if str(path)==MODEL_ID else {'local_files_only':True}
    tokenizer=AutoTokenizer.from_pretrained(str(path),**kwargs)
    model=AutoModel.from_pretrained(str(path),use_safetensors=True,**kwargs).to(device)
    return tokenizer,model


def train_encoder(data,work,epochs=1,batch_size=4,max_length=192,device='cuda'):
    import torch
    work=Path(work);out=work/'encoder';checkpoint=work/'encoder_checkpoint.pt'
    config={'model':MODEL_ID,'revision':MODEL_REVISION,'epochs':epochs,'batch_size':batch_size,
        'max_length':max_length,'lr':2e-5,'temperature':.07,'seed':505,
        'data':file_digest(work/'data_manifest.json'),'code':file_digest(__file__)}
    signature=digest(config)
    if (out/'run.json').exists():
        if json.loads((out/'run.json').read_text())['signature']!=signature:raise ValueError('Encoder config changed; use a new NEURAL_ROOT')
        return out
    random.seed(505);np.random.seed(505);torch.manual_seed(505)
    if device=='cuda':torch.cuda.manual_seed_all(505)
    tokenizer,model=load_encoder(MODEL_ID,device);model.gradient_checkpointing_enable();model.train()
    optimizer=torch.optim.AdamW(model.parameters(),lr=config['lr'])
    scaler=torch.amp.GradScaler('cuda',enabled=device=='cuda')
    examples=triplets(data);texts=dict(data['records'].iter_rows());start_epoch=0;start_batch=0
    if checkpoint.exists():
        state=torch.load(checkpoint,map_location='cpu',weights_only=True)
        if state['signature']!=signature:raise ValueError('Checkpoint configuration changed')
        model.load_state_dict(state['model']);optimizer.load_state_dict(state['optimizer']);scaler.load_state_dict(state['scaler'])
        start_epoch=state['epoch'];start_batch=state['next_batch'];torch.set_rng_state(state['rng'])
        if device=='cuda':torch.cuda.set_rng_state_all(state['cuda_rng'])
    started=time.perf_counter();steps=0;per_epoch=(len(examples)+batch_size-1)//batch_size;total_steps=epochs*per_epoch
    for epoch in range(start_epoch,epochs):
        order=np.random.default_rng(505+epoch).permutation(len(examples))
        for b,offset in enumerate(range(0,len(order),batch_size)):
            if epoch==start_epoch and b<start_batch:continue
            batch=[examples[i] for i in order[offset:offset+batch_size]];width=1+max(len(x[2]) for x in batch)
            ids=[x[0] for x in batch];valid=[]
            for eid,pos,neg in batch:
                items=[pos]+neg;ids.extend(items+[pos]*(width-len(items)));valid.append([True]*len(items)+[False]*(width-len(items)))
            tokens=tokenizer([texts[i] for i in ids],padding=True,truncation=True,max_length=max_length,return_tensors='pt').to(device)
            optimizer.zero_grad(set_to_none=True)
            with torch.autocast(device_type=device,dtype=torch.float16,enabled=device=='cuda'):
                embeddings=pool_embeddings(model(**tokens),tokens['attention_mask'])
                loss=contrastive_loss(embeddings[:len(batch)],embeddings[len(batch):].reshape(len(batch),width,-1),torch.tensor(valid,device=device))
            if not torch.isfinite(loss):raise ValueError('Nonfinite neural loss')
            scaler.scale(loss).backward();scaler.unscale_(optimizer);torch.nn.utils.clip_grad_norm_(model.parameters(),1.)
            scaler.step(optimizer);scaler.update();steps+=1
            if steps%25==0 or steps==1:
                done=epoch*per_epoch+b+1
                print(f'Encoder {done}/{total_steps}, loss {loss.item():.4f}, ETA {(total_steps-done)*(time.perf_counter()-started)/steps/3600:.2f} h',flush=True)
            if steps%100==0 or offset+batch_size>=len(order):
                state={'signature':signature,'model':model.state_dict(),'optimizer':optimizer.state_dict(),'scaler':scaler.state_dict(),
                    'epoch':epoch,'next_batch':b+1,'rng':torch.get_rng_state(),'cuda_rng':torch.cuda.get_rng_state_all() if device=='cuda' else []}
                temp=checkpoint.with_suffix('.tmp');torch.save(state,temp);temp.replace(checkpoint)
    out.mkdir(exist_ok=True);model.save_pretrained(out,safe_serialization=True);tokenizer.save_pretrained(out)
    atomic_json(out/'run.json',{'signature':signature,'config':config,'examples':len(examples),'seconds_this_run':time.perf_counter()-started})
    del model,optimizer;gc.collect()
    if device=='cuda':torch.cuda.empty_cache()
    return out


def encode_records(data,encoder,work,batch_size=64,max_length=192,device='cuda'):
    import torch
    folder=Path(work)/'embeddings';folder.mkdir(exist_ok=True)
    signature=digest({'model':file_digest(Path(encoder)/'model.safetensors'),'records':file_digest(Path(work)/'records.parquet'),
        'max_length':max_length,'batch_size':batch_size,'code':file_digest(__file__)})
    mp=folder/'manifest.json'
    if mp.exists():
        state=json.loads(mp.read_text())
        if state['signature']!=signature:raise ValueError('Embedding configuration changed')
    else:state={'signature':signature,'parts':{},'complete':False}
    tokenizer,model=load_encoder(encoder,device);model.eval();records=data['records'];started=time.perf_counter();new=0
    try:
        for part,batch in enumerate(records.iter_slices(4096)):
            key=f'part-{part:05d}.npy';path=folder/key
            if key in state['parts'] and path.exists() and file_digest(path)==state['parts'][key]:continue
            arrays=[]
            for sub in batch.iter_slices(batch_size):
                tokens=tokenizer(sub['text'].to_list(),padding=True,truncation=True,max_length=max_length,return_tensors='pt').to(device)
                with torch.inference_mode(),torch.autocast(device_type=device,dtype=torch.float16,enabled=device=='cuda'):
                    vectors=pool_embeddings(model(**tokens),tokens['attention_mask']).cpu().numpy()
                arrays.append(vectors)
            with path.with_suffix('.tmp').open('wb') as f:np.save(f,np.concatenate(arrays).astype(np.float32))
            path.with_suffix('.tmp').replace(path);state['parts'][key]=file_digest(path);atomic_json(mp,state)
            new+=batch.height;print(f'Encoded {min((part+1)*4096,records.height):,}/{records.height:,}; {new/(time.perf_counter()-started):.1f} records/s',flush=True)
        state['complete']=True;atomic_json(mp,state)
    finally:
        del model;gc.collect()
        if device=='cuda':torch.cuda.empty_cache()
    return np.concatenate([np.load(folder/k) for k in sorted(state['parts'])])


def compare(data,vectors,work,device='cuda'):
    import xgboost as xgb
    work=Path(work);lookup={eid:i for i,eid in enumerate(data['records']['entity_id'])};pairs=data['pairs']
    # Chunk the pair lookup to bound temporary embedding arrays.
    cos=[]
    for batch in pairs.iter_slices(10000):
        left=[lookup[x] for x in batch['source1_entity_id']];right=[lookup[x] for x in batch['candidate_entity_id']]
        cos.append(np.einsum('ij,ij->i',vectors[left],vectors[right]))
    table=pairs.join(data['features'],on=KEYS,maintain_order='left').with_columns(pl.Series('embedding_cosine',np.concatenate(cos)))
    def labels(role):return data['labels'].filter(pl.col('source1_entity_id').is_in(data['roles'].filter(pl.col('role')==role)['entity_id'].implode()))
    results=[]
    for name,columns in [('lexical_control',FEATURE_NAMES),('lexical_plus_e5',FEATURE_NAMES+['embedding_cosine'])]:
        train=table.filter(pl.col('role')=='train');tune=table.filter(pl.col('role')=='tune');audit=table.filter(pl.col('role')=='audit')
        model=xgb.XGBClassifier(n_estimators=1200,max_depth=6,learning_rate=.05,min_child_weight=5,
            subsample=.9,colsample_bytree=.9,reg_lambda=2.,tree_method='hist',device=device,n_jobs=8,
            random_state=42,early_stopping_rounds=60,eval_metric='logloss')
        weights=train.with_columns((1/pl.len().over('source1_entity_id')).alias('w'))['w'].to_numpy().copy();weights*=len(weights)/weights.sum()
        model.fit(train.select(columns).to_numpy(),train['label'].to_numpy(),sample_weight=weights,
            eval_set=[(tune.select(columns).to_numpy(),tune['label'].to_numpy())],verbose=100)
        actual=json.loads(model.get_booster().save_config())['learner']['generic_param']['device']
        if device=='cuda' and not actual.startswith('cuda'):raise RuntimeError('XGBoost CPU fallback')
        pt=model.predict_proba(tune.select(columns).to_numpy())[:,1];threshold,sweep=tune_threshold(tune,labels('tune'),pt)
        pa=model.predict_proba(audit.select(columns).to_numpy())[:,1];scores=entity_scores(audit,labels('audit'),pa,threshold)
        tp=audit.filter(pl.Series(pa>=threshold))['label'].sum();pred=int((pa>=threshold).sum());true=labels('audit')['match_count'].sum()
        result={'model':name,'threshold':threshold,'audit_macro_f05':scores['f05'].mean(),
            'pair_precision':tp/pred if pred else 0.,'pair_recall':tp/true if true else 0.,'best_iteration':model.best_iteration,
            'audit_anchors':scores.height,'training_anchors':data['roles'].filter(pl.col('role')=='train').height}
        model.save_model(work/(name+'.json'));sweep.write_csv(work/(name+'_thresholds.tsv'),separator='\t');scores.write_parquet(work/(name+'_audit.parquet'))
        atomic_json(work/(name+'_metadata.json'),{'features':columns,**result});results.append(result)
        del model;gc.collect()
    ap=table.filter(pl.col('role')=='audit');oracle=entity_scores(ap,labels('audit'),ap['label'].to_numpy(),.5)['f05'].mean()
    atomic_json(work/'comparison.json',{'results':results,'audit_retrieval_oracle':oracle,
        'note':'Fresh reference holdout; fixed lexical candidates. No dense-retrieval recall improvement is measured. Audit consumed after this comparison.'})
    return pl.DataFrame(results)
