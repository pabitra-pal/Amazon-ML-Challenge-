"""Optional runtime cache optimization; no changes to model or prediction signature."""
import time
import numpy as np
import deadline_neural as d

_BaseVectors = d.Vectors
class BatchedVectors(_BaseVectors):
    def get(self, ids):
        import torch
        started=time.perf_counter()
        ids=sorted(set(ids)); out={}
        for i in range(0,len(ids),400):
            group=ids[i:i+400]
            for eid,blob in self.c.execute('SELECT id,vector FROM vectors WHERE id IN ('+','.join('?'*len(group))+')',group):
                out[eid]=np.frombuffer(blob,dtype=np.float32)
        read_seconds=time.perf_counter()-started
        missing=[e for e in ids if e not in out]
        encoding=0.;writing=0.
        # One transaction per get() instead of one per 64-record GPU batch.
        # Float32 storage, ordering, batch size, and model operations are unchanged.
        with self.c:
            for i in range(0,len(missing),self.batch):
                group=missing[i:i+self.batch];t=time.perf_counter();rows=self.records.get(group)
                tokens=self.tokenizer([rows[e]['text'] for e in group],padding=True,truncation=True,max_length=self.length,return_tensors='pt').to(self.device)
                with torch.inference_mode(),torch.autocast(self.device,dtype=torch.float16,enabled=self.device=='cuda'):
                    array=d.pool_embeddings(self.model(**tokens),tokens['attention_mask']).cpu().numpy()
                encoding+=time.perf_counter()-t;t=time.perf_counter()
                self.c.executemany('INSERT INTO vectors VALUES(?,?)',[(eid,v.astype(np.float32).tobytes()) for eid,v in zip(group,array)])
                writing+=time.perf_counter()-t
                out.update(zip(group,array));self.encoded+=len(group)
            t=time.perf_counter()
        writing+=time.perf_counter()-t
        self.calls=getattr(self,'calls',0)+1
        if self.calls%10==1:
            print(f'Embedding timing: {len(ids):,} IDs, {len(missing):,} new; read {read_seconds:.2f}s, encode/load {encoding:.2f}s, write/commit {writing:.2f}s',flush=True)
        return out

def install():
    d.Vectors=BatchedVectors
    print('Batched embedding-cache commits enabled. Existing prediction chunks and vectors will be reused.')


def predict_all_timed(cache,index,aux,records,selected,work,batch_rows=250,device="cuda"):
    from deadline_neural import (Path, cache_parts, file_digest, digest, manifest, RetrievalPool, CONFIG, pl, PAIR_SCHEMA, feature_rows, KEYS, FEATURE_NAMES, add_cosine, FEATURES, json, atomic_json, gc)
    import xgboost as xgb
    work=Path(work);folder=work/'predictions';partsdir=folder/'parts';partsdir.mkdir(parents=True,exist_ok=True)
    paths,refs=cache_parts(cache,'test',1)
    assert file_digest(selected['model'])==selected['model_sha256']
    assert file_digest(Path(selected['encoder'])/'model.safetensors')==selected['encoder_sha256']
    sig=digest({'references':refs,'records':records.signature,'selected':selected,'batch':batch_rows,'code':file_digest(d.__file__)})
    m=manifest(folder,sig);m['total_rows']=refs['total_rows'];pool=RetrievalPool(index,aux,CONFIG,4)
    vectors=BatchedVectors(selected['encoder'],work/'test_vectors.sqlite',records,device=device)
    b=xgb.Booster();b.load_model(selected['model']);b.set_param({'device':device,'nthread':8})
    start=time.time();new=0;done=0
    try:
        for pi,path in enumerate(paths):
            frame=pl.read_parquet(path,columns=['entity_id'])
            for bi,batch in enumerate(frame.iter_slices(batch_rows)):
                key=f'part-{pi:05d}-{bi:05d}.parquet'
                if key in m['parts'] and (partsdir/key).exists() and file_digest(partsdir/key)==m['parts'][key]['sha256']:
                    done+=m['parts'][key]['rows'];continue
                timings={};batch_start=time.perf_counter();t=batch_start
                sources=records.get(batch['entity_id']);timings['source_read']=time.perf_counter()-t;t=time.perf_counter()
                lists=pool.retrieve([sources[e] for e in batch['entity_id']]);timings['retrieval']=time.perf_counter()-t;t=time.perf_counter()
                pairs=[{'source1_entity_id':eid,**p} for eid,items in zip(batch['entity_id'],lists) for p in items]
                pf=pl.DataFrame(pairs,schema=PAIR_SCHEMA)
                timings['pair_frame']=time.perf_counter()-t;t=time.perf_counter()
                if pairs:
                    targets=records.get(pf['candidate_entity_id']);timings['target_read']=time.perf_counter()-t;t=time.perf_counter()
                    arr=feature_rows(pairs,{**sources,**targets});timings['string_features']=time.perf_counter()-t;t=time.perf_counter()
                    f=pf.select(KEYS).hstack(pl.DataFrame(arr,schema=FEATURE_NAMES,orient='row'));timings['feature_frame']=time.perf_counter()-t;t=time.perf_counter()
                    f=add_cosine(f,vectors);timings['embeddings_cosine']=time.perf_counter()-t;t=time.perf_counter()
                    matrix=xgb.DMatrix(f.select(FEATURES).to_numpy());timings['dmatrix']=time.perf_counter()-t;t=time.perf_counter()
                    probs=b.predict(matrix,iteration_range=(0,int(selected['best_iteration'])+1));timings['xgboost']=time.perf_counter()-t;del matrix
                    if device=='cuda' and not json.loads(b.save_config())['learner']['generic_param']['device'].startswith('cuda'):raise RuntimeError('GPU inference fell back to CPU')
                else:probs=np.array([])
                t=time.perf_counter()
                out=[];offset=0
                for eid,items in zip(batch['entity_id'],lists):
                    candidates=[p['candidate_entity_id'] for p in items];n=len(items)
                    matches=[eid2 for eid2,p in zip(candidates,probs[offset:offset+n]) if p>=selected['threshold']];offset+=n
                    out.append({'source1_entity_id':eid,'candidate_entity_ids':candidates,'matched_entity_ids':matches})
                result=pl.DataFrame(out,schema={'source1_entity_id':pl.String,'candidate_entity_ids':pl.List(pl.String),'matched_entity_ids':pl.List(pl.String)})
                timings['output_build']=time.perf_counter()-t;t=time.perf_counter()
                temp=(partsdir/key).with_suffix('.tmp');result.write_parquet(temp);temp.replace(partsdir/key)
                m['parts'][key]={'sha256':file_digest(partsdir/key),'rows':batch.height,'candidates':len(pairs),'matches':sum(len(r['matched_entity_ids']) for r in out)}
                atomic_json(folder/'manifest.json',m);done+=batch.height;new+=batch.height
                timings['save']=time.perf_counter()-t;timings['total']=time.perf_counter()-batch_start
                entry={'part':key,'rows':batch.height,'timings_seconds':timings}
                with (work/'prediction_timings.jsonl').open('a') as log:log.write(json.dumps(entry)+'\n')
                if new<=batch_rows*3 or bi%10==0:print('Stage timing: '+', '.join(f'{k}={v:.2f}s' for k,v in timings.items()),flush=True)
                if bi%10==0:print(f'Hybrid prediction {done:,}/{refs["total_rows"]:,}; ETA {(refs["total_rows"]-done)*(time.time()-start)/max(new,1)/3600:.2f} h',flush=True)
        assert done==refs['total_rows'];m['complete']=True;atomic_json(folder/'manifest.json',m);return folder
    finally:pool.close();vectors.close();del b;gc.collect()


def precompute_test_embeddings(work, selected, max_hours=2.0, chunk_rows=4096,
                               reserve_hours=5.0, min_free_gib=6.0,
                               deadline=1790533740.0, max_chunks=None):
    """Single-writer, resumable ordered pass. Stop prediction before calling.

    Checkpoints advance only after embeddings commit. Interruption may redo one
    chunk's lookups but never discards completed prediction parts or vectors.
    """
    import json, shutil, zlib
    from pathlib import Path
    work=Path(work)
    if max_hours<=0 or chunk_rows<=0:raise ValueError('Positive time and chunk size required')
    records=d.Records(work/'test_records.sqlite')
    vectors=None
    state_path=work/'embedding_precompute.json'
    start=time.monotonic();chunks=0;reason='complete'
    try:
        assert d.file_digest(Path(selected['encoder'])/'model.safetensors')==selected['encoder_sha256']
        vectors=BatchedVectors(selected['encoder'],work/'test_vectors.sqlite',records)
        state=json.loads(state_path.read_text()) if state_path.exists() else {
            'signature':vectors.signature,'last_id':'','visited':0,'complete':False}
        if state['signature']!=vectors.signature:raise ValueError('Precompute model/records changed')
        if state['complete']:
            print('Test embedding precomputation already complete.');return state
        class ChunkRecords:
            def get(self, ids):return {eid:rows[eid] for eid in ids}
        vectors.records=ChunkRecords()
        while True:
            if time.monotonic()-start>=max_hours*3600:reason='time budget';break
            if deadline-time.time()<=reserve_hours*3600:reason='deadline reserve';break
            if shutil.disk_usage(work).free<min_free_gib*1024**3:reason='disk reserve';break
            if max_chunks is not None and chunks>=max_chunks:reason='chunk limit';break
            batch=records.c.execute('SELECT id,payload FROM records WHERE id>? ORDER BY id LIMIT ?',
                                    (state['last_id'],chunk_rows)).fetchall()
            if not batch:state['complete']=True;break
            t=time.monotonic()
            rows={eid:json.loads(zlib.decompress(blob)) for eid,blob in batch}
            before=vectors.encoded
            vectors.get(rows)
            state.update(last_id=batch[-1][0],visited=state['visited']+len(batch))
            d.atomic_json(state_path,state);chunks+=1
            if chunks==1 or chunks%10==0:
                elapsed=time.monotonic()-start
                print(f"Precompute: {state['visited']:,} records visited; {vectors.encoded:,} newly encoded this run; "
                      f"last chunk {time.monotonic()-t:.1f}s ({vectors.encoded-before:,} new); "
                      f"budget left {max(0,max_hours-elapsed/3600):.2f}h",flush=True)
        state['last_stop_reason']=reason
        d.atomic_json(state_path,state)
        print(f"Precompute stopped: {reason}; complete={state['complete']}. Resume prediction next.",flush=True)
        return state
    finally:
        if vectors is not None:vectors.close()
        records.close()
