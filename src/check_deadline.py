"""Offline synthetic end-to-end 06 test with a tiny encoder; no quality claims."""
from pathlib import Path
import tempfile,json,time,zipfile,shutil,os
import numpy as np
import polars as pl
from transformers import BertConfig,BertModel,BertTokenizerFast
from check_training import normalized
from deadline_neural import *


def main():
    device=os.environ.get("CHECK_DEVICE","cpu")
    with tempfile.TemporaryDirectory() as temp:
        root=Path(temp);cache=root/'cache';raw=root/'dataset/test';raw.mkdir(parents=True);work=root/'work';work.mkdir()
        refs=[];targets=[]
        for i in range(30):
            refs.append(normalized(f'S1-{i}',f'alpha shop {i}',f'{i} lake road'))
            targets.append(normalized(f'S2-{i}',f'alpha shop {i}',f'{i} lake road'))
        refs.append(normalized('S1-empty','zzzqqq',''))
        groups={1:refs,2:targets,3:[normalized('S3-x','remote firm','900 hill road')]}
        for split in ('train','test'):
            for source,rows in groups.items():
                rows=[{**r,'business_name':r['name_norm'],'business_address':r['address_norm']} for r in rows]
                folder=cache/split/f'{split}_source{source}';folder.mkdir(parents=True)
                pl.DataFrame(rows).write_parquet(folder/'part.parquet')
                (folder/'manifest.json').write_text(json.dumps({'complete':True,'total_rows':len(rows),'parts':{'part.parquet':{'rows':len(rows)}}}))
                if split=='test':
                    pl.DataFrame(rows).select('entity_id','business_name','business_address',pl.col('country_key').alias('country')).write_csv(raw/f'test_source{source}.tsv',separator='\t')
        index=root/'index.sqlite';aux=root/'fast';build_index(cache,'train',index,progress=False);prepare_fast_index(cache,'train',index,aux)
        store=build_records(cache,'train',root/'records.sqlite');records=Records(store)
        roles=pl.DataFrame({'entity_id':[r['entity_id'] for r in refs],'role':['train']*20+['tune']*5+['audit']*6})
        labels=pl.DataFrame({'source1_entity_id':[r['entity_id'] for r in refs],
            'matched_ids':[[f'S2-{i}'] for i in range(30)]+[[]],'match_count':[1]*30+[0]})
        deadline=time.time()+20*3600
        prep=prepare_pairs(roles,labels,index,aux,records,work,deadline)
        assert prep['training_references']==20
        # Each held-out reference keeps all label-blind candidates, including the singleton.
        assert sum(v['anchors'] for v in prep['parts'].values() if v.get('role')=='audit')==6
        base=root/'tiny';base.mkdir();(base/'vocab.txt').write_text('[PAD]\n[UNK]\n[CLS]\n[SEP]\n[MASK]\nquery\nshop\nroad\nother\n:')
        tok=BertTokenizerFast(vocab_file=str(base/'vocab.txt'));tok.save_pretrained(base)
        BertModel(BertConfig(vocab_size=10,hidden_size=16,num_hidden_layers=1,num_attention_heads=2,intermediate_size=32)).save_pretrained(base)
        encoder=train_large(records,work,base,deadline,device=device,min_references=1,max_length=32)
        assert encoder.exists()
        first_digest=file_digest(encoder/'model.safetensors')
        shutil.rmtree(encoder)
        encoder=train_large(records,work,base,deadline,device=device,min_references=1,max_length=32)
        assert file_digest(encoder/'model.safetensors')==first_digest
        encoded=embed_prepared(work,encoder,records,deadline,device=device)
        model,meta=fit_hybrid(encoded,labels,roles,work,deadline,device=device)
        chosen={'encoder':str(encoder),'model':str(model),'encoder_sha256':file_digest(encoder/'model.safetensors'),
            'model_sha256':file_digest(model),'threshold':meta['threshold'],'best_iteration':meta['best_iteration'],'name':'fixture'}
        testindex=root/'test_index.sqlite';testaux=root/'test_fast';build_index(cache,'test',testindex,progress=False);prepare_fast_index(cache,'test',testindex,testaux)
        teststore=build_records(cache,'test',root/'test_records.sqlite');tr=Records(teststore)
        try:
            outputwork=root/'predict';pred=predict_all(cache,testindex,testaux,tr,chosen,outputwork,batch_rows=8,device=device)
            first=pred/'parts'/sorted(json.loads((pred/'manifest.json').read_text())['parts'])[0];stamp=first.stat().st_mtime_ns
            predict_all(cache,testindex,testaux,tr,chosen,outputwork,batch_rows=8,device=device);assert first.stat().st_mtime_ns==stamp
        finally:tr.close();records.close()
        package_root=root/'submission';export_outputs(pred,package_root/'output')
        report=validate_outputs(package_root/'output/matching_results.tsv',package_root/'output/candidate_pairs.tsv',raw,root/'validation')
        assert report['rows']==31 and report['status']=='PASS'
        source=root/'source';(source/'src').mkdir(parents=True);(source/'src/reproduce_neural.py').write_text('# fixture')
        archive=package(work,package_root,source,chosen,{'fixture':True},report,'Test','Test Member')
        with zipfile.ZipFile(archive) as z:
            names=set(z.namelist());assert 'output/matching_results.tsv' in names and 'code/business_entity_resolution/models/encoder/model.safetensors' in names
            assert not any('optimizer.pt' in n or '.sqlite' in n for n in names)
    print('PASS: bounded record stores, sampling, training, feature encoding, XGBoost, prediction resume, full-ID validation and package layout.')

if __name__=='__main__':main()
