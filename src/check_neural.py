"""Offline neural pilot tests; tiny randomly initialized model, no downloads."""
from pathlib import Path
import tempfile, json, shutil
import numpy as np
import polars as pl
import neural_pilot as p


def main():
    import torch
    from transformers import BertConfig,BertModel,BertTokenizerFast
    roles=pl.DataFrame({'entity_id':['a','b','c','d'],'role':['train','tune','audit','train']})
    labels=pl.DataFrame({'source1_entity_id':['a','b','c','d'],'matched_ids':[['p'],['q'],['q'],['r']],'match_count':[1,1,1,1]})
    kept=p.purge_shared_positives(roles,labels)
    assert set(kept['entity_id'])=={'a','c','d'}
    assert 'दुकान' in p.serialize({'business_name':'दुकान','name_match':'dukan'})
    data={'roles':kept,'labels':labels,'pairs':pl.DataFrame({'source1_entity_id':['a','a','a','d'],
        'candidate_entity_id':['p','q','n','n'],'role':['train']*4,'label':[1,0,0,0],'best_rank':[1,2,3,1]})}
    groups=p.triplets(data)
    assert all('q' not in neg and pos not in neg for _,pos,neg in groups)
    a=torch.tensor([[1.,0.]],requires_grad=True)
    good=torch.tensor([[[1.,0.],[0.,1.]]]);bad=good.flip(1);mask=torch.tensor([[True,True]])
    assert p.contrastive_loss(a,good,mask)<p.contrastive_loss(a,bad,mask)
    p.contrastive_loss(a,good,mask).backward();assert torch.isfinite(a.grad).all()
    with tempfile.TemporaryDirectory() as tmp:
        root=Path(tmp);base=root/'base';base.mkdir();work=root/'run';work.mkdir()
        (base/'vocab.txt').write_text('[PAD]\n[UNK]\n[CLS]\n[SEP]\n[MASK]\nquery\nshop\nroad\nother\n:')
        tokenizer=BertTokenizerFast(vocab_file=str(base/'vocab.txt'));tokenizer.save_pretrained(base)
        BertModel(BertConfig(vocab_size=10,hidden_size=16,num_hidden_layers=1,num_attention_heads=2,intermediate_size=32)).save_pretrained(base)
        data['records']=pl.DataFrame({'entity_id':['a','d','p','r','n'],'text':['query: shop road','query: shop','query: shop road','query: shop','query: other']})
        data['records'].write_parquet(work/'records.parquet');(work/'data_manifest.json').write_text('{}')
        original=p.MODEL_ID;p.MODEL_ID=str(base)
        try:
            out=p.train_encoder(data,work,batch_size=2,max_length=16,device='cpu')
            vectors=p.encode_records(data,out,work,batch_size=2,max_length=16,device='cpu')
            assert vectors.shape==(5,16) and np.allclose(np.linalg.norm(vectors,axis=1),1)
            again=p.encode_records(data,out,work,batch_size=2,max_length=16,device='cpu');assert np.array_equal(vectors,again)
            # Recover from optimizer checkpoint, with final directory absent.
            shutil.rmtree(out)
            out=p.train_encoder(data,work,batch_size=2,max_length=16,device='cpu')
            assert (out/'run.json').exists()
        finally:p.MODEL_ID=original
    print('PASS: script preservation, split purge, false-negative exclusion, differentiable loss, tiny-model training, checkpoint recovery and encoding resume.')

if __name__=='__main__':main()
