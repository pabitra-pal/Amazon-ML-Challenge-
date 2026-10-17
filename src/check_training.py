"""Check entity metric edge cases, empty-text features, and cached joins."""
from pathlib import Path
import tempfile
import json
import numpy as np
import polars as pl
from train_matcher import (FEATURE_NAMES, TEXT_COLUMNS, features_for_pair, feature_cache,
                           label_pairs, entity_scores, tune_threshold, PAIR_SCHEMA,
                           retrieve_training)
from candidate_retrieval import build_index, COLUMNS


def normalized(eid,name='alpha estate',address='12 lake road'):
    return dict(entity_id=eid,country_key='india',name_norm=name,name_match=name,name_core=name,
        address_norm=address,address_match=address,address_numbers=['12'] if address else [],
        address_digit_tokens=['12'] if address else [],postal_candidates=[],
        name_has_nonlatin=False,address_has_nonlatin=False)


def main():
    labels=pl.DataFrame({'source1_entity_id':['S1-a','S1-b','S1-c'],
        'matched_ids':[['S2-a','S2-missed'],[],['S3-c']], 'match_count':[2,0,1]})
    pairs=pl.DataFrame({'source1_entity_id':['S1-a','S1-a','S1-b'],
                       'candidate_entity_id':['S2-a','S2-wrong','S2-wrong']})
    labeled=label_pairs(pairs,labels)
    assert labeled['label'].to_list()==[1,0,0]
    probabilities=np.array([.9,.7,.6])
    scored=entity_scores(labeled,labels,probabilities,.5)
    assert abs(scored['f05'].mean()-1/6)<1e-12
    scored=entity_scores(labeled,labels,probabilities,.85)
    assert abs(scored['f05'].mean()-(5/6+1)/3)<1e-12
    t,table=tune_threshold(labeled,labels,probabilities)
    assert .7<t<=.9 and table['macro_f05'][0]>(1/3)
    empty=entity_scores(labeled.head(0),labels,np.array([]),.5)
    assert abs(empty['f05'].mean()-1/3)<1e-12
    r=normalized('S1-a',name='',address='')
    other=normalized('S2-a',name='',address='')
    r.update({k+'_b':v for k,v in other.items() if k!='entity_id'})
    r.update(candidate_entity_id='S2-a',routes=2,best_rank=1,best_bm25=-1.)
    f=dict(zip(FEATURE_NAMES,features_for_pair(r)))
    assert len(f)==43 and f['name_exact']==0 and f['address_exact']==0
    assert f['name_ratio']==0 and f['number_conflict']==0 and f['name_missing_a']==1
    assert f['route_address']==1 and f['route_name']==0
    with tempfile.TemporaryDirectory() as temp:
        root=Path(temp)
        for source,rows in [(1,[normalized('S1-a')]),(2,[normalized('S2-a')]),(3,[normalized('S3-c')])]:
            folder=root/'train'/f'train_source{source}';folder.mkdir(parents=True)
            pl.DataFrame(rows).write_parquet(folder/'part.parquet')
            (folder/'manifest.json').write_text(json.dumps({'complete':True,'total_rows':len(rows),'parts':{'part.parquet':{'rows':len(rows)}}}))
        p=pl.DataFrame([{'source1_entity_id':'S1-a','candidate_entity_id':'S2-a',
                        'routes':3,'best_rank':1,'best_bm25':-2.}],schema=PAIR_SCHEMA)
        result=feature_cache(root,p,root/'features')
        assert result['name_exact'][0]==1 and result['address_exact'][0]==1
        assert result.select(FEATURE_NAMES).to_numpy().dtype==np.float32
        again=feature_cache(root,p,root/'features');assert result.equals(again)
        assert 'label' not in result.columns
        build_index(root,'train',root/'index.sqlite',progress=False)
        anchors=pl.DataFrame([normalized('S1-a')]).select(COLUMNS)
        retrieved=retrieve_training(anchors,root/'index.sqlite',{},root/'retrieved',batch_size=1)
        assert set(retrieved['candidate_entity_id'])=={'S2-a','S3-c'}
        stamp=(root/'retrieved/part-00000.parquet').stat().st_mtime_ns
        assert retrieved.equals(retrieve_training(anchors,root/'index.sqlite',{},root/'retrieved',batch_size=1))
        assert stamp==(root/'retrieved/part-00000.parquet').stat().st_mtime_ns
    print('Metric edge cases, missing candidates, label isolation, feature joins and cache checks passed.')

if __name__=='__main__':main()
