"""Small retrieval, label-isolation, interrupted-index and resume checks."""
from pathlib import Path
import tempfile
import json
import sqlite3
from unittest.mock import patch
import polars as pl
import candidate_retrieval as cr


def write_source(root, source, rows):
    folder = root/'train'/f'train_source{source}'
    folder.mkdir(parents=True)
    parts = {}
    for i, row in enumerate(rows):
        name = f'part-{i:05d}.parquet'
        pl.DataFrame([row], schema=cr.COLUMNS, orient='row').write_parquet(folder/name)
        parts[name] = {'rows': 1}
    (folder/'manifest.json').write_text(json.dumps({'complete':True,'parts':parts,'total_rows':len(rows)}))


def main():
    with tempfile.TemporaryDirectory() as temp:
        root=Path(temp)
        write_source(root,2,[['S2-1','india','esa esa esteta','house 266 84 pratap nagar jaipur','esa esa esteta'],
                             ['S2-2','us','northstar bakery','55 river road boston','northstar bakery']])
        write_source(root,3,[['S3-1','us','ss estate','house 266 84 pratap nagar jaipur','ss estate'],
                             ['S3-2','france','ecole paris','12 rue voltaire paris','ecole paris']])
        idx=root/'index.sqlite'
        original=cr.pl.read_parquet
        calls=0
        def interrupted(*args,**kwargs):
            nonlocal calls
            calls+=1
            if calls==3: raise KeyboardInterrupt('simulated interruption')
            return original(*args,**kwargs)
        try:
            with patch.object(cr.pl,'read_parquet',side_effect=interrupted):
                cr.build_index(root,'train',idx,progress=False)
        except KeyboardInterrupt: pass
        else: raise AssertionError('Interruption was not exercised')
        con=sqlite3.connect(idx)
        assert con.execute('SELECT COUNT(*) FROM docs').fetchone()[0]==2
        con.close()
        try: cr.Retriever(idx)
        except ValueError: pass
        else: raise AssertionError('Incomplete index accepted')
        cr.build_index(root,'train',idx,progress=False)
        cr.build_index(root,'train',idx,progress=False)
        con=sqlite3.connect(idx)
        assert con.execute('SELECT COUNT(*) FROM docs').fetchone()[0]==4
        con.close()
        r=cr.Retriever(idx)
        anchor=dict(zip(cr.COLUMNS,['S1-1','india','ss estate','house 266 84 pratap nagar jaipur','ss estate']))
        matches=r.retrieve(anchor)
        assert {x['candidate_entity_id'] for x in matches}=={'S2-1'}
        assert matches[0]['routes'] & 2, 'Address pass did not rescue the transliterated name'
        typo=dict(zip(cr.COLUMNS,['S1-2','us','nortstar bakerx','','nortstar bakerx']))
        matches=r.retrieve(typo)
        assert 'S2-2' in {x['candidate_entity_id'] for x in matches if x['routes'] & 4}
        singleton=dict(zip(cr.COLUMNS,['S1-3','india','zzzzqqqq','','zzzzqqqq']))
        assert r.retrieve(singleton)==[]
        labels=root/'train/train_ground_truth';labels.mkdir()
        pl.DataFrame({'source1_entity_id':['S1-1','S1-2','S1-3'],
                      'matched_ids':[['S2-1','S2-absent'],['S2-2'],[]]}).write_parquet(labels/'part.parquet')
        (labels/'manifest.json').write_text(json.dumps({'complete':True,'parts':{'part.parquet':{}}}))
        anchors=pl.DataFrame([anchor,typo,singleton])
        stats,details=cr.evaluate_pilot(root,r,anchors,root/'eval')
        assert stats['pair_recall']==2/3
        assert abs(stats['oracle_macro_f05_ceiling']-(1.25/1.5+1+1)/3)<1e-12
        candidates=pl.read_parquet(root/'eval/pilot_candidates.parquet')
        assert 'S2-absent' not in candidates['candidate_entity_id'].to_list(), 'Labels leaked into retrieval'
        r.close()
        m=root/'train/train_source2/manifest.json'
        data=json.loads(m.read_text());data['changed']=True;m.write_text(json.dumps(data))
        try:cr.build_index(root,'train',idx,progress=False)
        except ValueError:pass
        else:raise AssertionError('Changed index inputs accepted')
    print('Candidate retrieval, label isolation, recall calculation and index resume checks passed.')

if __name__=='__main__':main()
