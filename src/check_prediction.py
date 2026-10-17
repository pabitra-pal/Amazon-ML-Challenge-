"""Small end-to-end tests for exact retrieval, resumable scoring and TSV validation."""
from pathlib import Path
import tempfile,json,shutil,zipfile
import numpy as np
import polars as pl
from candidate_retrieval import build_index,Retriever
from fast_retrieval import prepare_fast_index,FastRetriever,compile_ranker
from predict_submission import build_target_store,run_predictions,export_outputs,package_submission
from validate_streaming import validate_outputs
from train_matcher import FEATURE_NAMES
from check_training import normalized

class FixtureScorer:
    meta={'retrieval_config':{}}
    threshold=.64
    signature={'fixture':'address-exact'}
    def predict(self,x):return np.where(x[:,FEATURE_NAMES.index('address_exact')]==1,.9,.1)


def main():
    with tempfile.TemporaryDirectory() as tmp:
        root=Path(tmp);cache=root/'normalized';raw=root/'raw';raw.mkdir()
        a=normalized('S1-a','ss estate','12 lake road')
        f=normalized('S1-f','ecole paris','20 rue voltaire');f['country_key']='france'
        z=normalized('S1-z','zzzzqqqq','')
        b=normalized('S2-b','esa esa esteta','12 lake road')
        u=normalized('S2-u','ss estate','12 lake road');u['country_key']='us'
        g=normalized('S3-g','ecole paris','20 rue voltaire');g['country_key']='france'
        groups={1:[a,f,z],2:[b,u],3:[g]}
        for source,records in groups.items():
            folder=cache/'test'/f'test_source{source}';folder.mkdir(parents=True)
            pl.DataFrame(records).write_parquet(folder/'part.parquet')
            (folder/'manifest.json').write_text(json.dumps({'complete':True,'total_rows':len(records),'parts':{'part.parquet':{'rows':len(records)}}}))
            pl.DataFrame({'entity_id':[r['entity_id'] for r in records],'business_name':[r['name_norm'] for r in records],
                          'business_address':[r['address_norm'] for r in records],'country':[r['country_key'] for r in records]}).write_csv(raw/f'test_source{source}.tsv',separator='\t')
        index=root/'index.sqlite';aux=root/'fast'
        build_index(cache,'test',index,progress=False);prepare_fast_index(cache,'test',index,aux);compile_ranker(aux/'native')
        original=Retriever(index);fast=FastRetriever(index,aux);vector=FastRetriever(index,aux,native=False)
        for row in [a,f,z]:
            e=original.retrieve(row);actual=fast.retrieve(row);np_result=vector.retrieve(row)
            assert [x['candidate_entity_id'] for x in e]==[x['candidate_entity_id'] for x in actual]
            assert actual==np_result
            for x,y in zip(e,actual):
                assert x['routes']==y['routes'] and x['best_rank']==y['best_rank']
                assert abs(x['best_bm25']-y['best_bm25'])<1e-12
        original.close();fast.close();vector.close()
        store=build_target_store(cache,'test',root/'targets.sqlite')
        work=root/'predictions';out=root/'output';scorer=FixtureScorer()
        state=run_predictions(cache,index,aux,store,scorer,work,batch_rows=2,workers=2)
        assert state['complete'] and state['total_rows']==3
        first=work/'parts'/sorted(state['parts'])[0];stamp=first.stat().st_mtime_ns
        run_predictions(cache,index,aux,store,scorer,work,batch_rows=2,workers=2)
        assert first.stat().st_mtime_ns==stamp
        # Missing output chunk is safely regenerated.
        first.unlink();run_predictions(cache,index,aux,store,scorer,work,batch_rows=2,workers=2)
        export_outputs(work,out)
        report=validate_outputs(out/'matching_results.tsv',out/'candidate_pairs.tsv',raw,root/'validation')
        assert report['status']=='PASS' and report['rows']==3 and report['matches']==2
        assert (out/'matching_results.tsv').read_text().endswith('S1-z\t\n')
        # Packaging preserves the required archive root and excludes compiled/cache files.
        package=root/'submission';code=package/'code/business_entity_resolution';code.mkdir(parents=True)
        shutil.copytree(out,package/'output')
        (code/'src').mkdir();(code/'src/rank_postings.c').write_text('/* fixture */')
        (code/'src/skip.so').write_bytes(b'fixture')
        (code/'README.md').write_text('fixture');(code/'requirements.txt').write_text('fixture')
        (package/'Documentation_template.md').write_text('**Team Name:** [To be filled by team]\n**Team Members:** [To be filled by team]\n**Submission Date:** [To be filled at final submission]\n\n**Status:** Pending.\n\n## Methodology\nFixture.\n')
        model=root/'model';model.mkdir()
        for name in ('xgboost_model.json','model_metadata.json','validation_metrics.json'):
            (model/name).write_text('{}')
        assert package_submission(package,model,report,{'status':'PASS'},state) is None
        archive=package_submission(package,model,report,{'status':'PASS'},state,'Fixture Team','Member One')
        with zipfile.ZipFile(archive) as z:
            names=set(z.namelist())
            assert {'output/matching_results.tsv','output/candidate_pairs.tsv','Documentation_template.md',
                    'code/business_entity_resolution/models/xgboost_model.json','code/business_entity_resolution/src/rank_postings.c'}<=names
            assert not any(n.endswith('.so') or n.startswith('submission/') for n in names)
            assert '[To be filled' not in z.read('Documentation_template.md').decode()
        # Unknown but well-formed candidate IDs must fail the full ID check.
        text=(out/'candidate_pairs.tsv').read_text();(out/'candidate_pairs.tsv').write_text(text.replace('S1-z\t\n','S1-z\tS2-not-in-test\n'))
        try:validate_outputs(out/'matching_results.tsv',out/'candidate_pairs.tsv',raw,root/'validation')
        except ValueError:pass
        else:raise AssertionError('Unknown candidate ID was accepted')
        state['complete']=False;(work/'manifest.json').write_text(json.dumps(state))
        try:export_outputs(work,out)
        except ValueError:pass
        else:raise AssertionError('Incomplete predictions exported')
    print('Exact ranking, France, empty rows, scoring resume, TSV export and full-ID validation checks passed.')

if __name__=='__main__':main()
