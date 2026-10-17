"""Reproduce a packaged 06 submission from normalized data and shipped weights."""
from pathlib import Path
import argparse,json
from deadline_neural import build_records,Records,predict_all,CONFIG
from candidate_retrieval import build_index
from fast_retrieval import prepare_fast_index
from predict_submission import export_outputs
from validate_streaming import validate_outputs


def main():
    parser=argparse.ArgumentParser();parser.add_argument('--cache-root',type=Path,required=True)
    parser.add_argument('--work-root',type=Path,required=True);parser.add_argument('--dataset-root',type=Path,required=True)
    args=parser.parse_args();code=Path(__file__).resolve().parents[1];root=code.parents[1]
    selected=json.loads((code/'models/selected.json').read_text())
    selected['encoder']=str(code/selected['encoder']);selected['model']=str(code/selected['model'])
    work=args.work_root;work.mkdir(parents=True,exist_ok=True)
    index=work/'test_search.sqlite';aux=work/'test_fast'
    build_index(args.cache_root,'test',index);prepare_fast_index(args.cache_root,'test',index,aux,CONFIG['max_postings'])
    store=build_records(args.cache_root,'test',work/'test_records.sqlite');records=Records(store)
    try:predictions=predict_all(args.cache_root,index,aux,records,selected,work)
    finally:records.close()
    export_outputs(predictions,root/'output')
    print(validate_outputs(root/'output/matching_results.tsv',root/'output/candidate_pairs.tsv',args.dataset_root/'test',work/'validation'))

if __name__=='__main__':main()
