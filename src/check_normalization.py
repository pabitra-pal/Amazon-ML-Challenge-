import nbformat, tempfile, pathlib, json, time
p=pathlib.Path(__file__).resolve().with_name('01_normalize_and_cache.ipynb')
nb=nbformat.read(p,as_version=4)
ns={}
for i,c in enumerate(nb.cells):
 if c.cell_type=='code' and not set(c.metadata.get('tags',[])) & {'pilot','full_run','verify_full','read_full'}:
  print('Executing cell',i,flush=True)
  exec(compile(c.source,f'notebook-cell-{i}','exec'),ns)
# End-to-end writer and recovery checks on small temporary source/label fixtures.
pl=ns['pl']
with tempfile.TemporaryDirectory() as tmp:
 root=pathlib.Path(tmp)
 train=root/'train';train.mkdir()
 ns['CACHE_ROOT']=root/'cache'; ns['CACHE_ROOT'].mkdir()
 ns['BATCH_ROWS']=2
 source=train/'train_source2.tsv'
 df=ns['fixture'].with_columns(pl.Series('entity_id',['S2-01','S2-02','S2-03','S2-04']))
 df.write_csv(source,separator='\t')
 first=ns['cache_file'](source)
 folder=ns['CACHE_ROOT']/'train/train_source2'
 parts=sorted(folder.glob('*.parquet'))
 cached=pl.read_parquet(parts)
 assert cached.select(ns['RAW_COLUMNS']).equals(df)
 stamps={part.name:part.stat().st_mtime_ns for part in parts}
 second=ns['cache_file'](source)
 assert stamps=={part.name:part.stat().st_mtime_ns for part in parts},'Resume rewrote complete files'
 # Simulate interrupted run and one missing committed part.
 manifest=json.loads((folder/'manifest.json').read_text());manifest['complete']=False
 (folder/'manifest.json').write_text(json.dumps(manifest));parts[-1].unlink()
 ns['cache_file'](source)
 assert pl.read_parquet(sorted(folder.glob('*.parquet'))).select(ns['RAW_COLUMNS']).equals(df)
 for part in parts[:-1]: assert stamps[part.name]==part.stat().st_mtime_ns
 gt=train/'train_ground_truth.tsv'
 pl.DataFrame({'source1_entity_id':['S1-1','S1-2'],'matched_entity_ids':['S2-01,S3-02',None]}).write_csv(gt,separator='\t')
 ns['cache_file'](gt,kind='ground_truth')
 labels=pl.read_parquet(sorted((ns['CACHE_ROOT']/'train/train_ground_truth').glob('*.parquet')))
 assert labels['match_count'].to_list()==[2,0]
 assert labels['matched_ids'][1].to_list()==[]
 # A changed configuration must reject an old cache.
 ns['BATCH_ROWS']=3
 try: ns['cache_file'](source)
 except AssertionError: pass
 else: raise AssertionError('Changed configuration was accepted')
print('ALL NOTEBOOK SMOKE AND RESUME CHECKS PASSED',flush=True)
