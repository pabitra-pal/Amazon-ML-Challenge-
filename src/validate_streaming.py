"""Bounded-memory validation against raw test IDs, including candidate membership.

Unlike the organizer's in-memory validator, this uses an on-disk ID index.
It additionally expects both outputs in original test Source 1 order, as emitted
by notebook 04. This is stricter than the organizer's arbitrary-order allowance.
"""
from pathlib import Path
from itertools import zip_longest
import csv
import hashlib
import json
import sqlite3
from train_matcher import atomic_json, file_digest


def prepare_id_index(test_dir,work_dir):
    test_dir=Path(test_dir);work_dir=Path(work_dir);work_dir.mkdir(parents=True,exist_ok=True)
    paths=[test_dir/f'test_source{s}.tsv' for s in (1,2,3)]
    signatures=[{'path':str(p.resolve()),'size':p.stat().st_size,'mtime_ns':p.stat().st_mtime_ns} for p in paths]
    signature=hashlib.sha256(json.dumps(signatures,sort_keys=True).encode()).hexdigest()
    path=work_dir/'test_ids.sqlite'
    if path.exists():
        c=sqlite3.connect(path)
        try:
            if c.execute("SELECT value FROM meta WHERE key='signature'").fetchone()!=(signature,):raise ValueError('Raw test inputs changed; use a new validation directory.')
            return path
        finally:c.close()
    temp=work_dir/'test_ids.tmp.sqlite'
    if temp.exists():temp.unlink()
    c=sqlite3.connect(temp)
    try:
        c.execute('PRAGMA synchronous=NORMAL')
        c.execute('CREATE TABLE ids(entity_id TEXT PRIMARY KEY,source INTEGER NOT NULL) WITHOUT ROWID')
        c.execute('CREATE TABLE meta(key TEXT PRIMARY KEY,value TEXT)')
        for source,p in enumerate(paths,start=1):
            count=0;batch=[]
            with p.open(encoding='utf-8',newline='') as f:
                reader=csv.reader(f,delimiter='\t');header=next(reader)
                if header!=['entity_id','business_name','business_address','country']:raise ValueError(f'Bad raw header: {p}')
                for row in reader:
                    if len(row)!=4 or not row[0].startswith(f'S{source}-'):raise ValueError(f'Malformed raw test row in {p}')
                    batch.append((row[0],source));count+=1
                    if len(batch)==10000:
                        with c:c.executemany('INSERT INTO ids VALUES(?,?)',batch)
                        batch=[]
                if batch:
                    with c:c.executemany('INSERT INTO ids VALUES(?,?)',batch)
            print(f'Validation ID index: Source {source}, {count:,} IDs',flush=True)
        final=[{'path':str(p.resolve()),'size':p.stat().st_size,'mtime_ns':p.stat().st_mtime_ns} for p in paths]
        if final!=signatures:raise ValueError('Raw test files changed during validation index build.')
        with c:c.execute('INSERT INTO meta VALUES(?,?)',('signature',signature))
        c.close();c=None;temp.replace(path)
        return path
    finally:
        if c is not None:c.close()


def validate_outputs(matching,candidates,test_dir,work_dir):
    matching=Path(matching);candidates=Path(candidates);work_dir=Path(work_dir)
    work_dir.mkdir(parents=True,exist_ok=True)
    atomic_json(work_dir/'validation_report.json',{'status':'RUNNING'})
    lookup=prepare_id_index(test_dir,work_dir)
    con=sqlite3.connect(f'{lookup.resolve().as_uri()}?mode=ro',uri=True)
    con.execute('PRAGMA mmap_size=4294967296')
    pending=set();rows=pair_count=match_count=0
    def flush():
        ids=list(pending)
        for start in range(0,len(ids),500):
            chunk=ids[start:start+500];marks=','.join('?' for _ in chunk)
            found={r[0] for r in con.execute(f'SELECT entity_id FROM ids WHERE source IN (2,3) AND entity_id IN ({marks})',chunk)}
            if len(found)!=len(chunk):raise ValueError(f'Candidate IDs absent from raw test targets: {sorted(set(chunk)-found)[:5]}')
        pending.clear()
    try:
        with matching.open(encoding='utf-8',newline='') as mf,candidates.open(encoding='utf-8',newline='') as cf,(Path(test_dir)/'test_source1.tsv').open(encoding='utf-8',newline='') as rf:
            mr=csv.reader(mf,delimiter='\t');cr=csv.reader(cf,delimiter='\t');rr=csv.reader(rf,delimiter='\t')
            if next(mr)!=['source1_entity_id','matched_entity_ids']:raise ValueError('Invalid matching header')
            if next(cr)!=['source1_entity_id','candidate_entity_ids']:raise ValueError('Invalid candidate header')
            next(rr)
            for expected,m,c in zip_longest(rr,mr,cr):
                if expected is None or m is None or c is None:raise ValueError('Missing or extra Source 1 output rows')
                if len(m)!=2 or len(c)!=2 or m[0]!=expected[0] or c[0]!=expected[0]:raise ValueError('Duplicate/missing/reordered Source 1 output or malformed TSV row')
                mids=m[1].split(',') if m[1] else [];cids=c[1].split(',') if c[1] else []
                if len(mids)!=len(set(mids)) or len(cids)!=len(set(cids)):raise ValueError('Duplicate IDs within a list')
                if any(not eid.startswith(('S2-','S3-')) for eid in mids+cids):raise ValueError('Invalid target ID prefix')
                if not set(mids)<=set(cids):raise ValueError('A match was never a scored candidate')
                pending.update(cids);rows+=1;pair_count+=len(cids);match_count+=len(mids)
                if rows%1000==0:flush()
                if rows%100000==0:print(f'Validated {rows:,} references',flush=True)
            flush()
        report={'status':'PASS','rows':rows,'candidate_pairs':pair_count,'matches':match_count,
                'matching_sha256':file_digest(matching),'candidate_sha256':file_digest(candidates),
                'checks':'Raw test ID existence, all Source 1 rows exactly once in source order, schemas, prefixes, list uniqueness and match subset.'}
        atomic_json(work_dir/'validation_report.json',report)
        return report
    finally:con.close()
