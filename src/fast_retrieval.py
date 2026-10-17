"""Same lexical candidate rules as 02, with cached postings and vectorized BM25.

No new retrieval terms, pruning or approximate nearest-neighbor search is added.
BM25 uses SQLite FTS5's unit-column-weight formula (k1=1.2, b=.75):
https://www.sqlite.org/fts5.html#the_bm25_function
"""
from pathlib import Path
from collections import OrderedDict
import os
import threading
import functools
import ctypes
import subprocess
import shutil
import hashlib
import json
import math
import sqlite3
import time
import numpy as np
import polars as pl
from candidate_retrieval import Retriever, name_grams, cache_parts

VERSION='2.0.0'


def index_signature(index):
    c=sqlite3.connect(f'{Path(index).resolve().as_uri()}?mode=ro',uri=True)
    try:
        if c.execute("SELECT value FROM meta WHERE key='complete'").fetchone()!=('true',):raise ValueError('Incomplete search index')
        return c.execute("SELECT value FROM meta WHERE key='signature'").fetchone()[0]
    finally:c.close()


def atomic_json(path,value):
    path=Path(path);temp=path.with_suffix(path.suffix+'.tmp')
    temp.write_text(json.dumps(value,indent=2)+'\n');temp.replace(path)


def build_frequency_cache(index,path):
    path=Path(path);path.parent.mkdir(parents=True,exist_ok=True)
    signature=index_signature(index)
    if path.exists():
        c=sqlite3.connect(path)
        try:
            if c.execute("SELECT value FROM meta WHERE key='signature'").fetchone()!=(signature,):raise ValueError('Stale frequency cache')
            return
        finally:c.close()
    temp=path.with_suffix('.tmp.sqlite')
    if temp.exists():temp.unlink()
    src=sqlite3.connect(f'{Path(index).resolve().as_uri()}?mode=ro',uri=True);dst=sqlite3.connect(temp)
    try:
        dst.execute('CREATE TABLE freq(term TEXT,col TEXT,doc INTEGER NOT NULL,PRIMARY KEY(term,col)) WITHOUT ROWID')
        dst.execute('CREATE TABLE meta(key TEXT PRIMARY KEY,value TEXT)')
        cur=src.execute('SELECT term,col,doc FROM vocab')
        with dst:
            while True:
                rows=cur.fetchmany(50000)
                if not rows:break
                dst.executemany('INSERT INTO freq VALUES(?,?,?)',rows)
            dst.executemany('INSERT INTO meta VALUES(?,?)',[('signature',signature),('documents',str(src.execute('SELECT COUNT(*) FROM docs').fetchone()[0]))])
        dst.close();dst=None;temp.replace(path)
    finally:
        src.close()
        if dst is not None:dst.close()


def build_dense_stats(cache_root,split,index,folder):
    folder=Path(folder);folder.mkdir(parents=True,exist_ok=True);mp=folder/'stats.json'
    signature=index_signature(index)
    if mp.exists():
        stats=json.loads(mp.read_text())
        if stats['signature']!=signature:raise ValueError('Stale dense statistics')
        for name,dtype in [('length.npy',np.uint32),('country.npy',np.uint16)]:
            a=np.load(folder/name,mmap_mode='r');assert a.shape==(stats['n']+1,) and a.dtype==dtype
        return
    c=sqlite3.connect(f'{Path(index).resolve().as_uri()}?mode=ro',uri=True)
    try:
        n=c.execute('SELECT COUNT(*) FROM docs').fetchone()[0]
        lengths=np.lib.format.open_memmap(folder/'length.tmp.npy',mode='w+',dtype=np.uint32,shape=(n+1,));lengths[0]=0
        seen=0
        for rid,blob in c.execute('SELECT id,sz FROM search_docsize ORDER BY id'):
            if rid!=seen+1:raise ValueError('Index rowids must be contiguous')
            total=0;v=0
            for byte in blob:
                v=(v<<7)|(byte&127)
                if byte<128:total+=v;v=0
            lengths[rid]=total;seen+=1
        assert seen==n
        average=float(np.sum(lengths,dtype=np.uint64))/n
        lengths.flush();del lengths
        codes=np.lib.format.open_memmap(folder/'country.tmp.npy',mode='w+',dtype=np.uint16,shape=(n+1,));codes[0]=0
        countries={};count=0
        for source in (2,3):
            parts,_=cache_parts(cache_root,split,source)
            for part in parts:
                frame=pl.read_parquet(part,columns=['entity_id','country_key'])
                ends=c.execute('SELECT rowid,entity_id FROM docs WHERE rowid IN (?,?) ORDER BY rowid',(count+1,count+frame.height)).fetchall()
                if ends[0][1]!=frame['entity_id'][0] or ends[-1][1]!=frame['entity_id'][-1]:raise ValueError('Cache/index row order mismatch')
                for country in frame['country_key'].unique():
                    if country not in countries:countries[country]=len(countries)+1
                if len(countries)>65535:raise ValueError('Too many country labels')
                codes[count+1:count+frame.height+1]=np.fromiter((countries[v] for v in frame['country_key']),dtype=np.uint16,count=frame.height)
                count+=frame.height
        assert count==n
        codes.flush();del codes
        (folder/'length.tmp.npy').replace(folder/'length.npy');(folder/'country.tmp.npy').replace(folder/'country.npy')
        atomic_json(mp,{'signature':signature,'n':n,'average':average,'countries':countries})
        print(f'{split}: dense lookup statistics ready ({n:,} records)',flush=True)
    finally:c.close()


def build_postings(index,frequencies,path,max_postings=100000):
    path=Path(path);signature=index_signature(index)+f':{VERSION}:{max_postings}'
    src=sqlite3.connect(f'{Path(index).resolve().as_uri()}?mode=ro',uri=True)
    freq=sqlite3.connect(f'{Path(frequencies).resolve().as_uri()}?mode=ro',uri=True)
    dst=sqlite3.connect(path)
    try:
        dst.execute('PRAGMA journal_mode=WAL');dst.execute('PRAGMA synchronous=NORMAL')
        dst.execute('CREATE TABLE IF NOT EXISTS meta(key TEXT PRIMARY KEY,value TEXT)')
        previous=dst.execute("SELECT value FROM meta WHERE key='signature'").fetchone()
        if previous and previous!=(signature,):raise ValueError('Postings inputs changed; use a new auxiliary directory')
        dst.execute('CREATE TABLE IF NOT EXISTS postings(term TEXT,col TEXT,data BLOB,PRIMARY KEY(term,col)) WITHOUT ROWID')
        if previous is None:
            with dst:dst.executemany('INSERT INTO meta VALUES(?,?)',[('signature',signature),('complete','false'),('last_term',''),('last_col',''),('count','0')])
        if dst.execute("SELECT value FROM meta WHERE key='complete'").fetchone()==('true',):return
        state=dict(dst.execute('SELECT key,value FROM meta'));done=int(state['count']);initial=done
        src.execute('CREATE VIRTUAL TABLE temp.instances USING fts5vocab(main,search,instance)')
        eligible="col IN ('names','address','grams') AND doc<=? AND length(term)>=2"
        total=freq.execute('SELECT COUNT(*) FROM freq WHERE '+eligible,(max_postings,)).fetchone()[0]
        cur=freq.execute('SELECT term,col,doc FROM freq WHERE '+eligible+' AND (term,col)>(?,?) ORDER BY term,col',(max_postings,state['last_term'],state['last_col']))
        started=time.perf_counter()
        while True:
            terms=cur.fetchmany(5000)
            if not terms:break
            blobs=[]
            for term,col,expected in terms:
                rows=src.execute('SELECT doc,count(*) FROM temp.instances WHERE term=? AND col=? GROUP BY doc',(term,col)).fetchall()
                a=np.asarray(rows,dtype=np.uint32).reshape(-1,2)
                if len(a)!=expected:raise ValueError('Posting frequency mismatch')
                blobs.append((term,col,a.tobytes()))
            with dst:
                dst.executemany('INSERT INTO postings VALUES(?,?,?)',blobs)
                done+=len(terms)
                dst.executemany('UPDATE meta SET value=? WHERE key=?',[(terms[-1][0],'last_term'),(terms[-1][1],'last_col'),(str(done),'count')])
            if done%50000==0 or done==total:
                elapsed=time.perf_counter()-started
                print(f'Postings: {done:,}/{total:,} terms | ETA {(total-done)*elapsed/max(done-initial,1)/60:.1f} min',flush=True)
                dst.execute('PRAGMA wal_checkpoint(PASSIVE)')
        assert done==total
        with dst:dst.execute("UPDATE meta SET value='true' WHERE key='complete'")
        dst.execute('PRAGMA wal_checkpoint(TRUNCATE)')
    finally:src.close();freq.close();dst.close()


def prepare_fast_index(cache_root,split,index,folder,max_postings=100000):
    folder=Path(folder);folder.mkdir(parents=True,exist_ok=True)
    build_frequency_cache(index,folder/'frequencies.sqlite')
    build_dense_stats(cache_root,split,index,folder)
    build_postings(index,folder/'frequencies.sqlite',folder/'postings.sqlite',max_postings)
    return folder


def compile_ranker(build_dir):
    build_dir=Path(build_dir);build_dir.mkdir(parents=True,exist_ok=True)
    source=Path(__file__).with_name('rank_postings.c')
    compiler=shutil.which('cc') or shutil.which('gcc')
    if not compiler:raise RuntimeError('A C99 compiler is needed for fast retrieval. Install gcc and rerun setup.')
    version=subprocess.run([compiler,'--version'],check=True,capture_output=True,text=True).stdout.splitlines()[0]
    signature=hashlib.sha256(source.read_bytes()+version.encode()).hexdigest()
    target=build_dir/f'rank_postings_{signature[:16]}.so'
    if not target.exists():
        temp=target.with_name(target.name+f'.{os.getpid()}.{threading.get_ident()}.tmp.so')
        subprocess.run([compiler,'-O3','-std=c99','-fPIC','-shared','-fno-fast-math','-ffp-contract=off',str(source),'-o',str(temp)],check=True)
        temp.replace(target)
    return target


class NativeRanker:
    def __init__(self,path,lengths,countries,stats):
        self.library=ctypes.CDLL(str(Path(path).resolve()))
        self.function=self.library.rank_postings
        self.u32=ctypes.POINTER(ctypes.c_uint32);self.u16=ctypes.POINTER(ctypes.c_uint16);self.f64=ctypes.POINTER(ctypes.c_double)
        self.function.argtypes=[ctypes.POINTER(self.u32),ctypes.POINTER(ctypes.c_uint64),self.f64,ctypes.c_int,ctypes.c_int,
            ctypes.c_double,ctypes.c_uint16,self.u32,self.u16,ctypes.c_uint64,ctypes.c_double,ctypes.c_int,self.u32,self.f64]
        self.function.restype=ctypes.c_int
        self.lengths,self.countries,self.stats=lengths,countries,stats
    def rank(self,posts,idfs,country_idf,country,k,split=0):
        if not posts or k<=0:return []
        pointers=(self.u32*len(posts))(*[p.ctypes.data_as(self.u32) for p in posts])
        sizes=np.asarray([len(p) for p in posts],dtype=np.uint64)
        weights=np.asarray(idfs,dtype=np.float64)
        ids=np.empty(k,dtype=np.uint32);scores=np.empty(k,dtype=np.float64)
        n=self.function(pointers,sizes.ctypes.data_as(ctypes.POINTER(ctypes.c_uint64)),weights.ctypes.data_as(self.f64),
            len(posts),split,country_idf,country,self.lengths.ctypes.data_as(self.u32),self.countries.ctypes.data_as(self.u16),
            len(self.lengths),self.stats['average'],k,ids.ctypes.data_as(self.u32),scores.ctypes.data_as(self.f64))
        if n<0:raise ValueError(f'Invalid native ranking inputs: {n}')
        order=np.lexsort((ids[:n],scores[:n]))
        return [(int(ids[j]),float(scores[j])) for j in order]


class FastRetriever(Retriever):
    """One instance per worker. Connections allow main-thread cleanup only."""
    def __init__(self,index,aux_dir,native=True,**config):
        super().__init__(index,**config)
        self.con.close();self.con=sqlite3.connect(f'{Path(index).resolve().as_uri()}?mode=ro',uri=True,check_same_thread=False)
        aux_dir=Path(aux_dir)
        self.freq_con=sqlite3.connect(f'{(aux_dir/"frequencies.sqlite").resolve().as_uri()}?mode=ro',uri=True,check_same_thread=False)
        self.post_con=sqlite3.connect(f'{(aux_dir/"postings.sqlite").resolve().as_uri()}?mode=ro',uri=True,check_same_thread=False)
        expected=index_signature(index)
        self.stats=json.loads((aux_dir/'stats.json').read_text())
        if self.stats['signature']!=expected:raise ValueError('Stale vector statistics')
        if self.freq_con.execute("SELECT value FROM meta WHERE key='signature'").fetchone()!=(expected,):raise ValueError('Stale frequencies')
        if self.post_con.execute("SELECT value FROM meta WHERE key='signature'").fetchone()!=(expected+f":{VERSION}:{self.config['max_postings']}",):raise ValueError('Stale postings')
        if self.post_con.execute("SELECT value FROM meta WHERE key='complete'").fetchone()!=('true',):raise ValueError('Postings incomplete')
        self.lengths=np.load(aux_dir/'length.npy',mmap_mode='r');self.countries=np.load(aux_dir/'country.npy',mmap_mode='r')
        self.pc=OrderedDict();self.pcbytes=0
        self.native=NativeRanker(compile_ranker(aux_dir/'native'),self.lengths,self.countries,self.stats) if native else None
        for c in (self.con,self.freq_con,self.post_con):c.execute('PRAGMA mmap_size=4294967296');c.execute('PRAGMA cache_size=-32768')

    def _frequency(self,token,field):
        row=self.freq_con.execute('SELECT doc FROM freq WHERE term=? AND col=?',(token,field)).fetchone()
        return row[0] if row else 0

    def idf(self,token,field):
        n=self.frequency(token,field)
        return max(math.log((self.stats['n']-n+.5)/(n+.5)),1.e-6)

    def postings(self,term,field):
        key=(term,field)
        if key in self.pc:self.pc.move_to_end(key);return self.pc[key]
        row=self.post_con.execute('SELECT data FROM postings WHERE term=? AND col=?',key).fetchone()
        if row is None:raise ValueError(f'Missing postings for selected term {key}')
        a=np.frombuffer(row[0],dtype=np.uint32).reshape(-1,2)
        while self.pc and self.pcbytes+a.nbytes>128*1024**2:
            _,old=self.pc.popitem(last=False);self.pcbytes-=old.nbytes
        self.pc[key]=a;self.pcbytes+=a.nbytes
        return a

    def union(self,posts,country):
        if not posts:return np.array([],dtype=np.uint32)
        ids=np.unique(np.concatenate([p[:,0] for p in posts]))
        return ids[self.countries[ids]==self.stats['countries'].get(country,0)]

    def scores(self,ids,terms,posts,country):
        if not len(ids):return np.array([],dtype=np.float64)
        denominator=1.2*(1.0-.75+.75*self.lengths[ids].astype(np.float64)/self.stats['average'])
        scores=self.idf(country,'country')*((1.*(1.2+1.))/(1.+denominator))
        for (term,field),p in zip(terms,posts):
            ix=np.searchsorted(p[:,0],ids);positions=np.flatnonzero(ix<len(p))
            positions=positions[p[ix[positions],0]==ids[positions]]
            freq=p[ix[positions],1].astype(np.float64)
            scores[positions]+=self.idf(term,field)*((freq*(1.2+1.))/(freq+denominator[positions]))
        return -scores

    def retrieve(self,record):
        country=record['country_key']
        if not country.isascii() or not country.isalpha():return super().retrieve(record)
        passes=[]
        for tokens,field,count in [(record['name_match'].split(),'names',4),(record['address_match'].split(),'address',4),(name_grams(record['name_core']),'grams',6)]:
            terms=[(t,field) for t in self.rare_terms(tokens,field,count)]
            posts=[self.postings(*t) for t in terms]
            passes.append((None if self.native else self.union(posts,country),terms,posts))
        ni,nt,np_=passes[0];ai,at,ap=passes[1]
        passes.append((None if self.native else np.intersect1d(ni,ai,assume_unique=True),nt+at,np_+ap))
        found={}
        for bit,(ids,terms,posts),k in zip((1,2,4,8),passes,(self.config['name_k'],self.config['address_k'],self.config['gram_k'],self.config['joint_k'])):
            if k<=0 or not terms or (bit==8 and (not nt or not at)):continue
            if self.native:
                ranked=self.native.rank(posts,[self.idf(*t) for t in terms],self.idf(country,'country'),
                    self.stats['countries'].get(country,0),k,split=len(nt) if bit==8 else 0)
            else:
                if not len(ids):continue
                scores=self.scores(ids,terms,posts,country)
                choices=np.arange(len(ids)) if len(ids)<=k else np.flatnonzero(scores<=np.partition(scores,k-1)[k-1])
                chosen=choices[np.lexsort((ids[choices],scores[choices]))[:k]]
                ranked=[(int(ids[j]),float(scores[j])) for j in chosen]
            for rank,(rid,score) in enumerate(ranked,start=1):
                value=found.setdefault(rid,{'routes':0,'best_rank':rank,'best_bm25':score})
                value['routes']|=bit;value['best_rank']=min(value['best_rank'],rank);value['best_bm25']=min(value['best_bm25'],score)
        if not found:return []
        marks=','.join('?' for _ in found)
        ids=dict(self.con.execute(f'SELECT rowid,entity_id FROM docs WHERE rowid IN ({marks})',list(found)))
        return [{'candidate_entity_id':ids[rid],**v} for rid,v in sorted(found.items(),key=lambda it:(it[1]['best_rank'],it[1]['best_bm25'],ids[it[0]]))]

    def close(self):
        if hasattr(self,'pc'):self.pc.clear()
        if hasattr(self,'freq_con'):self.freq_con.close()
        if hasattr(self,'post_con'):self.post_con.close()
        super().close()
