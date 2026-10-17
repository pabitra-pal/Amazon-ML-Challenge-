"""Run with the existing ml-challenge-er Python; creates an isolated overlay/kernel."""
from pathlib import Path
import os,sys,subprocess,venv

def main():
    from importlib.metadata import version
    assert version('polars')=='1.44.2' and version('xgboost')=='3.4.2', 'Activate ml-challenge-er first'
    here=Path(__file__).resolve()
    root=next((p for p in here.parents if (p/'cache/normalized_v1').is_dir()),Path.cwd())
    env=root/'.venv-neural';venv.EnvBuilder(with_pip=True,system_site_packages=True).create(env)
    python=env/'bin/python';cache=root/'cache/pip'
    def run(*args):subprocess.run([str(python),*args],check=True)
    common=['--cache-dir',str(cache),'--timeout','180']
    run('-m','pip','install',*common,'torch==2.6.0','--index-url','https://download.pytorch.org/whl/cu124')
    run('-m','pip','install',*common,'-r',str(here.parent.parent/'requirements-neural.txt'))
    run('-m','ipykernel','install','--user','--name','ml-challenge-neural','--display-name','Python (ml-challenge-neural)')
    print('Refresh Jupyter and select Python (ml-challenge-neural).')

if __name__=='__main__':main()
