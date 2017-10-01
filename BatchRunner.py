#!/usr/bin/env python
"""
batch processing of entire directories of images (one directory per night typically)
does NOT automatically delete original data

Steps
0) create list of directories to process
1) create index of randomly ordered filenames, opening each one to determine its relative elapsed time, creating index.h5
"""
import sys
from pathlib import Path
import subprocess
# %%
from argparse import ArgumentParser
p = ArgumentParser()
p.add_argument('indir',help='directory containing directories to process (top level directory)')
p.add_argument('outdir',help='directory to place index, extracted frames and AVI preview')
p.add_argument('-codepath',help='top level directory where Git repos are stored',default='~/code')
p = p.parse_args()
# %%
codedir = Path(p.codepath).expanduser()
indir = Path(p.indir).expanduser()
outdir = Path(p.outdir).expanduser()
print('using',sys.executable,'in',indir,'with',codedir,'extracting to',outdir)
# %% 0) find directories of data
dlist = [x for x in indir.iterdir() if (x/'spool').is_dir()]

for d in dlist:
# %% 1) create spool/index.h5
    indexcmd = ['python','FileTick.py', d /'spool','-s1296', '-z0']
    subprocess.check_call(indexcmd,cwd=codedir/'dmcutils')