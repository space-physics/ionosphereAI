#!/usr/bin/env python
"""
find minimum and maximum, stats in an HDF5 variable
"""
from cviono import Path
import h5py
import numpy as np
from warnings import warn

from argparse import ArgumentParser
p = ArgumentParser()
p.add_argument('fn',help='HDF5 filename')
p.add_argument('var',help='HDF5 variable to analyze')
p = p.parse_args()

fn = Path(p.fn).expanduser()

key = p.var

with h5py.File(str(fn),'r') as f:
    if f[key].size > 10e9:
        warn('these operations might take a long time due to large variable size {} elements'.format(f[key].size))

    dat = f[key][:]

#%%
fmin = dat.min()
fmax = dat.max()

print('min, max:  {}  {}'.format(fmin,fmax))

prc = np.array([0.01,0.05,0.5,0.95,0.99])
ptile = np.percentile(dat,prc)
print('for the {} percentiles'.format(prc*100))
print(ptile)

if ptile[-1] - ptile[0] < 20:
    print('if {} is a video file, it might have poor contrast or not have many changes'.format(fn))

