#!/usr/bin/env python
"""
find minimum and maximum, stats in an HDF5 variable
"""
from pathlib import Path
import h5py
import numpy as np
import warnings
from argparse import ArgumentParser


def main():
    p = ArgumentParser()
    p.add_argument('fn', help='HDF5 filename')
    p.add_argument('var', help='HDF5 variable to analyze')
    p = p.parse_args()

    fn = Path(p.fn).expanduser()

    key = p.var

    with h5py.File(fn, 'r') as f:
        if f[key].size > 10e9:
            warnings.warn(f'this might take a long time: large variable size {f[key].size} elements')

        dat = f[key][:]

    # %%
    fmin = dat.min()
    fmax = dat.max()

    print('min, max: ', fmin, fmax)

    prc = np.array([0.01, 0.05, 0.5, 0.95, 0.99])
    ptile = np.percentile(dat, prc)
    print(f'for the {prc*100} percentiles')
    print(ptile)

    if ptile[-1] - ptile[0] < 20:
        print(f'if {fn} is a video file, it might have poor contrast or not have many changes')


if __name__ == '__main__':
    main()
