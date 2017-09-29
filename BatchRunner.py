#!/usr/bin/env python
"""
batch processing of entire directories of images (one directory per night typically)
does NOT automatically delete original data
"""
import subprocess
from argparse import ArgumentParser
p = ArgumentParser()
p.add_argument('datadir',help='directory containing directories to process (top level directory)')
p.add_argument('outdir',help='directory to place index, extracted frames and AVI preview')
p = p.parse_args()


