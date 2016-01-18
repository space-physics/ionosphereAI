#!/usr/bin/env python3
"""
front end (used from Terminal) to auroral detection program
Michael Hirsch
"""
from __future__ import absolute_import
from tempfile import gettempdir
from pathlib import Path
#
from cvhst.detectaurora import loopaurorafiles

def rundetect(p):
    uparams = {'rejvid':p.rejectvid,
              'framestep':p.step,
              'startstop':p.frames,
              'montstep':p.ms,
              'clim':p.contrast,
              'paramfn':p.paramfn,
              'rejdet':p.rejectdet,
              'outdir':Path(p.outdir).expanduser(),
              'detfn': Path(p.outdir).expanduser() / p.detfn,
              'fps':p.fps
              }

    uparams['outdir'].mkdir(parents=True,exist_ok=True)


    if p.savetiff:
        savevideo='tif'
    elif p.savevideo:
        savevideo='vid'
    else:
        savevideo=''
#%% run program (allowing ctrl+c to exit)
    try:
        #note, if a specific file is given, vidext is ignored
        flist = sorted(Path(p.indir).expanduser().glob('*'+p.vidext))
        print('found {} {} files in {}'.format(len(flist),p.vidext,p.indir))

        if p.profile:
            import cProfile,pstats
            profFN = 'profstats.pstats'
            cProfile.run('loopaurorafiles(flist, uparams, detfn,savevideo, p.framebyframe, p.verbose)',profFN)
            pstats.Stats(profFN).sort_stats('time','cumulative').print_stats(50)
            aurstat = None
        else:
            aurstat = loopaurorafiles(flist, uparams, savevideo, p.framebyframe, p.verbose)
    except KeyboardInterrupt:
        print('aborting per user request')

    return aurstat

if __name__=='__main__':
    from argparse import ArgumentParser
    p = ArgumentParser(description='detects aurora in raw video files')
    p.add_argument('indir',help='specify file, OR top directory over which to recursively find video files')
    p.add_argument('-e','--vidext',help='suffix of raw video file (.DMCdata,.h5,.fits)',default='.h5')
    p.add_argument('--fps',help='output file FPS (note VLC needs fps>=3)',type=float,default=3)
    p.add_argument('-b','--framebyframe',help='space bar toggles play/pause', action='store_true')
    p.add_argument('-s','--savevideo',help='save video at each step (can make enormous files)',action='store_true')
    p.add_argument('-t','--savetiff',help='save tiff at each step (can make enormous files)',action='store_true')
    p.add_argument('-k','--step',help='frame step skip increment',type=int,default=10)
    p.add_argument('-f','--frames',help='start stop frames (default all)',type=int,nargs=2,default=(None,)*2)
    p.add_argument('-o','--outdir',help='directory to put output files in',default=gettempdir()) #None doesn't work with Windows
    p.add_argument('-d','--detfn',help='master file to save detections and statistics in HDF5, under outdir',default='auroraldet.h5')
    p.add_argument('--ms',help='keogram/montage step [1000] dont make it too small like 1 or output is as big as original file!',type=int,default=1000)
    p.add_argument('-c','--contrast',help='[low high] data numbers to bound video contrast',type=int,nargs=2,default=(None,)*2)
    p.add_argument('--rejectvid',help='reject raw video files with less than this many frames',type=int,default=10)
    p.add_argument('-r','--rejectdet',help='reject files that have fewer than this many detections',type=int,default=10)
    p.add_argument('-p','--paramfn',help='parameter file for cameras',default='camparam.xlsx')
    p.add_argument('-v','--verbose',help='verbosity',action='store_true')
    p.add_argument('--profile',help='profile debug',action='store_true')
    p = p.parse_args()

    aurstat = rundetect(p)
