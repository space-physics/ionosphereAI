#!/usr/bin/env python
"""
front end (used from Terminal) to auroral detection program
Michael Hirsch

./Detect.py ~/data/2011-03-01/optical  ~/data/2011-03-01/optical/cv 2011.ini -s

"""
import logging
logging.basicConfig(format='%(asctime)s.%(msecs)03d %(filename)s/%(funcName)s:%(lineno)d %(message)s',
                    datefmt='%Y-%m-%d %H:%M:%S')
from pathlib import Path
#
from cviono import loopaurorafiles

COMPLVL = 4 #tradeoff b/w speed and filesize for TIFF
PSHOW = []
#('thres',
 #        'stat',
 #        'final')
#'raw' #often not useful due to no autoscale
#'rawscaled'      #True  #why not just showfinal
#'hist' ogram
# 'flowvec'
#'flowhsv'
#'thres'
#'morph'
#'final'

def rundetect(p):
    P = {
    'rejvid':    p.rejectvid,
    'framestep': p.step,
    'startstop': p.frames,
    'montstep':  p.ms,
    'clim':      p.contrast,
    'paramfn':   p.paramfn,
    'rejdet':    p.rejectdet,
    'odir':      Path(p.odir).expanduser(),
    'detfn':     Path(p.odir).expanduser() / p.detfn,
    'fps':       p.fps,
     'framebyframe': p.framebyframe,
     'verbose': p.verbose,
     'pshow': PSHOW,
     'complvl': COMPLVL
    }

    P['odir'].mkdir(parents=True,exist_ok=True)


    if p.savetiff:
        P['savevideo']='tif'
    elif p.savevideo:
        P['savevideo']='vid'
    else:
        P['savevideo']=None
#%% run program (allowing ctrl+c to exit)
    aurstat=None #in case of keybaord abort
    try:
        #note, if a specific file is given, vidext is ignored
        idir = Path(p.indir).expanduser()
        if idir.is_file():
            flist = [idir]
        elif idir.is_dir():
            flist = sorted(idir.glob('*'+p.vidext))
        else:
            raise FileNotFoundError('{} is not a path or file'.format(idir))
        print('found {} {} files in {}'.format(len(flist),p.vidext,p.indir))

        if p.profile:
            import cProfile,pstats
            profFN = 'profstats.pstats'
            cProfile.run('loopaurorafiles(flist, P)',profFN)
            pstats.Stats(profFN).sort_stats('time','cumulative').print_stats(50)
            aurstat = None
        else:
            aurstat = loopaurorafiles(flist, P)
    except KeyboardInterrupt:
        print('')

    return aurstat

if __name__=='__main__':
    from argparse import ArgumentParser
    p = ArgumentParser(description='detects aurora in raw video files')
    p.add_argument('indir',help='specify file, OR top directory over which to recursively find video files')
    p.add_argument('odir',help='directory to put output files in')
    p.add_argument('paramfn',help='parameter file for cameras')
    p.add_argument('-e','--vidext',help='suffix of raw video file (.DMCdata,.h5,.fits)',default='.h5')


    p.add_argument('--fps',help='output file FPS (note VLC needs fps>=3)',type=float,default=3)
    p.add_argument('-b','--framebyframe',help='space bar toggles play/pause', action='store_true')
    p.add_argument('-s','--savevideo',help='save video at each step (can make enormous files)',action='store_true')
    p.add_argument('-t','--savetiff',help='save tiff at each step (can make enormous files)',action='store_true')
    p.add_argument('-k','--step',help='frame step skip increment',type=int,default=10)
    p.add_argument('-f','--frames',help='start stop frames (default all)',type=int,nargs=2,default=(None,)*2)
    p.add_argument('-d','--detfn',help='master file to save detections and statistics in HDF5, under odir',default='auroraldet.h5')
    p.add_argument('--ms',help='keogram/montage step [1000] dont make it too small like 1 or output is as big as original file!',type=int,default=1000)
    p.add_argument('-c','--contrast',help='[low high] data numbers to bound video contrast',type=int,nargs=2,default=(None,)*2)
    p.add_argument('--rejectvid',help='reject raw video files with less than this many frames',type=int,default=10)
    p.add_argument('-r','--rejectdet',help='reject files that have fewer than this many detections',type=int,default=10)

    p.add_argument('-v','--verbose',help='verbosity',action='store_true')
    p.add_argument('--profile',help='profile debug',action='store_true')
    p = p.parse_args()

    aurstat = rundetect(p)
