#!/usr/bin/env python3
"""
front end (used from Terminal) to auroral detection program
Michael Hirsch
"""
from histutils.walktree import walktree
from cvhst.detectaurora import loopaurorafiles

def rundetect(p):
    uparams = {'rejvid':p.rejectvid,
              'framestep':p.step,
              'startstop':p.frames,
              'montstep':p.ms,
              'clim':p.contrast,
              'paramfn':p.paramfn,
              'rejdet':p.rejectdet,
              'outdir':p.outdir,
              'fps':p.fps
              }

    if p.savetiff:
        savevideo='tif'
    elif p.savevideo:
        savevideo='vid'
    else:
        savevideo=''
#%% run program (allowing ctrl+c to exit)
    try:
        #note, if a specific file is given, vidext is ignored
        flist = walktree(p.indir,'*.' + p.vidext)
        print(len(flist))
        if p.profile:
            import cProfile,pstats
            profFN = 'profstats.pstats'
            cProfile.run('loopaurorafiles(flist, uparams, savevideo, p.framebyframe, p.verbose)',profFN)
            pstats.Stats(profFN).sort_stats('time','cumulative').print_stats(50)
        else:
            loopaurorafiles(flist, uparams, savevideo, p.framebyframe, p.verbose)
            #show()
    except KeyboardInterrupt:
        exit('aborting per user request')

if __name__=='__main__':
    from argparse import ArgumentParser
    p = ArgumentParser(description='detects aurora in raw video files')
    p.add_argument('indir',help='specify file, OR top directory over which to recursively find video files',nargs='+')
    p.add_argument('-e','--vidext',help='extension of raw video file',default='DMCdata')
    p.add_argument('--fps',help='output file FPS (note VLC needs fps>=3)',type=float,default=3)
    p.add_argument('-p','--framebyframe',help='space bar toggles play/pause', action='store_true')
    p.add_argument('-s','--savevideo',help='save video at each step (can make enormous files)',action='store_true')
    p.add_argument('-t','--savetiff',help='save tiff at each step (can make enormous files)',action='store_true')
    p.add_argument('-k','--step',help='frame step skip increment (default 10000)',type=int,default=1)
    p.add_argument('-f','--frames',help='start stop frames (default all)',type=int,nargs=2,default=(None,)*2)
    p.add_argument('-o','--outdir',help='directory to put output files in',type=str,default='') #None doesn't work with Windows
    p.add_argument('--ms',help='keogram/montage step [1000] dont make it too small like 1 or output is as big as original file!',type=int,default=1000)
    p.add_argument('-c','--contrast',help='[low high] data numbers to bound video contrast',type=int,nargs=2,default=(None,)*2)
    p.add_argument('--rejectvid',help='reject raw video files with less than this many frames',type=int,default=10)
    p.add_argument('-r','--rejectdet',help='reject files that have fewer than this many detections',type=int,default=10)
    p.add_argument('--paramfn',help='parameter file for cameras',type=str,default='camparam.xlsx')
    p.add_argument('-v','--verbose',help='verbosity',action='store_true')
    p.add_argument('--profile',help='profile debug',action='store_true')
    p = p.parse_args()

    rundetect(p)
