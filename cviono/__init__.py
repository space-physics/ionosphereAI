#!/usr/bin/env python
"""
Michael Hirsch Dec 2014
This program detects aurora in multi-terabyte raw video data files
Also used for the Haystack passive FM radar ionospheric activity detection
"""
import logging
import cv2
print(f'OpenCV {cv2.__version__}')
#
from .io import getparam, getvidinfo, keyhandler, savestat
from .reader import getraw, setscale
from .cvops import dooptflow,dothres,dodespeck,domorph,doblob
from .cvsetup import setupkern,svsetup,svrelease,setupof,setupfigs,statplot
#
from datetime import datetime
from pytz import UTC
from pandas import DataFrame
from pathlib import Path
from time import time
from matplotlib.pylab import draw, pause, close
#
from morecvutils.connectedComponents import setupblob


def loopaurorafiles(U):

    P = getparam(U['paramfn']) # ConfigParser object

    # note, if a specific file is given, vidext is ignored
    idir = Path(U['indir']).expanduser()
    if idir.is_file():
        flist = [idir]
    elif idir.is_dir():
        flist = sorted(idir.glob('*'+P.get('main', 'vidext')))
    else:
        raise FileNotFoundError(f'{idir} is not a path or file')

    if not flist:
        raise FileNotFoundError(f'no files found: {U["indir"]}')

    U['nfile'] = len(flist)

    print(f'found {U["nfile"]} files: {U["indir"]}')

    aurstat = DataFrame(columns=['mean', 'median', 'variance', 'detect']) # FIXME this is where detect is being cast to float, despite being int in individual dataFrames
# %% process files
    if P.get('main','vidext') == '.dat':
        stat = procfiles(flist,P,U)
        aurstat = aurstat.append(stat)
    else:
        for f in flist:  # iterate over files in list
            stat = procfiles(f,P,U)
            aurstat = aurstat.append(stat)
# %% sort,plot,save results for all files
    aurstat.sort_index(inplace=True)  # sort by time
    savestat(aurstat, U['detfn'], idir)

    if aurstat.index[0] > 1e9: #ut1 instead of index
        dt = [datetime.fromtimestamp(t,tz=UTC) for t in stat.index]
    else:
        dt=None
# %% master detection plot
    U['pshow'] += ('stat',)
    fgst = statplot(dt, aurstat, U, P, U['odir'])[3]
    draw(); pause(0.01)
    fgst.savefig(str(U['detfn'].with_suffix('.png')), bbox_inches='tight', dpi=100)
    fgst.savefig(str(U['detfn'].with_suffix('.svg')), bbox_inches='tight', dpi=100)

    return aurstat

def procfiles(f,P,U):
    finf, U = getvidinfo(f, P, U)

    if finf['nframe'] < 100 and finf['reader'] != 'spool':
        print(f'SKIPPING {f} with only {finf["nframe"]} frames')
        return

    try:
        U = setscale(f, U, finf)  # in case auto contrast per file
    except ValueError as e:
        logging.error(f'{f}  {e}')
        print()
        return

    stat = procaurora(f, P, U, finf)

    return stat


def procaurora(f, P,U,finf):
    tic = time()

    if finf is None:
        return
#%% setup optional video/tiff writing (mainly for debugging or publication)
    svh = svsetup(P, U)
#%% setup blob
    blobdetect = setupblob(P.getfloat('blob','minblobarea'),
                           P.getfloat('blob','maxblobarea'),
                           P.getfloat('blob','minblobdist'))
#%% cv opt. flow matrix setup
    lastflow,gmm = setupof(U,P)
# %% kernel setup
    U = setupkern(P, U)
# %% mag plots setup
    U, stat = setupfigs(finf, f, U, P)
# %% list of files or handle?
    try:
        flist = finf['flist'].iloc[finf['frameind']].tolist()
    except KeyError:
        flist=f

    N = finf['frameind'][:-1]
# %% start main loop
    #print('start main loop')
    for i, iraw in enumerate(N):
        if finf['reader'] == 'spool':
            f = finf['path'] / flist[i]
            iraw = 0
        #print(f,i,iraw)
# %% load and filter
        framegray, frameref, U = getraw(f, iraw-N[0],iraw, finf, svh, P, U)[:3]
# %% compute optical flow or Background/Foreground
        if gmm is None:
            flow, mag, stat = dooptflow(framegray, frameref, lastflow,
                                             iraw, i, U, P, stat)
            lastflow = flow.copy()  # FIXME is the .copy() strictly necessary?
        else:  # background/foreground
            mag = gmm.apply(framegray)
# %% threshold
        thres = dothres(mag, stat['median'].iat[i], P, i, svh,  U, gmm is not None)
#%% despeckle
        despeck = dodespeck(thres,P.getint('filter','medfiltsize'),i,svh, U)
#%% morphological ops
        morphed = domorph(despeck, svh, U)
#%% blob detection
        stat = doblob(morphed,blobdetect,framegray,i,svh,stat, U) #lint:ok
#%% plotting in loop
        """
        http://docs.opencv.org/modules/highgui/doc/user_interface.html
        """
        if U['pshow']:
            draw(); pause(0.01)

        if not i % 50:
            print(f'i={iraw:0d} {stat["detect"].iloc[i-50:i].values}')
            if (framegray == 255).sum() > 40: #arbitrarily allowing up to 40 pixels to be saturated at 255, to allow for bright stars and faint aurora
                print('* Warning: video may be saturated at value 255, missed detections can result')
            if (framegray == 0).sum() > 4:
                print('* Warning: video may be saturated at value 0, missed detections can result')

        if U['pshow']:
            if U['framebyframe']: #wait indefinitely for spacebar press
                keypressed = cv2.waitKey(0)
                U['framebyframe'],dobreak = keyhandler(keypressed,U['framebyframe'])
            else:
                keypressed = cv2.waitKey(1)
                framebyframe, dobreak = keyhandler(keypressed,U['framebyframe'])

            if dobreak:
                break
#%% done looping this file  save results for this file
    print(f'{time()-tic:.1f} seconds to process {f}')

    if U['odir']:
        detfn =    Path(U['odir']).expanduser()/(f.stem +'_detections.h5')
        detpltfn = Path(U['odir']).expanduser()/(f.stem +'_detections.png')
        if detfn.is_file():
            logging.warning(f'overwriting existing {detfn}')

        try:
            savestat(stat,detfn, U['indir'])
            if 'stat' in U['pshow']:
                print(f'saving detection plot to {detpltfn}')
                U['fdet'].savefig(str(detpltfn), dpi=100, bbox_inches='tight')
        except Exception as e:
            logging.critical(f'trouble saving detection result  {e} ')
        finally:
            svrelease(svh, U['savevideo'])

    close('all')

    return stat
