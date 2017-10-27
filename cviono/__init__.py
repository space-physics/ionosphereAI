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
import h5py
from datetime import datetime
from pytz import UTC
from pandas import DataFrame
from pathlib import Path
from time import time
import numpy as np
from scipy.ndimage import zoom
from matplotlib.pylab import draw, pause, close
#
from histutils import setupimgh5
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

# %% process files
    if P.get('main','vidext') == '.dat':
        aurstat = procfiles(flist,P,U)
    else:
        aurstat = DataFrame(columns=['mean', 'median', 'variance', 'detect']) # FIXME this is where detect is being cast to float, despite being int in individual dataFrames
        for f in flist:  # iterate over files in list
            stat = procfiles(f,P,U)
            aurstat = aurstat.append(stat)

    if aurstat is None:
        return
# %% sort,plot,save results for all files

    aurstat.sort_index(inplace=True)  # sort by time
    savestat(aurstat, U['detfn'], idir, U)

    if aurstat.index[0] > 1e9: #ut1 instead of index
        dt = [datetime.fromtimestamp(t,tz=UTC) for t in stat.index]
    else:
        dt=None
# %% master detection plot
    U['pshow'] += ['stat']
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
    if len(N)==0:
        logging.error(f'no files found to detect in {f}')
        return
# %% start main loop
    #print('start main loop')
    if finf['reader'] == 'spool':
        zy,zx = zoom(getraw(finf['path']/flist[0], 0,0, finf, svh, P, U)[3], 0.1, order=0).shape
        #zy,zx=(64,64)
        setupimgh5(U['detfn'], np.ceil(N.size/U['previewdecim'])+1, zy, zx, np.uint16, writemode='w',key='/preview',cmdlog=U['cmd'])

    j=0
    for i, iraw in enumerate(N):
        if finf['reader'] == 'spool':
            f = finf['path'] / flist[i]
            iraw = 0
        #print(f,i,iraw)
# %% load and filter
        framegray, frameref, U, frame16 = getraw(f, iraw-N[0],iraw, finf, svh, P, U)
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

        if not i % U['previewdecim']:
            j+=1
            if finf['reader'] == 'spool':
                with h5py.File(U['detfn'], 'r+', libver='latest') as f5:
                    #ipy = slice(finf['supery']//2-zy//2,finf['supery']//2+zy//2)
                    #ipx = slice(finf['superx']//2-zx//2,finf['superx']//2+zx//2)
                    #preview = frame16[ipy,ipx].astype(np.uint16)
                    preview = zoom(frame16, 0.1, order=0,mode='nearest')
#                    assert (frame16[ipy,ipx] == preview).all(),'preview failure to convert'

                    updatestr = f'mean frame {i}: {preview.mean():.1f}  j= {j}'
                    print(updatestr)
                    f5['/preview'][j,...] = preview


#                with h5py.File(U['detfn'], 'r', libver='latest') as f5:
#                    assert (f5['/preview'][j,...] == preview).all(),'preview failure to store'
#                    assert not (f5['/preview'][j,...] == 0).all(),'preview all 0 frame'

#                if U['verbose']:
#                    fgv = figure(1002)
#                    fgv.clf()
#                    ax = fgv.gca()
#                    hi = ax.imshow(preview)
#                    ax.set_title(updatestr)
#                    fgv.colorbar(hi,ax=ax)
#                    draw(); pause(0.001)


            print(f'{U["framestep"]*i/N[-1]*100:.2f}% {stat["detect"].iloc[i-U["previewdecim"]:i].values}')
            if (framegray == 255).sum() > 40: #arbitrarily allowing up to 40 pixels to be saturated at 255, to allow for bright stars and faint aurora
                logging.warning('video saturated at 255')
            if (framegray == 0).sum() > 4:
                logging.warning('video saturated at 0')

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
            if finf['reader'] != 'spool':
                savestat(stat,detfn, U['indir'],U)
                if 'stat' in U['pshow']:
                    print(f'saving detection plot to {detpltfn}')
                    U['fdet'].savefig(str(detpltfn), dpi=100, bbox_inches='tight')
        except Exception as e:
            logging.critical(f'trouble saving detection result  {e} ')
        finally:
            svrelease(svh, U['savevideo'])

    close('all')

    return stat
