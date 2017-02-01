#!/usr/bin/env python
"""
Michael Hirsch Dec 2014
This program detects aurora in multi-terabyte raw video data files
It is also used for the Haystack passive FM radar ionospheric activity detection
"""
from __future__ import division, absolute_import
import logging
import cv2
print('OpenCV {}'.format(cv2.__version__)) #some installs of OpenCV don't give a consistent version number, just a build number and I didn't bother to parse this.
#
from datetime import datetime
from pytz import UTC
from pandas import DataFrame
from pathlib import Path
from time import time
from matplotlib.pylab import draw, pause, close
#
from .io import getparam,getvidinfo,keyhandler,savestat
from .reader import getraw, setscale
from .cvops import dooptflow,dothres,dodespeck,domorph,doblob
from .cvsetup import setupkern,svsetup,svrelease,setupof,setupfigs,statplot
#
from cvutils.connectedComponents import setupblob

def loopaurorafiles(flist, up):
    if not flist:
        raise ValueError('no files found')

    P = getparam(up['paramfn'])

    aurstat = DataFrame(columns=['mean','median','variance','detect'])

    for f in flist: #iterate over files in list
        finf,ap = getvidinfo(f, P,up)

        if finf['nframe'] < 100:
            print('SKIPPING {} with only {} frames'.format(f,finf['nframe']))
            continue

        try:
            ap = setscale(f,ap,finf) # in case auto contrast per files
        except ValueError as e:
            logging.error('{}  {}'.format(f,e))
            print()
            continue

        stat = procaurora(f, P, up,ap,finf)
        aurstat = aurstat.append(stat)
#%% sort,plot,save results for all files
    try:
        aurstat.sort_index(inplace=True) #sort by time
        savestat(aurstat,up['detfn'])

        if stat.index[0] > 1e9: #ut1 instead of index
            dt = [datetime.fromtimestamp(t,tz=UTC) for t in stat.index]
        else:
            dt=None

        fgst = statplot(dt,aurstat.index,aurstat['mean'],aurstat['median'],aurstat['detect'],
                 fn=up['odir'],pshow='stat')[3]
        draw(); pause(0.001)
        fgst.savefig(str(up['detfn'].with_suffix('.png')),bbox_inches='tight',dpi=100)

        return aurstat
    except UnboundLocalError:
        raise RuntimeError('no good files found in {}'.format(flist[0].parent))


def procaurora(f, P,up,ap,finf):
    framebyframe = up['framebyframe']
    tic = time()

    if finf is None:
        return
#%% setup optional video/tiff writing (mainly for debugging or publication)
    svh = svsetup(ap, P, up)
#%% setup blob
    blobdetect = setupblob(P.getfloat('blob','minblobarea'),
                           P.getfloat('blob','maxblobarea'),
                           P.getfloat('blob','minblobdist'))
#%% cv opt. flow matrix setup
    uv, lastflow,gmm = setupof(ap,P)
    isgmm = lastflow is None
#%% kernel setup
    kern = setupkern(ap, P)
#%% mag plots setup
    pl,stat = setupfigs(finf,f, up['pshow'])
#%% start main loop
    for i,iraw in enumerate(finf['frameind'][:-1]):
#%% load and filter
        try:
            framegray,frameref,ap = getraw(f,iraw,finf,svh,ap, P,up)[:3]
        except Exception as e:
            print('{}  {}'.format(f,e))
            break
#%% compute optical flow or Background/Foreground
        if ~isgmm:
            flow,ofmaggmm,stat = dooptflow(framegray,frameref,lastflow,uv,
                                               iraw,i, ap, P,pl,stat, up['pshow'])
            lastflow = flow.copy() #I didn't check if the .copy() is strictly necessary
        else: #background/foreground
            ofmaggmm = gmm.apply(framegray)
#%% threshold
        thres = dothres(ofmaggmm, stat['median'].iat[i],ap, P,i,svh, up['pshow'],isgmm)
#%% despeckle
        despeck = dodespeck(thres,P.getint('filter','medfiltsize'),i,svh, up['pshow'])
#%% morphological ops
        morphed = domorph(despeck,kern,svh, up['pshow'])
#%% blob detection
        stat = doblob(morphed,blobdetect,framegray,i,svh,pl,stat, up['pshow']) #lint:ok
#%% plotting in loop
        """
        http://docs.opencv.org/modules/highgui/doc/user_interface.html
        """
        draw(); pause(0.01)

        if not i % 40:
            print('i={:0d} {}'.format(iraw,stat.loc[i-40:i,'detect'].values))
            if (framegray == 255).sum() > 40: #arbitrarily allowing up to 40 pixels to be saturated at 255, to allow for bright stars and faint aurora
                print('* Warning: video may be saturated at value 255, missed detections can result')
            if (framegray == 0).sum() > 4:
                print('* Warning: video may be saturated at value 0, missed detections can result')

        if up['pshow']:
            if framebyframe: #wait indefinitely for spacebar press
                keypressed = cv2.waitKey(0)
                framebyframe,dobreak = keyhandler(keypressed,framebyframe)
            else:
                keypressed = cv2.waitKey(1)
                framebyframe, dobreak = keyhandler(keypressed,framebyframe)

            if dobreak:
                break
#%% done looping this file  save results for this file
    print('{:.1f} seconds to process {}'.format(time()-tic,f))

    if up['odir']:
        detfn =    Path(up['odir']).expanduser()/(f.stem +'_detections.h5')
        detpltfn = Path(up['odir']).expanduser()/(f.stem +'_detections.png')
        if detfn.is_file():
            logging.warning('overwriting existing %s', detfn)

        try:
            savestat(stat,detfn)

            print('saving detection plot to {}'.format(detpltfn))
            pl['fdet'].savefig(str(detpltfn),dpi=100,bbox_inches='tight')

        except Exception as e:
            logging.critical('trouble saving detection result   '.format(e))
        finally:
            svrelease(svh, up['savevideo'])

    close('all')

    return stat
