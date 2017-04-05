from warnings import warn
import logging
import cv2
from astropy.io import fits
import h5py
import numpy as np
from scipy.signal import wiener
from scipy.misc import bytescale
#
from matplotlib.pyplot import figure, hist
#
from .getpassivefm import getfmradarframe
from histutils.rawDMCreader import getDMCframe
from dmcutils.neospool import readNeoSpool

def setscale(fn,ap,finf):
    """
    if user has not set fixed upper/lower bounds for 16-8 bit conversion, do it automatically,
    since not specifying fixed contrast destroys the working of auto CV detection
    """
    pt = [0.01,0.99] #percentiles
    mod = False

    if not isinstance(ap['rawlim'][0],(float,int)):
        mod = True
        prc = samplepercentile(fn,pt[0],finf)
        ap['rawlim'][0] = prc

    if not isinstance(ap['rawlim'][1],(float,int)):
        mod = True
        prc = samplepercentile(fn,pt[1],finf)
        ap['rawlim'][1] = prc

#%%
    cdiff = ap['rawlim'][1] - ap['rawlim'][0]

    assert cdiff>0, f'raw limits do not make sense  lower: {ap["rawlim"][0]}   upper: {ap["rawlim"][1]}'

    if cdiff < 20:
        raise ValueError('your video may have very poor contrast and not work for auto detection')
#%%
    if mod:
        print(f'data number lower,upper  {ap["rawlim"][0]}  {ap["rawlim"][1]}')

    return ap

def samplepercentile(fn,pct,finf):
    """
    for 16-bit files mainly
    """
    isamp = (finf['nframe'] * np.array([.1,.25,.5,.75,.9])).astype(int)  # pick a few places in the file to touch

    isamp = isamp[:finf['nframe']-1] #for really small files

    dat = np.empty((isamp.size,finf['supery'],finf['superx']),float)

    tmp = getraw(fn,0,finf)[3]
    if tmp.dtype.itemsize < 2:
        warn('usually we use autoscale with 16-bit video, not 8-bit.')

    for j,i in enumerate(isamp):
        dat[j,...] = getraw(fn, i, finf)[3]

    return np.percentile(dat,pct).astype(int)


def getraw(fn,ifrm,finf,svh=None,ap=None,cp=None,up=None):
    """ this function reads the reference frame too--which makes sense if youre
       only reading every Nth frame from the multi-TB file instead of every frame
    """
    if up and (not isinstance(ap['rawlim'][0],(float,int)) or not isinstance(ap['rawlim'][1],(float,int))):
        warn('not specifying fixed contrast will lead to very bad automatic detection results')

    frameref = None #just in case not used
    if cp:
        dowiener = cp.get('filter','wienernhood')
        if not dowiener.strip():
            dowiener = False
        else:
            dowiener = int(dowiener)

    else:
        dowiener = None

    if not ap:
        ap = {'twoframe': None, 'rawlim': (None,None)}
#%% reference frame
    if finf['reader'] == 'raw':
        if ap['twoframe']:
            frameref = getDMCframe(fn,ifrm,finf)[0]
            frameref = bytescale(frameref, ap['rawlim'][0], ap['rawlim'][1])
            if dowiener:
                frameref = wiener(frameref, dowiener)

        try:
            frame16,rfi = getDMCframe(fn,ifrm+1,finf)
            framegray = bytescale(frame16, ap['rawlim'][0], ap['rawlim'][1])
        except (ValueError,IOError):
            ap['rawframeind'] = np.delete(ap['rawframeind'], np.s_[ifrm:])
            raise
    elif finf['reader'] == 'spool':
        rfi = ifrm

        if ap['twoframe']:
            frameref = readNeoSpool(fn,finf,[ifrm])[0].squeeze()
            frameref = bytescale(frameref, ap['rawlim'][0], ap['rawlim'][1])
            if dowiener:
                frameref = wiener(frameref, dowiener)

            try:
                frame16 = readNeoSpool(fn,finf,[ifrm+1])[0].squeeze()
                framegray = bytescale(frame16, ap['rawlim'][0], ap['rawlim'][1])
            except (ValueError,IOError):
                ap['rawframeind'] = np.delete(ap['rawframeind'], np.s_[ifrm:])
                raise
    elif finf['reader'] == 'cv2':
        if ap['twoframe']:
            retval,frameref = fn.read() #TODO best to open cv2.VideoReader in calling function as CV_CAP_PROP_POS_FRAMES is said not to always work vis keyframes
            if not retval:
                if ifrm==0:
                    logging.error('could not read video file, sorry')
                print('done reading video.')
                return None, None, ap
            if frameref.ndim>2:
                frameref = cv2.cvtColor(frameref, cv2.COLOR_RGB2GRAY)
            if dowiener:
                frameref = wiener(frameref, dowiener)

        retval,frame16 = fn.read() #TODO this is skipping every other frame!
        # TODO can we use dfid.set(cv.CV_CAP_PROP_POS_FRAMES,ifrm) to set 0-based index of next frame?
        rfi = ifrm
        if not retval:
            raise IOError('could not read video from {}'.format(fn))

        if frame16.ndim>2:
            framegray = cv2.cvtColor(frame16, cv2.COLOR_RGB2GRAY)
        else:
            framegray = frame16 #copy NOT needed
    elif finf['reader'] == 'h5fm':   #one frame per file
        if ap['twoframe']:
            frameref = getfmradarframe(fn[ifrm])[2]
            frameref = bytescale(frameref, ap['rawlim'][0], ap['rawlim'][1])
        frame16 = getfmradarframe(fn[ifrm+1])[2]
        rfi = ifrm
        framegray = bytescale(frame16, ap['rawlim'][0], ap['rawlim'][1])
    elif finf['reader'] == 'h5vid':
        with h5py.File(str(fn),'r',libver='latest') as f:
            if ap['twoframe']:
                frameref = bytescale(f['/rawimg'][ifrm,...],
                                     ap['rawlim'][0], ap['rawlim'][1])
                if dowiener:
                    frameref = wiener(frameref, dowiener)

            #keep frame16 for histogram
            frame16 = f['/rawimg'][ifrm+1,...]
        framegray = bytescale(frame16,
                                 ap['rawlim'][0], ap['rawlim'][1])
        rfi = ifrm

    elif finf['reader'] == 'fits':
        #memmap = False required thru Astropy 1.1.1 due to BZERO used...
        with fits.open(str(fn),mode='readonly',memmap=False) as f:
            if ap['twoframe']:
                frameref = bytescale(f[0].data[ifrm,...],
                                     ap['rawlim'][0], ap['rawlim'][1])
                if dowiener:
                    frameref = wiener(frameref, dowiener)

            frame16 = f[0].data[ifrm+1,...]

        framegray = bytescale(frame16, ap['rawlim'][0], ap['rawlim'][1])

        rfi = ifrm #TODO: incorrect raw index with sequence of fits files
    else:
        raise TypeError(f'unknown reader type {finf["reader"]}')


#%% current frame
    if 'rawframeind' in ap:
        ap['rawframeind'][ifrm] = rfi

    if dowiener:
        framegray = wiener(framegray, dowiener)

    if up and 'raw' in up['pshow']:
        # cv2.imshow just divides by 256, NOT autoscaled!
        # http://docs.opencv.org/modules/highgui/doc/user_interface.html
        cv2.imshow('video', framegray)
#%% plotting
    if up and 'rawscaled' in up['pshow']:
        cv2.imshow('raw video, scaled to 8-bit', framegray)
    # image histograms (to help verify proper scaling to uint8)
    if up and 'hist' in up['pshow']:
        ax=figure().gca()
        hist(frame16.flatten(), bins=128, fc='w',ec='k', log=True)
        ax.set_title('raw uint16 values')

        ax=figure().gca()
        hist(framegray.flatten(), bins=128, fc='w',ec='k', log=True)
        ax.set_xlim((0,255))
        ax.set_title('normalized video into opt flow')

    if svh and svh['video'] is not None:
        if up['savevideo'] == 'tif':
            svh['video'].save(framegray, compress=up['complvl'])
        elif up['savevideo'] == 'vid':
            svh['video'].write(framegray)

    return framegray,frameref,ap,frame16
