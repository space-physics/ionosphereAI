#!/usr/bin/python2
"""
Michael Hirsch Dec 2014
This program detects aurora in multi-terabyte raw video data files
It is also used for the Haystack passive FM radar ionospheric activity detection
"""
from __future__ import division, absolute_import
import logging
import cv2
print('OpenCV '+str(cv2.__version__)) #some installs of OpenCV don't give a consistent version number, just a build number and I didn't bother to parse this.
#
from astropy.io import fits
import h5py
from pandas import read_excel, DataFrame
from pathlib import Path
import numpy as np
from scipy.signal import wiener
from scipy.misc import bytescale
from time import time
#
from .cvops import dooptflow,dothres,dodespeck,domorph,doblob
from .cvsetup import setupkern,svsetup,svrelease,setupof,setupfigs
from .getpassivefm import getfmradarframe
#
from cvutils.getaviprop import getaviprop
from cvutils.connectedComponents import setupblob
#
from histutils.rawDMCreader import getDMCparam,getDMCframe,getserialnum,getNeoParam

#plot disable
pshow = ('thres',
         'meanmedian',
         'final',
         'det','savedet')
#'raw' #often not useful due to no autoscale
#'rawscaled'      #True  #why not just showfinal
#'hist' ogram
# 'flowvec'
#'flowhsv'
#'thres'
#'ofmag'
#'meanmedian'
#'morph'
#'final'
#'det'
#'savedet' #save detection results (normally use this)
complvl = 4 #tradeoff b/w speed and filesize for TIFF

#only import matplotlib if needed to save time
if np.in1d(('det','hist','ofmag','meanmedian','savedet'),pshow).any():
    from matplotlib.pylab import draw, pause, figure, hist


def loopaurorafiles(flist, up,savevideo, framebyframe, verbose):
    if not flist:
        raise ValueError('no files found')

    camser,camparam = getcamparam(up['paramfn'],flist)

    aurstat = DataFrame(columns=['mean','median','variance','detect'])

    for f,s in zip(flist,camser): #iterate over files in list
        stat = procaurora(f,s,camparam,up,savevideo,framebyframe,verbose)
        aurstat.append(stat)

    aurstat.sort_index(inplace=True) #sort by time
    savestat(aurstat,up['detfn'])

    return aurstat


def procaurora(f,s,camparam,up,savevideo,framebyframe,verbose=False):
    tic = time()

    try:
        cp = camparam[s] #pick the parameters for this camera from pandas DataFrame
    except (KeyError,ValueError):
        logging.info('using first column of {} as I didnt find serial # {} in it.'.format(up['paramfn'],s))
        cp = camparam.iloc[:,0] #fallback to first column

    finf,ap,dfid = getvidinfo(f,cp,up,verbose)
    if finf is None:
        return
#%% setup optional video/tiff writing (mainly for debugging or publication)
    svh = svsetup(savevideo,complvl, ap, cp, up,pshow)
#%% setup blob
    blobdetect = setupblob(cp['minblobarea'], cp['maxblobarea'], cp['minblobdist'])
#%% cv opt. flow matrix setup
    uv,lastflow, ofmed, gmm = setupof(ap,cp)
#%% kernel setup
    kern = setupkern(ap,cp)
#%% mag plots setup
    pl,stat = setupfigs(finf,f,pshow)
#%% start main loop
    for jfrm,ifrm in enumerate(finf['frameind'][:-1]):
#%% load and filter
        try:
            framegray,frameref,ap = getraw(dfid,ifrm,finf,svh,ap,cp,savevideo,verbose)
        except:
            break
#%% compute optical flow or Background/Foreground
        if lastflow is not None: #very fast way to check mode
            flow,ofmaggmm,stat = dooptflow(framegray,frameref,lastflow,uv,
                                               ifrm,jfrm, ap,cp,pl,stat,pshow)
            lastflow = flow.copy() #I didn't check if the .copy() is strictly necessary
        else: #background/foreground
            ofmaggmm = gmm.apply(framegray)
#%% threshold
        thres = dothres(ofmaggmm, ofmed, ap,cp,svh,pshow)
#%% despeckle
        despeck = dodespeck(thres,cp['medfiltsize'],svh,pshow)
#%% morphological ops
        morphed = domorph(despeck,kern,svh,pshow)
#%% blob detection
        stat = doblob(morphed,blobdetect,framegray,ifrm,jfrm,svh,pl,stat,pshow) #lint:ok
#%% plotting in loop
        """
        http://docs.opencv.org/modules/highgui/doc/user_interface.html
        """

        if np.in1d(('det','hist','ofmag','meanmedian'),pshow).any():
            draw(); pause(0.001)

        if not ifrm % 50:
            print('frame {:0d}'.format(ifrm))
            if (framegray == 255).sum() > 40: #arbitrarily allowing up to 40 pixels to be saturated at 255, to allow for bright stars and faint aurora
                print('* Warning: video may be saturated at value 255, missed detections can result')
            if (framegray == 0).sum() > 4:
                print('* Warning: video may be saturated at value 0, missed detections can result')

        if framebyframe: #wait indefinitely for spacebar press
            keypressed = cv2.waitKey(0)
            framebyframe,dobreak = keyhandler(keypressed,framebyframe)
        else:
            keypressed = cv2.waitKey(1)
            framebyframe, dobreak = keyhandler(keypressed,framebyframe)
        if dobreak:
            break
#%% done looping this file

#%% close data file
    try:
        if finf['reader'] in ('raw','fits'):
            dfid.close()
        elif finf['reader'] == 'cv2':
            dfid.release()
    except Exception as e:
        print(str(e))
#%% save results for this file
    print('{:.1f} seconds to process {}'.format(time()-tic,f))

    if 'savedet' in pshow:
        detfn =    Path(up['outdir']).expanduser()/(f.stem +'_detections.h5')
        detpltfn = Path(up['outdir']).expanduser()/(f.stem +'_detections.png')
        if detfn.is_file():
            logging.warning('overwriting existing %s', detfn)

        try:
            savestat(stat,detfn)

            print('saving detection plot to {}'.format(detpltfn))
            pl['fdet'].savefig(str(detpltfn),dpi=100,bbox_inches='tight')

        except Exception as e:
            logging.critical('trouble saving detection result   '.format(e))
        finally:
            svrelease(svh,savevideo)

    return stat


def keyhandler(keypressed,framebyframe):
    if keypressed == -1: # no key pressed
        return (framebyframe,False)
    elif keypressed == 1048608: #space
        return (not framebyframe, False)
    elif keypressed == 1048603: #escape
        return (None, True)
    else:
        print('keypress code: ' + str(keypressed))
        return (framebyframe,False)

def savestat(stat,fn):
    assert isinstance(stat,DataFrame)
    print('saving detections & statistics to {}'.format(fn))

    with h5py.File(str(fn),'w',libver='latest') as h5:
        h5['/detect']  = stat['detect']
        h5['/mean']    = stat['mean']
        h5['/median']  = stat['median']
        h5['/variance']= stat['variance']

def getraw(dfid,ifrm,finf,svh,ap,cp,savevideo,verbose):
    """ this function reads the reference frame too--which makes sense if youre
       only reading every Nth frame from the multi-TB file instead of every frame
    """
    frameref = None #just in case not used
    dowiener = np.isfinite(cp['wienernhood'])
#%% reference frame

    if finf['reader'] == 'raw' and dfid:
        if ap['twoframe']:
            frameref = getDMCframe(dfid,ifrm,finf,verbose)[0]
            frameref = bytescale(frameref, ap['rawlim'][0], ap['rawlim'][1])
            if dowiener:
                frameref = wiener(frameref,cp['wienernhood'])

        try:
            frame16,rfi = getDMCframe(dfid,ifrm+1,finf)
            framegray = bytescale(frame16, ap['rawlim'][0], ap['rawlim'][1])
        except (ValueError,IOError):
            ap['rawframeind'] = np.delete(ap['rawframeind'], np.s_[ifrm:])
            raise

    elif finf['reader'] == 'cv2':
        if ap['twoframe']:
            retval,frameref = dfid.read()
            if not retval:
                if ifrm==0:
                    logging.error('could not read video file, sorry')
                print('done reading video.')
                return None, None, ap
            if frameref.ndim>2:
                frameref = cv2.cvtColor(frameref, cv2.COLOR_RGB2GRAY)
            if dowiener:
                frameref = wiener(frameref,cp['wienernhood'])

        retval,frame16 = dfid.read() #TODO this is skipping every other frame!
        # TODO can we use dfid.set(cv.CV_CAP_PROP_POS_FRAMES,ifrm) to set 0-based index of next frame?
        rfi = ifrm
        if not retval:
            raise IOError('could not read video from {}'.format(dfid))

        if frame16.ndim>2:
            framegray = cv2.cvtColor(frame16, cv2.COLOR_RGB2GRAY)
        else:
            framegray = frame16 #copy NOT needed
    elif finf['reader'] == 'h5fm':   #one frame per file
        if ap['twoframe']:
            frameref = getfmradarframe(dfid[ifrm])[2]
            frameref = bytescale(frameref, ap['rawlim'][0], ap['rawlim'][1])
        frame16 = getfmradarframe(dfid[ifrm+1])[2]
        rfi = ifrm
        framegray = bytescale(frame16, ap['rawlim'][0], ap['rawlim'][1])
    elif finf['reader'] == 'h5vid':
        with h5py.File(str(dfid),'r',libver='latest') as f:
            if ap['twoframe']:
                frameref = bytescale(f['/rawimg'][ifrm,...],
                                     ap['rawlim'][0], ap['rawlim'][1])
                if dowiener:
                    frameref = wiener(frameref,cp['wienernhood'])

            #keep frame16 for histogram
            frame16 = f['/rawimg'][ifrm+1,...]
        framegray = bytescale(frame16,
                                 ap['rawlim'][0], ap['rawlim'][1])
        rfi = ifrm

    elif finf['reader'] == 'fits':
        #memmap = False required thru Astropy 1.1.1 due to BZERO used...
        #with fits.open(str(dfid),mode='readonly',memmap=False) as f:
        if ap['twoframe']:
            frameref = bytescale(dfid[0].data[ifrm,...],
                                 ap['rawlim'][0], ap['rawlim'][1])
            if dowiener:
                frameref = wiener(frameref,cp['wienernhood'])

        frame16 = dfid[0].data[ifrm+1,...]
        framegray = bytescale(frame16,
                                 ap['rawlim'][0], ap['rawlim'][1])

        rfi = ifrm #TODO: incorrect raw index with sequence of fits files
    else:
        raise TypeError('unknown reader type {}'.format(finf['reader']))


#%% current frame
    ap['rawframeind'][ifrm] = rfi

    if dowiener:
        framegray = wiener(framegray,cp['wienernhood'])

    if 'raw' in pshow:
        # cv2.imshow just divides by 256, NOT autoscaled!
        # http://docs.opencv.org/modules/highgui/doc/user_interface.html
        cv2.imshow('video', framegray)
#%% plotting
    if 'rawscaled' in pshow:
        cv2.imshow('raw video, scaled to 8-bit', framegray)
    # image histograms (to help verify proper scaling to uint8)
    if 'hist' in pshow:
        figure(321).clf()
        ax=figure(321).gca()
        hist(frame16.flatten(), bins=128, fc='w',ec='k', log=True)
        ax.set_title('raw uint16 values')

        figure(322).clf()
        ax=figure(322).gca()
        hist(framegray.flatten(), bins=128, fc='w',ec='k', log=True)
        ax.set_xlim((0,255))
        ax.set_title('normalized video into opt flow')

    if svh['video'] is not None:
        if savevideo == 'tif':
            svh['video'].save(framegray,compress=complvl)
        elif savevideo == 'vid':
            svh['video'].write(framegray)

    return framegray,frameref,ap

def getvidinfo(fn,cp,up,verbose):
    print('using {} for {}'.format(cp['ofmethod'],fn))
    if verbose:
        print('minBlob={} maxBlob={} maxNblob={}'.format(cp['minblobarea'],cp['maxblobarea'],cp['maxblobcount']) )

    if fn.suffix.lower() == '.dmcdata':
        xypix=(cp['xpix'],cp['ypix'])
        xybin=(cp['xbin'],cp['ybin'])
        if up['startstop'][0] is None:
            finf = getDMCparam(fn,xypix,xybin,up['framestep'],verbose=verbose)
        else:
            finf = getDMCparam(fn,xypix,xybin,
                     (up['startstop'][0], up['startstop'][1], up['framestep']),
                      verbose=verbose)
        finf['reader']='raw'

        dfid = open(fn,'rb') #I didn't use the "with open(f) as ... " because I want to swap in other file readers per user choice

    elif fn.suffix.lower() in ('.h5','.hdf5'):
        dfid = fn
#%% determine if optical or passive radar
        with h5py.File(str(fn),'r') as f:
            try: #hst image/video file
                finf = {'reader':'h5vid'}
                finf['nframe'] = f['rawimg'].shape[0]
                finf['superx'] = f['rawimg'].shape[2]
                finf['supery'] = f['rawimg'].shape[1]
                print('HDF5 video file detected {}'.format(fn))
            except KeyError: # Haystack passive FM radar file
                finf = {'reader':'h5fm'}
                finf['nframe'] = 1 # currently the passive radar uses one file per frame
                range_km,vel_mps = getfmradarframe(fn)[:2] #assuming all frames are the same size
                finf['superx'] = range_km.size
                finf['supery'] = vel_mps.size
                print('HDF5 passive FM radar file detected {}'.format(fn))
        finf['frameind'] = np.arange(finf['nframe'],dtype=np.int64)
    elif fn.suffix.lower() in ('.fit','.fits'):
        """
        have not tried memmap=True
        with memmap=False, whole file is read in to access even single data element.
        Linux file system caching should make this speedy even with memmap=False
        --Except if using over network, then the first read can take a long time.
        http://docs.astropy.org/en/stable/io/fits/#working-with-large-files
        """
        finf = getNeoParam(fn,up['framestep'])[0]
        finf['reader']='fits'

        dfid = fits.open(str(fn),mode='readonly',memmap=False)

        #finf['frameind'] = np.arange(0,finf['nframe'],up['framestep'],dtype=np.int64)
    else: #assume video file
        #TODO start,stop,step is not yet implemented, simply uses every other frame
        print('attempting to read {} with OpenCV.'.format(fn))
        finf = {'reader':'cv2'}

        dfid = cv2.VideoCapture(fn)
        vidparam=getaviprop(dfid)
        finf['nframe'] = vidparam['nframe']
        finf['superx'] = vidparam['xpix']
        finf['supery'] = vidparam['ypix']

        finf['frameind']=np.arange(finf['nframe'],dtype=np.int64)


#%% extract analysis parameters
    ap = {'twoframe':bool(cp['twoframe']), # note this should be 1 or 0 input, not the word, because even the word 'False' will be bool()-> True!
          'ofmethod':cp['ofmethod'].lower(),
          'rawframeind': np.empty(finf['nframe'],np.int64), #int64 for very large files on Windows Python 2.7, long is not available on Python3
          'rawlim': (cp['cmin'], cp['cmax']),
          'xpix': finf['superx'], 'ypix':finf['supery'],
          'thresmode':cp['thresholdmode'].lower()}

    return finf, ap, dfid

def getcamparam(paramfn,flist):
    #uses pandas and xlrd to parse the spreadsheet parameters
    if flist[0].suffix == '.DMCdata':
        camser = getserialnum(flist)
    else:
        #FIXME add your own criteria to pick which spreadsheet paramete column to use.
        # for now I tell it to just use the first column (same criteria for all files)
        logging.info('using first column of spreadsheet only for camera parameters')
        camser = [None] * len(flist)

    camparam = read_excel(paramfn,index_col=0,header=0) #returns a nicely indexable DataFrame
    return camser, camparam
