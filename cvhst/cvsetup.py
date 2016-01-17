import logging
from pathlib2 import Path
import cv2
try:
    from cv2.cv import FOURCC as fourcc #Windows needs from cv2.cv
except ImportError as e:
    from cv2 import VideoWriter_fourcc as fourcc
#
from tempfile import gettempdir
import numpy as np
from matplotlib.pylab import figure
from matplotlib.colors import LogNorm
#
from cvutils.calcOptFlow import setupuv

def setupkern(ap,cp):
    if not cp['openradius'] % 2:
        raise ValueError('openRadius must be ODD')

    kern = {}
    kern['open'] = cv2.getStructuringElement(cv2.MORPH_ELLIPSE,
                                           (cp['openradius'], cp['openradius']))
    kern['erode'] = kern['open']
    kern['close'] = cv2.getStructuringElement(cv2.MORPH_RECT,
                                            (cp['closewidth'],cp['closeheight']))
    #cv2.imshow('open kernel',openkernel)
    print('open kernel');  print(kern['open'])
    print('close kernel'); print(kern['close'])
    print('erode kernel'); print(kern['erode'])

    return kern

def svsetup(savevideo,complvl,ap, cp, up,pshow):
    xpix = ap['xpix']; ypix= ap['ypix']
    dowiener = np.isfinite(cp['wienernhood'])


    tdir = Path(gettempdir())
    if savevideo:
        print('dumping video output to '+tdir)
    svh = {'video':None, 'wiener':None,'thres':None,'despeck':None,
           'erode':None,'close':None,'detect':None,'save':savevideo,'complvl':complvl}
    if savevideo == 'tif':
        #complvl = 6 #0 is uncompressed
        try:
            from tifffile import TiffWriter  #pip install tifffile
        except ImportError as e:
            logging.error('I cannot save iterated video results due to missing tifffile module \n'
                          'try   pip install tifffile \n {}'.format(e))
            return svh

        if dowiener:
            svh['wiener'] = TiffWriter(str(tdir/'wiener.tif'))
        else:
            svh['wiener'] = None

        svh['video']  = TiffWriter(str(tdir/'video.tif')) if 'rawscaled' in pshow else None
        svh['thres']  = TiffWriter(str(tdir/'thres.tif')) if 'thres' in pshow else None
        svh['despeck']= TiffWriter(str(tdir/'despk.tif')) if 'thres' in pshow else None
        svh['erode']  = TiffWriter(str(tdir/'erode.tif')) if 'morph' in pshow else None
        svh['close']  = TiffWriter(str(tdir/'close.tif')) if 'morph' in pshow else None
        # next line makes big file
        svh['detect'] = None #TiffWriter(join(tdir,'detect.tif')) if showfinal else None


    elif savevideo == 'vid':
        wfps = up['fps']
        if wfps<3:
            logging.warning('VLC media player had trouble with video slower than about 3 fps')


        """ if grayscale video, isColor=False
        http://stackoverflow.com/questions/9280653/writing-numpy-arrays-using-cv2-videowriter

        These videos are casually dumped to the temporary directory.
        """
        cc4 = fourcc(*'FFV1')
        """
        try 'MJPG' or 'XVID' if FFV1 doesn't work.
        see https://github.com/scienceopen/python-test-functions/blob/master/videowritetest.py for more info
        """
        if dowiener:
            svh['wiener'] = cv2.VideoWriter(str(tdir/'wiener.avi'),cc4, wfps,(ypix,xpix),False)
        else:
            svh['wiener'] = None

        svh['video']  = cv2.VideoWriter(str(tdir/'video.avi'), cc4,wfps, (ypix,xpix),False) if 'rawscaled' in pshow else None
        svh['thres']  = cv2.VideoWriter(str(tdir/'thres.avi'), cc4,wfps, (ypix,xpix),False) if 'thres' in pshow else None
        svh['despeck']= cv2.VideoWriter(str(tdir/'despk.avi'), cc4,wfps, (ypix,xpix),False) if 'thres' in pshow else None
        svh['erode']  = cv2.VideoWriter(str(tdir/'erode.avi'), cc4,wfps, (ypix,xpix),False) if 'morph' in pshow else None
        svh['close']  = cv2.VideoWriter(str(tdir/'close.avi'), cc4,wfps, (ypix,xpix),False) if 'morph' in pshow else None
        svh['detect'] = cv2.VideoWriter(str(tdir/'detct.avi'), cc4,wfps, (ypix,xpix),True)  if 'final' in pshow else None

        for k,v in svh.items():
            if v is not None and not v.isOpened():
                raise TypeError('trouble writing video for ' + k)

    return svh

def svrelease(svh,savevideo):
    try:
        if savevideo=='tif':
            for k,v in svh.items():
                if v is not None:
                    v.close()
        elif savevideo == 'vid':
            for k,v in svh.items():
                if v is not None:
                    v.release()
    except Exception as e:
        print(str(e))


def setupof(ap,cp):
    xpix = ap['xpix']; ypix = ap['ypix']

    umat = None; vmat = None; gmm=None; ofmed=None
    lastflow = None #if it stays None, signals to use GMM
    if ap['ofmethod'] == 'hs':
        try:
            umat,vmat = setupuv((ypix,xpix))
            lastflow = np.nan #nan instead of None to signal to use OF instead of GMM
        except NameError as e:
            raise ImportError("OpenCV 3 doesn't have legacy cv functions such as {}. You're using OpenCV {}.  Original error: {}".format(ap['ofmethod'],cv2.__version__,e))

    elif ap['ofmethod'] == 'farneback':
        lastflow = np.zeros((ypix,xpix,2))
    elif ap['ofmethod'] == 'mog':
        try:
            gmm = cv2.BackgroundSubtractorMOG(history=cp['nhistory'],
                                               nmixtures=cp['nmixtures'],)
        except AttributeError as e:
            raise ImportError('MOG is for OpenCV2 only.   ' + str(e))
    elif ap['ofmethod'] == 'mog2':
        print('* CAUTION: currently inputting the same paramters gives different'+
        ' performance between OpenCV 2 and 3. Informally OpenCV 3 works a lot better.')
        try:
            gmm = cv2.BackgroundSubtractorMOG2(history=cp['nhistory'],
                                               varThreshold=cp['varThreshold'], #default 16
#                                                nmixtures=cp['nmixtures'],
)
        except AttributeError as e:
            gmm = cv2.createBackgroundSubtractorMOG2(history=cp['nhistory'],
                                                     varThreshold=cp['varThreshold'],
                                                     detectShadows=True)
            gmm.setNMixtures(cp['nmixtures'])
            gmm.setComplexityReductionThreshold(cp['CompResThres'])
    elif ap['ofmethod'] == 'knn':
        try:
            gmm = cv2.createBackgroundSubtractorKNN(history=cp['nhistory'],
                                                    detectShadows=True)
        except AttributeError as e:
            raise ImportError('KNN is for OpenCV3 only. ' + str(e))
    elif ap['ofmethod'] == 'gmg':
        try:
            gmm = cv2.createBackgroundSubtractorGMG(initializationFrames=cp['nhistory'])
        except AttributeError as e:
            raise ImportError('GMG is for OpenCV3 only, but is currently part of opencv_contrib. ' + str(e))

    else:
        raise TypeError('unknown method {}'.format(ap['ofmethod']))

    return (umat, vmat), lastflow, ofmed, gmm


def setupfigs(finf,fn,pshow):
    hiom = None
    if 'ofmag' in pshow:
        figure(30).clf()
        figom = figure(30)
        axom = figom.gca()
        hiom = axom.imshow(np.zeros((finf['supery'],finf['superx'])),vmin=1e-4, vmax=0.1,
                           origin='bottom', norm=LogNorm())#, cmap=lcmap) #arbitrary limits
        axom.set_title('optical flow magnitude')
        figom.colorbar(hiom,ax=axom)

    hpmn = None; hpmd = None; medpl = None; meanpl = None
    if 'meanmedian' in pshow:
        medpl = np.zeros(finf['frameind'].size, dtype=float) #don't use nan, it won't plot
        meanpl = medpl.copy()
        figure(31).clf()
        figmm = figure(31)
        axmm = figmm.gca()
        axmm.set_title('mean and median optical flow')
        axmm.set_xlabel('frame index #')
        axmm.set_ylim((0,5e-4))

        hpmn = axmm.plot(meanpl, label='mean')
        hpmd = axmm.plot(medpl, label='median')
        axmm.legend(loc='best')


    detect = np.nan #in case it falls through to h5py writer
    hpdt = None; fgdt = None
    if 'det' in pshow or 'savedet' in pshow:
        detect = np.zeros(finf['frameind'].size, dtype=int)
        figure(40).clf()
        fgdt = figure(40)
        axdt = fgdt.gca()
        axdt.set_title('Detections of Aurora: {}'.format(fn))
        axdt.set_xlabel('frame index #')
        axdt.set_ylabel('number of detections')
        axdt.set_ylim((0,10))
        hpdt = axdt.plot(detect)

    return {'iofm':hiom, 'pmean':hpmn, 'pmed':hpmd, 'median':medpl, 'mean':meanpl,
            'pdet':hpdt, 'fdet':fgdt, 'detect':detect}
