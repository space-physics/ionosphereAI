import logging
import cv2
try:
    from cv2.cv import FOURCC as fourcc #Windows needs from cv2.cv
except ImportError as e:
    from cv2 import VideoWriter_fourcc as fourcc
#
from sys import stderr
from pandas import DataFrame
from datetime import datetime
from pytz import UTC
import numpy as np
from matplotlib.pylab import figure,subplots#draw,pause
from matplotlib.colors import LogNorm


def setupkern(P,up):
    openrad = P.getint('morph','openradius')
    if not openrad % 2:
        raise ValueError('openRadius must be ODD')


    up['open'] = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (openrad,openrad))

    up['erode'] = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (openrad,openrad))
    up['close'] = cv2.getStructuringElement(cv2.MORPH_RECT,
                                        (P.getint('morph','closewidth'),
                                         P.getint('morph','closeheight')))

    # cv2.imshow('open kernel',openkernel)
    print('open kernel',  up['open'])
    print('close kernel', up['close'])
    print('erode kernel', up['erode'])

    return up

def svsetup(P, up):
    savevideo = up['savevideo']
    x = up['xpix']; y = up['ypix']
    pshow = up['pshow']

    dowiener = P.get('filter','wienernhood')
    if not dowiener.strip():
        dowiener = False
    else:
        dowiener = int(dowiener)


    if savevideo:
        print(f'dumping video output to {up["odir"]}')
    svh = {'video':None, 'wiener':None,'thres':None,'despeck':None,
           'erode':None,'close':None,'detect':None,'save':savevideo,'complvl':up['complvl']}
    if savevideo == 'tif':
        #complvl = 6 #0 is uncompressed
        try:
            from tifffile import TiffWriter  #pip install tifffile
        except ImportError as e:
            logging.error('I cannot save iterated video results due to missing tifffile module \n'
                          'try   pip install tifffile \n {}'.format(e))
            return svh

        if dowiener:
            svh['wiener'] = TiffWriter(str(up['odir'] / 'wiener.tif'))
        else:
            svh['wiener'] = None

        svh['video']  = TiffWriter(str(up['odir'] / 'video.tif')) if 'rawscaled' in pshow else None
        svh['thres']  = TiffWriter(str(up['odir'] / 'thres.tif')) if 'thres' in pshow else None
        svh['despeck']= TiffWriter(str(up['odir'] / 'despk.tif')) if 'thres' in pshow else None
        svh['erode']  = TiffWriter(str(up['odir'] / 'erode.tif')) if 'morph' in pshow else None
        svh['close']  = TiffWriter(str(up['odir'] / 'close.tif')) if 'morph' in pshow else None
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
        #cc4 = fourcc(*'FFV1')
        cc4 = fourcc(*'FMP4')
        """
        try 'MJPG' 'XVID' 'FMP4' if FFV1 doesn't work.

        https://github.com/scienceopen/pyimagevideo
        """
        if dowiener:
            svh['wiener'] = cv2.VideoWriter(str(up['odir'] / 'wiener.avi'),cc4, wfps,(y, x),False)
        else:
            svh['wiener'] = None

        svh['video']  = cv2.VideoWriter(str(up['odir'] / 'video.avi'), cc4,wfps, (y, x),False) if 'rawscaled' in pshow else None
        svh['thres']  = cv2.VideoWriter(str(up['odir'] / 'thres.avi'), cc4,wfps, (y, x),False) if 'thres' in pshow else None
        svh['despeck']= cv2.VideoWriter(str(up['odir'] / 'despk.avi'), cc4,wfps, (y, x),False) if 'thres' in pshow else None
        svh['erode']  = cv2.VideoWriter(str(up['odir'] / 'erode.avi'), cc4,wfps, (y, x),False) if 'morph' in pshow else None
        svh['close']  = cv2.VideoWriter(str(up['odir'] / 'close.avi'), cc4,wfps, (y, x),False) if 'morph' in pshow else None
        svh['detect'] = cv2.VideoWriter(str(up['odir'] / 'detct.avi'), cc4,wfps, (y, x),True)  if 'final' in pshow else None

        for k,v in svh.items():
            try:
                if not v.isOpened():
                    logging.error('trouble writing video for {}'.format(k))
            except AttributeError: #not a cv2 object, duck typing
                pass

    return svh

def svrelease(svh,savevideo):
    try:
        if savevideo=='tif':
            for k,v in svh.items():
                if v is not None:
                    v.close()
        elif savevideo == 'vid':
            for k,v in svh.items():
                try:
                    v.release()
                except AttributeError:
                    pass
    except Exception as e:
        print(str(e))


def setupof(ap,cp):
    xpix = ap['xpix']; ypix = ap['ypix']

    gmm=None
    lastflow = None #if it stays None, signals to use GMM
    if ap['ofmethod'] == 'hs':
        pass
    elif ap['ofmethod'] == 'farneback':
        lastflow = np.zeros((ypix,xpix,2))
    elif ap['ofmethod'] == 'mog':
        try:
            gmm = cv2.BackgroundSubtractorMOG(history=cp['nhistory'],
                                               nmixtures=cp['nmixtures'],)
        except AttributeError as e:
            raise ImportError(f'MOG is for OpenCV2 only.   {e}')
    elif ap['ofmethod'] == 'mog2':
        print('* CAUTION: currently inputting the same paramters gives different'+
        ' performance between OpenCV 2 and 3. Informally OpenCV 3 works a lot better.',file=stderr)
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
            raise ImportError(f'KNN is for OpenCV3 only. {e}')
    elif ap['ofmethod'] == 'gmg':
        try:
            gmm = cv2.createBackgroundSubtractorGMG(initializationFrames=cp['nhistory'])
        except AttributeError as e:
            raise ImportError(f'GMG is for OpenCV3 only, but is currently part of opencv_contrib. {e}')

    else:
        raise TypeError('unknown method {}'.format(ap['ofmethod']))

    return lastflow,gmm


def setupfigs(finf, fn, up):
# %% optical flow magnitude plot

    if 'thres' in up['pshow']:
        fg = figure()
        axom = fg.gca()
        hiom = axom.imshow(np.zeros((finf['supery'],finf['superx'])),
                           vmin=1e-5, vmax=0.1,  # arbitrary limits
                           origin='top',  # origin=top like OpenCV
                           norm=LogNorm())  # cmap=lcmap)
        axom.set_title(f'optical flow magnitude{fn}')
        fg.colorbar(hiom,ax=axom)
    else:
        hiom = None

# %% stat plot
    try:
        dt = [datetime.fromtimestamp(t,tz=UTC) for t in finf['ut1'][:-1]]
        ut = finf['ut1'][:-1]
    except KeyError:
        dt = ut = None

    stat = DataFrame(index=ut,columns=['mean','median','variance','detect'])
    stat['detect'] = np.zeros(finf['frameind'].size-1, dtype=int)
    stat[['mean','median','variance']] = np.zeros((finf['frameind'].size-1,3),dtype=float)

    hpmn, hpmd, hpdt, fgdt= statplot(dt,stat,fn,up['pshow'])

#    draw(); pause(0.001) #catch any plot bugs

    up['iofm']  = hiom
    up['pmean'] = hpmn
    up['pmed']  = hpmd
    up['pdet']  = hpdt
    up['fdet']  = fgdt

    return up,stat

def statplot(dt,stat,fn=None,pshow='stat'):



    def _timelbl(ax,x,y,lbl=None):
        if x is not None:
            hpl = ax.plot(x,y,label=lbl)
            ax.set_xlabel('Time [UTC]')
        else:
            hpl = ax.plot(stat.index,y,label=lbl)
            ax.set_xlabel('frame index #')
        return hpl

    if 'stat' in pshow:
        fgdt,axs = subplots(1,2,figsize=(12,5))
        ax = axs[0]
        ax.set_title(f'optical flow statistics{fn}')
        ax.set_xlabel('frame index #')
        ax.set_ylim((0,5e-3))

        hpmn = _timelbl(ax, dt, stat['mean'],  'mean')
        hpmd = _timelbl(ax, dt, stat['median'], 'median')
        ax.legend(loc='best')
#%% detections
        ax = axs[1]
        ax.set_title(f'Detections of Aurora:{fn}')
        ax.set_ylabel('number of detections')
        ax.set_ylim((0,10))

        hpdt = _timelbl(ax, dt, stat['detect'])
    else:
        hpmn = None; hpmd = None; hpdt = None; fgdt = None

    return hpmn, hpmd, hpdt, fgdt
