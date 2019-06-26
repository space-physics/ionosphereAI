#!/usr/bin/env python
import numpy as np
import pandas
import logging
from typing import Dict, Any, Tuple

try:
    import cv2
except ImportError:
    cv2 = None

try:
    from pyoptflow import HornSchunck
except ImportError:
    HornSchunck = None

try:
    from morecvutils.cv2draw import draw_flow, flow2magang, draw_hsv
except ImportError:
    draw_flow = flow2magang = draw_hsv = None
# from matplotlib.pyplot import draw,pause #for debug plot


def dooptflow(Inew: np.ndarray,
              Iref: np.ndarray,
              lastflow: np.ndarray,
              jfrm: int,
              up: Dict[str, Any],
              P,
              stat: pandas.DataFrame) -> Tuple[np.ndarray, np.ndarray, pandas.DataFrame]:

    assert Inew.ndim == Iref.ndim == 2, 'these are 2-D data'
    assert lastflow.shape[:2] == Inew.shape == Iref.shape, 'these are image-like data'
    assert lastflow.ndim == 3 and lastflow.shape[2] == 2, 'u, v motion data'

    if up['ofmethod'] == 'hs':
        if HornSchunck is None:
            raise ImportError('pip install pyotflow')
        u, v = HornSchunck(Iref, Inew, P.getfloat('main', 'hssmooth'), P.getint('main', 'hsiter'))
        flow = np.dstack((u, v))  # this is how OpenCV expects it.
    elif up['ofmethod'] == 'farneback':
        """
        http://docs.opencv.org/trunk/modules/video/doc/motion_analysis_and_object_tracking.html

        flow.shape == (x,y,2) (last dimension is u, v flow estimate)
        """
        flow = cv2.calcOpticalFlowFarneback(Iref, Inew,
                                            flow=lastflow,  # need flow= for opencv2/3 compatibility
                                            pyr_scale=0.5,
                                            levels=1,
                                            winsize=3,
                                            iterations=5,
                                            poly_n=3,
                                            poly_sigma=1.5,
                                            flags=1)
    else:  # using non-of method
        return ([], [], {})
# %% Normalize [0, 1]
    """
    in reader.py:getraw() we scale to uint8 for OpenCV.
    FIXME is this necessary for anything besides cv2.imshow() ?
    """
    flow /= 255.
# %% zero out edges of image (which have very high flow, unnaturally)
    '''
    maybe this can be done more elegantly, maybe via pad or take?
    http://stackoverflow.com/questions/13525266/multiple-slice-in-list-indexing-for-numpy-array
    '''
    te = P.getint('filter', 'trimedgeof')
    flow[:te, ...] = 0.
    flow[-te:, ...] = 0.
    flow[:, :te, :] = 0.
    flow[:, -te:, :] = 0.
# %% compute median and magnitude
    ofmag = np.hypot(flow[..., 0], flow[..., 1])
    stat['median'].iat[jfrm] = np.median(ofmag)  # we don't know if it will be index or ut1 in index
    stat['mean'].iat[jfrm] = ofmag.mean()
    stat['variance'].iat[jfrm] = np.var(ofmag)

    try:
        up['pmed'][0].set_ydata(stat['median'].values)
        up['pmean'][0].set_ydata(stat['mean'].values)
    except TypeError:  # if None
        pass

    if 'thres' in up['pshow']:
        # cv2.imshow('flowMag', ofmag) #was only grayscale, I wanted color
        up['iofm'].set_data(ofmag)

    if 'flowvec' in up['pshow']:
        if draw_flow is None or cv2 is None:
            raise ImportError('pip install morecvutils')
        cv2.imshow('flow vectors', draw_flow(Inew, flow))
    if 'flowhsv' in up['pshow']:
        if flow2magang is None or draw_hsv is None or cv2 is None:
            raise ImportError('pip install morecvutils')
        mag, ang = flow2magang(flow, np.uint8)
        cv2.imshow('flowHSV', draw_hsv(mag, ang, np.uint8))

#    draw(); pause(0.001) #debug
    return flow, ofmag, stat


def dothres(ofmaggmm: np.ndarray,
            medianflow: float,
            P,
            i: int,
            svh: Dict[str, Any],
            up: Dict[str, Any],
            isgmm: bool) -> np.ndarray:
    """
    flow threshold, considering median

    Results
    -------
    thres: np.ndarray of uint8
        thresholded image shape x,y
    """
    if not isgmm:  # OptFlow based
        if up['thresmode'] == 'median':
            #            if medianflow>1e-6:  #median is scalar
            lowthres = P.getfloat('blob', 'ofthresmin') * medianflow  # median is scalar!
            hithres = P.getfloat('blob', 'ofthresmax') * medianflow  # median is scalar!
            logging.debug(f'Low, high threshold for optical flow {lowthres}, {hithres}')
#            else: #median ~ 0
#                lowthres = 0
#                hithres = np.inf

        elif up['thresmode'] == 'runningmean':
            raise NotImplementedError(f'{up["thresmode"]} not yet implemented')
        else:
            raise NotImplementedError(f'{up["thresmode"]} not yet implemented')

        thres = ((ofmaggmm < hithres) & (ofmaggmm > lowthres)).astype(np.uint8) * 255
    else:
        """
        0: background, 127: shadow, 255: foreground
        """
        thres = (255 * (ofmaggmm == 255)).astype(np.uint8)

    """
    This is the opposite of np.clip
    1) make boolean of  min < flow < max
    2) convert to uint8
    3) (0,255) since that's what cv2.imshow wants

    the logical_and, *, and & are almost exactly the same speed.
    "&" felt the most Pythonic.
     has to be 0,255 because that's what opencv functions (imshow and computation) want
    """

    if svh.get('thres'):
        if svh['save'] == 'tif':
            svh['thres'].save(thres, compress=svh['complvl'])
        elif svh['save'] == 'vid':
            svh['thres'].write(thres)

    if 'thres' in up['pshow']:
        cvtxt(str(i), thres)
        cv2.imshow('thresholded', thres)
    """ threshold image by lowThres < abs(OptFlow) < highThres
    the low threshold helps elimate a lot of "false" OptFlow from camera
    noise
    the high threshold helps eliminate star "twinkling," which appears to
    make very large Optical Flow magnitude

    we multiply boolean by 255 because cv2.imshow expects only values on [0,255] and does not autoscale
    """
    return thres


def dodespeck(mot: np.ndarray,
              medfiltsize: int,
              i: int,
              svh: Dict[str, Any],
              up: Dict[str, Any]) -> np.ndarray:
    """
    Despeckling algorithm

    Parameters
    ----------
    mot: np.ndarray of uint8
        motion boolean data
        mot is shape of image: x,y
        mot is supposed to be boolean, but OpenCV boolean is {0,255} instead of {0,1}

    Results
    -------
    despeck: np.ndarray of uint8
        despeckled data
    """
    despeck = cv2.medianBlur(mot, ksize=medfiltsize)
# %%
    if svh.get('despeck'):
        if svh.get('save') == 'tif':
            svh['despeck'].save(despeck, compress=svh['complvl'])
        elif svh.get('save') == 'vid':
            svh['despeck'].write(despeck)

    if 'thres' in up['pshow']:
        cvtxt(str(i), despeck)
        cv2.imshow('despeck', despeck)

    return despeck


def domorph(mot: np.ndarray,
            svh: Dict[str, Any],
            up: Dict[str, Any]) -> np.ndarray:
    """
    Morphological operations

    Parameters
    ----------
    mot: np.ndarray of uint8
        motion boolean data
        mot is shape of image: x,y
        mot is supposed to be boolean, but OpenCV boolean is {0,255} instead of {0,1}

    Results
    -------
    closed: np.ndarray of uint8
        data after morphological processing

    http://docs.opencv.org/master/doc/py_tutorials/py_imgproc/py_morphological_ops/py_morphological_ops.html
    """
    # opened = cv2.morphologyEx(despeck, cv2.MORPH_OPEN, openkernel)
    eroded = cv2.erode(mot, up['erode'])
    closed = cv2.morphologyEx(eroded, cv2.MORPH_CLOSE, up['close'])

    if svh.get('erode'):
        if svh['save'] == 'tif':
            svh['erode'].save(eroded, compress=svh['complvl'])
        elif svh['save'] == 'vid':
            svh['erode'].write(eroded)

    if svh.get('close'):
        if svh['save'] == 'tif':
            svh['close'].save(closed, compress=svh['complvl'])
        elif svh['save'] == 'vid':
            svh['close'].write(closed)

    if 'morph' in up['pshow']:
        # cv2.imshow('opened', opened)
        cv2.imshow('morphed', closed)

    return closed


def doblob(mot: np.ndarray,
           blobdetect,
           framegray: np.ndarray,
           i: int,
           svh: Dict[str, Any],
           stat: pandas.DataFrame,
           U: Dict[str, Any]) -> pandas.DataFrame:
    """
    http://docs.opencv.org/master/modules/features2d/doc/drawing_function_of_keypoints_and_matches.html
    http://docs.opencv.org/trunk/modules/features2d/doc/drawing_function_of_keypoints_and_matches.html
    """
# %% how many blobs
    keypoints = blobdetect.detect(mot)
    nkey = len(keypoints)
    stat['detect'].iat[i] = nkey  # we don't know if it will be index or ut1 in index
# %% plot blobs
    final = framegray.copy()  # is the .copy necessary?

    final = cv2.drawKeypoints(framegray, keypoints, outImage=final,
                              flags=cv2.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS)

    cv2.putText(final, text=str(nkey), org=(10, 510),
                fontFace=cv2.FONT_HERSHEY_PLAIN, fontScale=5,
                color=(0, 255, 0), thickness=2)

    if 'final' in U['pshow']:
        cvtxt(str(i), final)
        cv2.imshow('final', final)

    if svh.get('detect'):
        if svh['save'] == 'tif':
            svh['detect'].save(final, compress=svh['complvl'])
        elif svh['save'] == 'vid':
            svh['detect'].write(final)

# %% plot detection vs. time
#    if 'savedet' in pshow: #updates plot with current info
    try:
        U['pdet'][0].set_ydata(stat['detect'].values)
    except TypeError:
        pass

    return stat


def cvtxt(txt: str,
          img: np.ndarray):
    """
    Superimpose text on cv2 imshow
    """
    cv2.putText(img, text=txt, org=(5, 20),
                fontFace=cv2.FONT_HERSHEY_PLAIN, fontScale=2,
                color=(0, 255, 0), thickness=1)
