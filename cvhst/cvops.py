import cv2
import numpy as np
try:
    from cv2 import cv #necessary for Windows, "import cv" doesn't work
except:
#TODO: openCV 3.0 has legacy code buried in opencv-extra
    pass
#
from cvutils.cv2draw import draw_flow,flow2magang,draw_hsv

from matplotlib.pyplot import draw,pause #for debug plot

def dooptflow(framegray,frameref,lastflow,uv,ifrm,jfrm,ap,cp,pl,stat,pshow):

    if ap['ofmethod'] == 'hs':
        """
        http://docs.opencv.org/modules/legacy/doc/motion_analysis.html
        """
        cvref = cv.fromarray(frameref)
        cvgray = cv.fromarray(framegray)
        #result is placed in u,v
        # matlab vision.OpticalFlow Horn-Shunck has default maxiter=10, terminate=eps, smoothness=1
        # in TrackingOF7.m I used maxiter=8, terminate=0.1, smaothness=0.1
        """
        ***************************
        Note that smoothness parameter for cv.CalcOpticalFlowHS needs to be SMALLER than matlab
        to get similar result. Useless when smoothness was 1 in python, but it's 1 in Matlab!
        *****************************
        """
        cv.CalcOpticalFlowHS(cvref, cvgray, False, uv[0], uv[1],
                             cp['hssmooth'],
                             (cv.CV_TERMCRIT_ITER | cv.CV_TERMCRIT_EPS, 8, 0.1))

        # reshape to numpy float32, xpix x ypix x 2
        flow = np.dstack((np.asarray(uv[0]), np.asarray(uv[1])))

    elif ap['ofmethod'] == 'farneback':
        """
        http://docs.opencv.org/trunk/modules/video/doc/motion_analysis_and_object_tracking.html
        """
        flow = cv2.calcOpticalFlowFarneback(frameref, framegray,
                                            flow=lastflow, #need flow= for opencv2/3 compatibility
                                           pyr_scale=0.5,
                                           levels=1,
                                           winsize=3,
                                           iterations=5,
                                           poly_n = 3,
                                           poly_sigma=1.5,
                                           flags=1)
    else: #using non-of method
        return None,None,None,None
#%% zero out edges of image (which have very high flow, unnaturally)
    '''
    maybe this can be done more elegantly, maybe via pad or take?
    http://stackoverflow.com/questions/13525266/multiple-slice-in-list-indexing-for-numpy-array
    '''
    te = cp['trimedgeof']
    flow[:te,...] = 0.; flow[-te:,...] = 0.
    flow[:,:te,:] = 0.; flow[:,-te:,:] = 0.

    flow /= 255. #make like matlab, which has normalized data input (opencv requires uint8)
#%% compute median and magnitude
    ofmag = np.hypot(flow[...,0], flow[...,1])
    stat['median'].iat[jfrm] = np.median(ofmag)#we don't know if it will be index or ut1 in index
    stat['mean'].iat[jfrm] = ofmag.mean()
    stat['variance'].iat[jfrm] = np.var(ofmag)

    if 'ofmag' in pshow:
        pl['pmed'][0].set_ydata(stat['median'].values)
        pl['pmean'][0].set_ydata(stat['mean'].values)

    if 'flowvec' in pshow:
        cv2.imshow('flow vectors', draw_flow(framegray,flow) )
    if 'flowhsv' in pshow:
        mag,ang = flow2magang(flow,np.uint8)
        cv2.imshow('flowHSV', draw_hsv(mag,ang,np.uint8) )
    if 'ofmag' in pshow:
        #cv2.imshow('flowMag', ofmag) #was only grayscale, I wanted color
        pl['iofm'].set_data(ofmag)


#    draw(); pause(0.001) #debug
    return flow,ofmag, stat

def dothres(ofmaggmm,medianflow,ap,cp,svh,pshow):
    """
    flow threshold, considering median
    """
    if medianflow is not None: #OptFlow based
        if ap['thresmode'] == 'median':
            if medianflow>1e-6:  #median is scalar
                lowthres = cp['ofthresmin'] * medianflow #median is scalar!
                hithres =  cp['ofthresmax'] * medianflow #median is scalar!
            else: #median ~ 0
                lowthres = 0
                hithres = np.inf

        elif ap['thresmode'] == 'runningmean':
            exit('*** ' + ap['thresmode'] + ' not yet implemented')
        else:
            exit('*** ' + ap['thresmode'] + ' not yet implemented')
    else:
        hithres = 255; lowthres=0 #TODO take from spreadsheed as gmmlowthres gmmhighthres
    """
    This is the oppostite of np.clip
    1) make boolean of  min < flow < max
    2) convert to uint8
    3) (0,255) since that's what cv2.imshow wants
    """
    # the logical_and, *, and & are almost exactly the same speed. The & felt the most Pythonic.
    #thres = np.logical_and(ofmaggmm < hithres, ofmaggmm > lowthres).astype(np.uint8) * 255
    thres = ((ofmaggmm<hithres) & (ofmaggmm>lowthres)).astype('uint8') * 255
    # has to be 0,255 because that's what opencv functions (imshow and computation) want


    if svh['thres'] is not None:
        if svh['save'] == 'tif':
            svh['thres'].save(thres,compress=svh['complvl'])
        elif svh['save'] == 'vid':
            svh['thres'].write(thres)

    if 'thres' in pshow:
        cv2.imshow('thresholded', thres)
    """ threshold image by lowThres < abs(OptFlow) < highThres
    the low threshold helps elimate a lot of "false" OptFlow from camera
    noise
    the high threshold helps eliminate star "twinkling," which appears to
    make very large Optical Flow magnitude

    we multiply boolean by 255 because cv2.imshow expects only values on [0,255] and does not autoscale
    """
    return thres

def dodespeck(thres,medfiltsize,svh,pshow):

    despeck = cv2.medianBlur(thres,ksize=medfiltsize)

    if svh['despeck'] is not None:
        if svh['save'] == 'tif':
            svh['despeck'].save(despeck,compress=svh['complvl'])
        elif svh['save'] == 'vid':
            svh['despeck'].write(despeck)

    if 'thres' in pshow:
        cv2.imshow('despeck', despeck)

    return despeck

def domorph(despeck,kern,svh,pshow):
    """
    http://docs.opencv.org/master/doc/py_tutorials/py_imgproc/py_morphological_ops/py_morphological_ops.html
    """
   # opened = cv2.morphologyEx(despeck, cv2.MORPH_OPEN, openkernel)
    eroded = cv2.erode(despeck,kern['erode'])
    closed = cv2.morphologyEx(eroded, cv2.MORPH_CLOSE, kern['close'])

    if svh['erode'] is not None:
        if svh['save'] == 'tif':
            svh['erode'].save(eroded,compress=svh['complvl'])
        elif svh['save'] == 'vid':
            svh['erode'].write(eroded)

    if svh['close'] is not None:
        if svh['save'] == 'tif':
            svh['close'].save(closed,compress=svh['complvl'])
        elif svh['save'] == 'vid':
            svh['close'].write(closed)

    if 'morph' in pshow:
        #cv2.imshow('opened', opened)
        cv2.imshow('morphed',closed)

    return closed

def doblob(morphed,blobdetect,framegray,ifrm,jfrm,svh,pl,stat,pshow):
    """
    http://docs.opencv.org/master/modules/features2d/doc/drawing_function_of_keypoints_and_matches.html
    http://docs.opencv.org/trunk/modules/features2d/doc/drawing_function_of_keypoints_and_matches.html
    """
#%% how many blobs
    keypoints = blobdetect.detect(morphed)
    nkey = len(keypoints)
    stat['detect'].iat[jfrm] = nkey #we don't know if it will be index or ut1 in index
#%% plot blobs
    final = framegray.copy() # is the .copy necessary?

    final = cv2.drawKeypoints(framegray, keypoints, outImage=final,
                              flags=cv2.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS)

    cv2.putText(final, text=str(nkey), org=(10,510),
                fontFace=cv2.FONT_HERSHEY_PLAIN, fontScale=5,
                color=(0,255,0), thickness=2)

    if 'final' in pshow:
        cv2.imshow('final',final)

    if svh['detect'] is not None:
        if svh['save'] == 'tif':
            svh['detect'].save(final,compress=svh['complvl'])
        elif svh['save'] =='vid':
            svh['detect'].write(final)

#%% plot detection vs. time
    if 'det' in pshow or 'savedet' in pshow: #updates plot with current info
        pl['pdet'][0].set_ydata(stat['detect'].values)

#    draw(); pause(0.001) #debug

    return stat
