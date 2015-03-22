#!/usr/bin/python2
"""we temporarily use python 2.7 until OpenCV 3 is out of beta (will work with Python 3)
Michael Hirsch Dec 2014
This program detects aurora in multi-terabyte raw video data files
It is a major cleanup of the processDrive.sh, filelooper.m, TrackingOF7.m Frankenprograms

0) recursively find all .DMCdata files under requested root directory
1)
"""
from __future__ import division, print_function
try:
    import cv2
except ImportError as e:
    print('*** This program requires OpenCV2 or OpenCV3 installed into your Python')
    exit(str(e))
try:
    from cv2 import cv #necessary for Windows, "import cv" doesn't work
    from cv import FOURCC as fourcc
    from cv2 import SimpleBlobDetector as SimpleBlobDetector
except ImportError:
    from cv2 import VideoWriter_fourcc as fourcc
    from cv2 import SimpleBlobDetector_create as SimpleBlobDetector
    #print('legacy OpenCV functions not available, Horn-Schunck method not available')
    #print('legacy OpenCV functions are available in OpenCV2, but not OpenCV3.')
print('OpenCV '+str(cv2.__version__))
from re import search
from pandas import read_excel
from os.path import join,isfile, splitext
import numpy as np
from scipy.signal import wiener
from scipy.misc import bytescale
import sys
from time import time
import h5py
from tempfile import gettempdir
#
sys.path.append('../hist-utils')
from walktree import walktree
from rawDMCreader import getDMCparam,getDMCframe

#plot disable
showraw=False #often not useful due to no autoscale
showrawscaled=False      #True  #why not just showfinal
showhist=False
showflowvec = False
showflowhsv = False
showthres = True      #True
showofmag = False
showmeanmedian = False
showmorph = False      #True
showfinal = True
plotdet = False
savedet = False
cmplvl = 4 #tradeoff b/w speed and filesize for TIFF

#only import matplotlib if needed to save time
if savedet or plotdet or showhist or showofmag or showmeanmedian:
    from matplotlib.pylab import draw, pause, figure, hist
    from matplotlib.colors import LogNorm
    #from matplotlib.cm import get_cmap


def main(flist, up, savevideo, framebyframe, verbose):

    camser,camparam = getcamparam(up['paramfn'])

    for f,s in zip(flist,camser): #iterate over files in list
        tic = time()
        #setup output file
        stem,ext = splitext(f)
        detfn = join(up['outdir'],f +'_detections.h5')
        if isfile(detfn):
            print('** overwriting existing ' + detfn)

        cp = camparam[s] #pick the parameters for this camara from pandas DataFrame

        finf,ap = getvidinfo(f,cp,up,verbose)  # UPDATE with more general function for radar/camera
#%% setup optional video/tiff writing (mainly for debugging or publication)
        svh = svsetup(savevideo, ap, cp, up)
#%% setup blob
        blobdetect = setupblob(cp['minblobarea'], cp['maxblobarea'], cp['minblobdist'])
#%% cv opt. flow matrix setup
        uv,lastflow, ofmed, gmm = setupof(ap,cp)
#%% kernel setup
        kern = setupkern(ap,cp)
#%% mag plots setup
        pl = setupfigs(finf,f)
#%% open this file and start loops
        if ext == '.DMCdata':
            dfid = open(f,'rb')
        elif ext == '.h5':
            dfid = h5py.File(f,'r',libver='latest')
        else:
            exit('*** I didnt understand file extension ' + ext)

        for ifrm in finf['frameind'][:-1]:
#%% load and filter
            framegray,frameref,ap = getraw(dfid,ifrm,finf,svh,ap,cp,savevideo,verbose)
            if framegray is None: break
#%% compute optical flow or Background/Foreground
            if lastflow is not None: #very fast way to check mode
                flow,ofmaggmm,ofmed,pl = dooptflow(framegray,frameref,lastflow,uv,ifrm, ap,cp,pl)
                lastflow = flow.copy() #I didn't check if the .copy() is strictly necessary
            else: #background/foreground
                ofmaggmm = gmm.apply(framegray)
#%% threshold
            thres = dothres(ofmaggmm, ofmed, ap,cp,svh)
#%% despeckle
            despeck = dodespeck(thres,cp['medfiltsize'],svh)
#%% morphological ops
            morphed = domorph(despeck,kern,svh)
#%% blob detection
            final = doblob(morphed,blobdetect,framegray,ifrm,svh,pl,savevideo)
#%% plotting in loop
            """
            http://docs.opencv.org/modules/highgui/doc/user_interface.html
            """

            if plotdet or showhist or showofmag or showmeanmedian:
                draw(); pause(0.001)

            if not ifrm % 20:
                print('frame {:0d}'.format(ifrm))
                if (framegray == 255).sum() > 40: #arbitrarily allowing up to 40 pixels to be saturated at 255, to allow for bright stars and faint aurora
                    print('* Warning: video may be saturated at 255, missed detections can result')
                if (framegray == 0).sum() > 4:
                    print('* Warning: video may be saturated at 0, missed detections can result')

            if framebyframe: #wait indefinitely for spacebar press
                keypressed = cv2.waitKey(0)
                framebyframe,dobreak = keyhandler(keypressed,framebyframe)
            else:
                keypressed = cv2.waitKey(1)
                framebyframe, dobreak = keyhandler(keypressed,framebyframe)
            if dobreak:
                break
#%% done looping this file
        dfid.close()
        print('{:0.1f}'.format(time()-tic) + ' seconds to process ' + f)
        if savedet:
            detfn = stem + '_det.h5'
            detpltfn = stem + '_det.png'
            print('saving detections to ' + detfn)
            with h5py.File(detfn,libver='latest') as h5fid:
                h5fid.create_dataset("/det", data=pl['detect'])
            print('saving detection plot to ' + detpltfn)
            pl['fdet'].savefig(detpltfn,dpi=100,bbox_inches='tight')

        svrelease(svh,savevideo)

        return final

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

def setupkern(ap,cp):
    if not cp['openradius'] % 2:
        exit('*** detectaurora: openRadius must be ODD')

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

def svsetup(savevideo,ap, cp, up):
    xpix = ap['xpix']; ypix= ap['ypix']
    dowiener = np.isfinite(cp['wienernhood'])


    tdir = gettempdir()
    svh = {}
    if savevideo == 'tif':
        #complvl = 6 #0 is uncompressed
        from tifffile import TiffWriter  #pip install tifffile
        if dowiener:
            svh['wiener'] = TiffWriter(join(tdir,'wiener.tif'))
        else:
            svh['wiener'] = None

        svh['video']  = TiffWriter(join(tdir,'video.tif')) if showrawscaled else None
        svh['thres']  = TiffWriter(join(tdir,'thres.tif')) if showthres else None
        svh['despeck']= TiffWriter(join(tdir,'despk.tif')) if showthres else None
        svh['erode']  = TiffWriter(join(tdir,'erode.tif')) if showmorph else None
        svh['close']  = TiffWriter(join(tdir,'close.tif')) if showmorph else None
        # next line makes big file
        svh['detect'] = None #TiffWriter(join(tdir,'detect.tif')) if showfinal else None


    elif savevideo == 'vid':
        wfps = up['fps']
        if wfps<3:
            print('* note: VLC media player had trouble with video slower than about 3 fps')


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
            svh['wiener'] = cv2.VideoWriter(join(tdir,'wiener.avi'),cc4, wfps,(ypix,xpix),False)
        else:
            svh['wiener'] = None

        svh['video']  = cv2.VideoWriter(join(tdir,'video.avi'), cc4,wfps, (ypix,xpix),False) if showrawscaled else None
        svh['thres']  = cv2.VideoWriter(join(tdir,'thres.avi'), cc4,wfps, (ypix,xpix),False) if showthres else None
        svh['despeck']= cv2.VideoWriter(join(tdir,'despk.avi'), cc4,wfps, (ypix,xpix),False) if showthres else None
        svh['erode']  = cv2.VideoWriter(join(tdir,'erode.avi'), cc4,wfps, (ypix,xpix),False) if showmorph else None
        svh['close']  = cv2.VideoWriter(join(tdir,'close.avi'), cc4,wfps, (ypix,xpix),False) if showmorph else None
        svh['detect'] = cv2.VideoWriter(join(tdir,'detct.avi'), cc4,wfps, (ypix,xpix),True) if showfinal else None

        for k,v in svh.items():
            if v is not None and not v.isOpened():
                exit('*** trouble writing video for ' + k)
    else:
        svh = {'video':None, 'wiener':None,'thres':None,'despeck':None,'erode':None,'close':None,'detect':None}

    return svh

def svrelease(svh,savevideo):
    if savevideo=='tif':
        for k,v in svh.items():
            if v is not None:
                v.close()
    elif savevideo == 'vid':
        for k,v in svh.items():
            if v is not None:
                v.release()


def setupof(ap,cp):
    xpix = ap['xpix']; ypix = ap['ypix']

    umat = None; vmat = None; gmm=None; ofmed=None
    lastflow = None #if it stays None, signals to use GMM
    if ap['ofmethod'] == 'hs':
        try:
            umat =   cv.CreateMat(ypix, xpix, cv.CV_32FC1)
            vmat =   cv.CreateMat(ypix, xpix, cv.CV_32FC1)
            lastflow = np.nan #nan instead of None to signal to use OF instead of GMM
        except NameError as e:
            print("*** OpenCV 3 doesnt have legacy cv functions. You're using OpenCV " +str(cv2.__version__))
            exit(str(e))
    elif ap['ofmethod'] == 'farneback':
        lastflow = np.zeros((ypix,xpix,2))
    elif ap['ofmethod'] == 'mog':
        try:
            gmm = cv2.BackgroundSubtractorMOG(history=cp['nhistory'],
                                               nmixtures=cp['nmixtures'],)
        except AttributeError as e:
            exit('*** MOG is for OpenCV2 only.   ' + str(e))
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
            exit('KNN is for OpenCV3 only. ' + str(e))
    elif ap['ofmethod'] == 'gmg':
        try:
            gmm = cv2.createBackgroundSubtractorGMG(initializationFrames=cp['nhistory'])
        except AttributeError as e:
            exit('GMG is for OpenCV3 only, but is currently part of opencv_contrib. ' + str(e))

    else:
        exit('*** unknown method ' + ap['ofmethod'])

    return (umat, vmat), lastflow, ofmed, gmm


def setupblob(minblobarea, maxblobarea, minblobdist):
    blobparam = cv2.SimpleBlobDetector_Params()
    blobparam.filterByArea = True
    blobparam.filterByColor = False
    blobparam.filterByCircularity = False
    blobparam.filterByInertia = False
    blobparam.filterByConvexity = False

    blobparam.minDistBetweenBlobs = minblobdist
    blobparam.minArea = minblobarea
    blobparam.maxArea = maxblobarea
    #blobparam.minThreshold = 40 #we have already made a binary image
    return SimpleBlobDetector(blobparam)

def getraw(dfid,ifrm,finf,svh,ap,cp,savevideo,verbose):
    dowiener = np.isfinite(cp['wienernhood'])
#%% reference frame
    if ap['twoframe']:
        frameref = getDMCframe(dfid,ifrm,finf,verbose)[0]

        if dowiener:
            frameref = wiener(frameref,cp['wienernhood'])
        frameref = bytescale(frameref, ap['rawlim'][0], ap['rawlim'][1])#sixteen2eight(frameref, rawlim)
#%% current frame
    frame16,rfi = getDMCframe(dfid,ifrm+1,finf)
    if frame16 is None or rfi is None:
        ap['rawframeind'] = np.delete(ap['rawframeind'], np.s_[ifrm:])
        return None, None, ap

    framegray = frame16.copy() # keeping frame16 as 16-bit for analysis plots
    ap['rawframeind'][ifrm] = rfi

    if dowiener:
        framegray = wiener(framegray,cp['wienernhood'])

    if showraw:
        # cv2.imshow just divides by 256, NOT autoscaled!
        # http://docs.opencv.org/modules/highgui/doc/user_interface.html
        cv2.imshow('video', framegray)
    framegray = bytescale(framegray, ap['rawlim'][0], ap['rawlim'][1]) #sixteen2eight(framegray, rawlim)
#%% plotting
    if showrawscaled:
        cv2.imshow('raw video, scaled to 8-bit', framegray)
    # image histograms (to help verify proper scaling to uint8)
    if showhist:
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
            svh['video'].save(framegray,compress=cmplvl)
        elif savevideo == 'vid':
            svh['video'].write(framegray)

    return framegray,frameref,ap

def dooptflow(framegray,frameref,lastflow,uv,ifrm,ap,cp,pl):

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
    ofmed = np.median(ofmag)
    if showofmag:
        pl['median'][ifrm] = ofmed
        pl['mean'][ifrm] = ofmag.mean()
        pl['pmed'][0].set_ydata(pl['median'])
        pl['pmean'][0].set_ydata(pl['mean'])

    if showflowvec:
        cv2.imshow('flow vectors', draw_flow(framegray,flow) )
    if showflowhsv:
        cv2.imshow('flowHSV', draw_hsv(flow) )
    if showofmag:
        #cv2.imshow('flowMag', ofmag) #was only grayscale, I wanted color
        pl['iofm'].set_data(ofmag)

    return flow,ofmag, ofmed, pl

def dothres(ofmaggmm,medianflow,ap,cp,svh):
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
        if savevideo == 'tif':
            svh['thres'].save(thres,compress=cmplvl)
        elif savevideo == 'vid':
            svh['thres'].write(thres)

    if showthres:
        cv2.imshow('thresholded', thres)
    """ threshold image by lowThres < abs(OptFlow) < highThres
    the low threshold helps elimate a lot of "false" OptFlow from camera
    noise
    the high threshold helps eliminate star "twinkling," which appears to
    make very large Optical Flow magnitude

    we multiply boolean by 255 because cv2.imshow expects only values on [0,255] and does not autoscale
    """
    return thres

def dodespeck(thres,medfiltsize,svh):

    despeck = cv2.medianBlur(thres,ksize=medfiltsize)

    if svh['despeck'] is not None:
        if savevideo == 'tif':
            svh['despeck'].save(despeck,compress=cmplvl)
        elif savevideo == 'vid':
            svh['despeck'].write(despeck)

    if showthres:
        cv2.imshow('despeck', despeck)

    return despeck

def domorph(despeck,kern,svh):
    """
    http://docs.opencv.org/master/doc/py_tutorials/py_imgproc/py_morphological_ops/py_morphological_ops.html
    """
   # opened = cv2.morphologyEx(despeck, cv2.MORPH_OPEN, openkernel)
    eroded = cv2.erode(despeck,kern['erode'])
    closed = cv2.morphologyEx(eroded, cv2.MORPH_CLOSE, kern['close'])

    if svh['erode'] is not None:
        if savevideo == 'tif':
            svh['erode'].save(eroded,compress=cmplvl)
        elif savevideo == 'vid':
            svh['erod'].write(eroded)

    if svh['close'] is not None:
        if savevideo == 'tif':
            svh['close'].save(closed,compress=cmplvl)
        elif savevideo == 'vid':
            svh['close'].write(closed)

    if showmorph:
        #cv2.imshow('opened', opened)
        cv2.imshow('morphed',closed)

    return closed

def doblob(morphed,blobdetect,framegray,ifrm,svh,pl,savevideo):
    """
    http://docs.opencv.org/master/modules/features2d/doc/drawing_function_of_keypoints_and_matches.html
    http://docs.opencv.org/trunk/modules/features2d/doc/drawing_function_of_keypoints_and_matches.html
    """
    keypoints = blobdetect.detect(morphed)
    nkey = len(keypoints)
    final = framegray.copy() # is the .copy necessary?

    final = cv2.drawKeypoints(framegray, keypoints, outImage=final,
                              flags=cv2.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS)

    cv2.putText(final, text=str(nkey), org=(10,510),
                fontFace=cv2.FONT_HERSHEY_PLAIN, fontScale=5,
                color=(0,255,0), thickness=2)

    if showfinal:
        cv2.imshow('final',final)

    if svh['detect'] is not None:
        if savevideo == 'tif':
            svh['detect'].save(final,compress=cmplvl)
        elif savevideo =='vid':
            svh['detect'].write(final)

    if plotdet or savedet:
        pl['detect'][ifrm] = nkey
        pl['pdet'][0].set_ydata(pl['detect'])

def setupfigs(finf,fn):


    hiom = None
    if showofmag:
        figure(30).clf()
        figom = figure(30)
        axom = figom.gca()
        hiom = axom.imshow(np.zeros((finf['supery'],finf['superx'])),vmin=1e-4, vmax=0.1,
                           origin='bottom', norm=LogNorm())#, cmap=lcmap) #arbitrary limits
        axom.set_title('optical flow magnitude')
        figom.colorbar(hiom,ax=axom)

    hpmn = None; hpmd = None; medpl = None; meanpl = None
    if showmeanmedian:
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


    detect = None; hpdt = None; fgdt = None
    if plotdet or savedet:
        detect = np.zeros(finf['frameind'].size, dtype=int)
        figure(40).clf()
        fgdt = figure(40)
        axdt = fgdt.gca()
        axdt.set_title('Detections of Aurora: ' + fn,fontsize =10)
        axdt.set_xlabel('frame index #')
        axdt.set_ylabel('number of detections')
        axdt.set_ylim((0,10))
        hpdt = axdt.plot(detect)

    return {'iofm':hiom, 'pmean':hpmn, 'pmed':hpmd, 'median':medpl, 'mean':meanpl,
            'pdet':hpdt, 'fdet':fgdt, 'detect':detect}

def draw_flow(img, flow, step=16):
    """
    this came from opencv/examples directory
    another way: http://docs.opencv.org/trunk/doc/py_tutorials/py_gui/py_drawing_functions/py_drawing_functions.html
    """
    #scaleFact = 1. #arbitary factor to make flow visible
    canno = (0, 65535, 0)  # 65535 since it's 16-bit images
    h, w = img.shape[:2]
    y, x = np.mgrid[step//2:h:step, step//2:w:step].reshape(2,-1)
    fx, fy =  flow[y,x].T
    #create line endpoints
    lines = np.vstack([x, y, (x+fx), (y+fy)]).T.reshape(-1, 2, 2)
    lines = np.int32(lines + 0.5)
    #create image and draw
    vis = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)

    cv2.polylines(vis, lines, isClosed=False, color=canno, thickness=1, lineType=8)
    #set_trace()
    #draw filled green circles
    for (x1, y1), (x2, y2) in lines:
        cv2.circle(vis, center=(x1, y1), radius=1, color=canno, thickness=-1)
    return vis

def draw_hsv(flow):
    scaleFact = 10 #arbitary factor to make flow visible
    h, w = flow.shape[:2]
    fx, fy = scaleFact*flow[:,:,0], scaleFact*flow[:,:,1]
    ang = np.arctan2(fy, fx) + np.pi
    v = np.sqrt(fx*fx+fy*fy)
    hsv = np.zeros((h, w, 3), np.uint8)
    hsv[...,0] = ang*(180/np.pi/2)
    hsv[...,1] = 255
    hsv[...,2] = np.minimum(v*4, 255)
    bgr = cv2.cvtColor(hsv, cv2.COLOR_HSV2BGR)
    return bgr

def getvidinfo(fn,cp,up,verbose):
    print('using ' + cp['ofmethod'] + ' for ' + fn)
    if verbose:
        print('minBlob='+str(cp['minblobarea']) + ' maxBlob='+
          str(cp['maxblobarea']) + ' maxNblob=' + str(cp['maxblobcount']) )

    xypix=(cp['xpix'],cp['ypix'])
    xybin=(cp['xbin'],cp['ybin'])
    if up['startstop'][0] is None:
        finf = getDMCparam(fn,xypix,xybin,up['framestep'],verbose)
    else:
        finf = getDMCparam(fn,xypix,xybin,
                           (up['startstop'][0], up['startstop'][1], up['framestep']))
#%% extract analysis parameters
    ap = {'twoframe':bool(cp['twoframe']), # note this should be 1 or 0 input, not the word, because even the word 'False' will be bool()-> True!
          'ofmethod':cp['ofmethod'].lower(),
          'rawframeind': np.empty(finf['nframe'],dtype='int64'), #int64 for very large files on Windows Python 2.7, long is not available on Python3
          'rawlim': (cp['cmin'], cp['cmax']),
          'xpix': finf['superx'], 'ypix':finf['supery'],
          'thresmode':cp['thresholdmode'].lower()}



    return finf, ap

def getserialnum(flist):
    sn = []
    for f in flist:
        sn.append(int(search(r'(?<=CamSer)\d{3,6}',f).group()))
    return sn

def getcamparam(paramfn):
    #uses pandas and xlrd to parse the spreadsheet parameters
    camser = getserialnum(flist)
    camparam = read_excel(paramfn,index_col=0,header=0)
    return camser, camparam

if __name__=='__main__':
    from argparse import ArgumentParser
    p = ArgumentParser(description='detects aurora in raw video files')
    p.add_argument('indir',help='specify file, OR top directory over which to recursively find video files',type=str)
    p.add_argument('vidext',help='extension of raw video file',nargs='?',type=str,default='DMCdata')
    p.add_argument('--fps',help='output file FPS (note VLC needs fps>=3)',type=float,default=3)
    p.add_argument('-p','--framebyframe',help='space bar toggles play/pause', action='store_true')
    p.add_argument('-s','--savevideo',help='save video at each step (can make enormous files)',action='store_true')
    p.add_argument('-t','--savetiff',help='save tiff at each step (can make enormous files)',action='store_true')
    p.add_argument('-k','--step',help='frame step skip increment (default 10000)',type=int,default=1)
    p.add_argument('-f','--frames',help='start stop frames (default all)',type=int,nargs=2,default=(None,None))
    p.add_argument('-o','--outdir',help='directory to put output files in',type=str,default='') #None doesn't work with Windows
    p.add_argument('--ms',help='keogram/montage step [1000] dont make it too small like 1 or output is as big as original file!',type=int,default=1000)
    p.add_argument('-c','--contrast',help='[low high] data numbers to bound video contrast',type=int,nargs=2,default=(None,None))
    p.add_argument('--rejectvid',help='reject raw video files with less than this many frames',type=int,default=10)
    p.add_argument('-r','--rejectdet',help='reject files that have fewer than this many detections',type=int,default=10)
    p.add_argument('--paramfn',help='parameter file for cameras',type=str,default='camparam.xlsx')
    p.add_argument('-v','--verbose',help='verbosity',action='store_true')
    p.add_argument('--profile',help='profile debug',action='store_true')
    a = p.parse_args()

    uparams = {'rejvid':a.rejectvid,
              'framestep':a.step,
              'startstop':a.frames,
              'montstep':a.ms,'clim':a.contrast,
              'paramfn':a.paramfn,'rejdet':a.rejectdet,'outdir':a.outdir,
              'fps':a.fps
              }

    if a.savetiff:
        savevideo='tif'
    elif a.savevideo:
        savevideo='vid'
    else:
        savevideo=''

    try:
        flist = walktree(a.indir,'*.' + a.vidext)

        if a.profile:
            import cProfile
            from profilerun import goCprofile
            profFN = 'profstats.pstats'
            print('saving profile results to ' + profFN)
            cProfile.run('main(flist, uparams, savevideo, a.framebyframe, a.verbose)',profFN)
            goCprofile(profFN)
        else:
            final = main(flist, uparams, savevideo, a.framebyframe, a.verbose)
            #show()
    except KeyboardInterrupt:
        exit('aborting per user request')
