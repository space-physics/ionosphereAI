#!/usr/bin/python2
"""we temporarily use python 2.7 until OpenCV 3 is out of beta (will work with Python 3)
Michael Hirsch Dec 2014
This program detects aurora in multi-terabyte raw video data files
It is a major cleanup of the processDrive.sh, filelooper.m, TrackingOF7.m Frankenprograms

0) recursively find all .DMCdata files under requested root directory
1)
"""
from __future__ import division, print_function
import cv2
from cv2 import cv #necessary for Windows, "import cv" doesn't work
from re import search
from pandas import read_excel
from os.path import join,isfile, splitext
from numpy import (isfinite,empty,uint32,delete,mgrid,vstack,int32,arctan2,
                   sqrt,zeros,pi,uint8,minimum,s_,asarray, median, dstack,
                   hypot,inf, logical_and)
from scipy.signal import wiener
import sys
from time import time
import h5py
#from pdb import set_trace
#
sys.path.append('../hist-utils')
from walktree import walktree
from sixteen2eight import sixteen2eight
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

if savedet or plotdet or showhist or showofmag or showmeanmedian:
    from matplotlib.pylab import draw, pause, figure, hist
    from matplotlib.colors import LogNorm
    #from matplotlib.cm import get_cmap


def main(flist, params, savevideo, framebyframe, verbose):
    camser,camparam = getcamparam(params['paramfn'])

    for f,s in zip(flist,camser):
        tic = time()
        stem,ext = splitext(f)
        detfn = join(params['outdir'],f +'_detections.h5')
        if isfile(detfn):
            print('** overwriting existing ' + detfn)

        cparam = camparam[s]

        finf = loadvid(f,cparam,params,verbose)
#%% ingest parameters and preallocate
        twoframe = bool(cparam['twoframe'])
        ofmethod = cparam['ofmethod'].lower()
        rawframeind = empty(finf['nframe'],dtype=uint32)
        rawlim = (cparam['cmin'], cparam['cmax'])
        xpix = finf['superx']; ypix = finf['supery']
        thresmode = cparam['thresholdmode'].lower()
#%% setup avi writing
        svh = svsetup(savevideo, xpix, ypix, isfinite(cparam['wienernhood']))
#%% setup blob
        blobdetect = setupblob(cparam['minblobarea'],cparam['maxblobarea'])
#%% kernel setup
        umat, vmat, cvref, cvgray = setupof(ofmethod,xpix,ypix)

        if not cparam['openradius'] % 2:
            exit('*** detectaurora: openRadius must be ODD')
        openkernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE,
                                               (cparam['openradius'], cparam['openradius']))
        erodekernel = openkernel
        closekernel = cv2.getStructuringElement(cv2.MORPH_RECT, (cparam['closewidth'],cparam['closeheight']))
        #cv2.imshow('open kernel',openkernel)
        print('open kernel');  print(openkernel)
        print('close kernel'); print(closekernel)
        print('erode kernel'); print(erodekernel)

        with open(f, 'rb') as dfid: #TODO need to use the "old-fashioned" syntax and dfid.close()
            jfrm = 0
#%% mag plots setup
            hiom, hpmn, hpmd,medpl,meanpl,fgdt,hpdt,detect = setupfigs(showmeanmedian,
                                                            showofmag,
                                                            plotdet, savedet, finf,f)

            for ifrm in finf['frameind']:
#%% load and filter
                framegray,frameref,rawframeind = getraw(dfid,ifrm,jfrm,finf,svh,
                                                        rawlim,twoframe,rawframeind,cparam['wienernhood'],verbose)
                if framegray is None: break
#%% compute optical flow
                flow,ofmag, ofmed,medpl,meanpl = dooptflow(framegray,frameref,ofmethod,
                                              cparam['trimedgeof'], cparam['hssmooth'],
                                              umat, vmat, cvref, cvgray, medpl, meanpl, jfrm,hpmd,hpmn,hiom)
#%% threshold
                thres = dothres(ofmag, ofmed , thresmode, cparam['ofthresmin'],cparam['ofthresmax'],svh)
#%% despeckle
                despeck = dodespeck(thres,cparam['medfiltsize'],svh)
#%% morphological ops
                morphed = domorph(despeck,erodekernel,closekernel,svh)
#%% blob detection
                final = doblob(morphed,blobdetect,framegray,detect,jfrm,svh,hpdt,savevideo)
#%% plotting in loop
                """
                http://docs.opencv.org/modules/highgui/doc/user_interface.html
                """
                   
                if plotdet or showhist or showofmag or showmeanmedian:
                    draw(); pause(0.001)

                if not jfrm % 10:
                    print('iteration ' + str(jfrm) + '  frame ' + str(ifrm))
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
#%% done looping
            print('{:0.1f}'.format(time()-tic) + ' seconds to process ' + f)
            if savedet:
                detfn = stem + '_det.h5'
                detpltfn = stem + '_det.png'
                print('saving detections to ' + detfn)
                with h5py.File(detfn,libver='latest') as h5fid:
                    h5fid.create_dataset("/det", data=detect)
                print('saving detection plot to ' + detpltfn)
                fgdt.savefig(detpltfn,dpi=100,bbox_inches='tight')

            svrelease(svh)
            
            return final
            
def keyhandler(keypressed,framebyframe):
    if keypressed == -1:
        return (framebyframe,False)
    elif keypressed == 1048608: #space
        return (not framebyframe, False)
    elif keypressed == 1048603: #escape
        return (None, True)
    else:
        print('keypress code: ' + str(keypressed))
        return (framebyframe,False)

def svsetup(savevideo,xpix,ypix,dowiener):
    """ if grayscale video, isColor=False
    http://stackoverflow.com/questions/9280653/writing-numpy-arrays-using-cv2-videowriter
    """
    wfps=3
    fourcc = cv2.cv.FOURCC(*'FFV1')
    svh = {}
    if savevideo:
        if dowiener:
            svh['wiener'] = cv2.VideoWriter('/tmp/wiener.avi',fourcc, wfps,(ypix,xpix),False)
        else:
            svh['wiener'] = None
        
        svh['video']  = cv2.VideoWriter('/tmp/video.avi',fourcc,wfps,(ypix,xpix),False)
        svh['thres']  = cv2.VideoWriter('/tmp/thres.avi',fourcc,wfps,(ypix,xpix),False)
        svh['despeck']= cv2.VideoWriter('/tmp/despeck.avi',fourcc,wfps,(ypix,xpix),False)   
        svh['erode']  = cv2.VideoWriter('/tmp/eroded.avi',fourcc,wfps,(ypix,xpix),False)
        svh['close']  = cv2.VideoWriter('/tmp/closed.avi',fourcc,wfps,(ypix,xpix),False)
        svh['detect'] = cv2.VideoWriter('/tmp/detect.avi',fourcc,wfps,(ypix,xpix),True) #the annotation is color!
        
        for k,v in svh.items():
            if v is not None and not v.isOpened(): 
                exit('*** trouble writing video for ' + k)
    else:
        svh = {'wiener':None,'thres':None,'despeck':None,'erode':None,'close':None,'detect':None}

    return svh
    
def svrelease(svh):
        for k,v in svh.items():
            if v is not None:
                v.release()
                
def setupof(ofmethod,xpix,ypix):
    if ofmethod == 'hs':
        umat =   cv.CreateMat(ypix, xpix, cv.CV_32FC1)
        vmat =   cv.CreateMat(ypix, xpix, cv.CV_32FC1)
        cvref =  cv.CreateMat(ypix, xpix, cv.CV_8UC1)
        cvgray = cv.CreateMat(ypix, xpix, cv.CV_8UC1)
    return umat, vmat, cvref, cvgray
    
def setupblob(minblobarea,maxblobarea):
    blobparam = cv2.SimpleBlobDetector_Params()
    blobparam.filterByArea = True
    blobparam.filterByColor = False
    blobparam.filterByCircularity = False
    blobparam.filterByInertia = False
    blobparam.filterByConvexity = False

    blobparam.minDistBetweenBlobs = 50.0
    blobparam.minArea = minblobarea
    blobparam.maxArea = maxblobarea
    #blobparam.minThreshold = 40 #we have already made a binary image
    return cv2.SimpleBlobDetector(blobparam)
                
def getraw(dfid,ifrm,jfrm,finf,svh,rawlim,twoframe,rawframeind,wienernhood,verbose):
    dowiener = isfinite(wienernhood)
    
    if twoframe:
        frameref = getDMCframe(dfid,ifrm,finf,verbose)[0]

        if dowiener:
            frameref = wiener(frameref,wienernhood)
        frameref = sixteen2eight(frameref, rawlim)

    frame16,rfi = getDMCframe(dfid,ifrm+1,finf)
    if frame16 is None or rfi is None:
        delete(rawframeind,s_[jfrm:])
        return None, None, rawframeind
        
    framegray,rawframeind[jfrm] = (frame16, rfi)

    if dowiener:
        framegray = wiener(framegray,wienernhood)

    if showraw:
        #this just divides by 256, NOT autoscaled!
        # http://docs.opencv.org/modules/highgui/doc/user_interface.html
        cv2.imshow('video', framegray)
    framegray = sixteen2eight(framegray, rawlim)
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
        svh['video'].write(framegray)

    return framegray,frameref,rawframeind
    
def dooptflow(framegray,frameref,ofmethod,trimedge,
              hssmooth,umat, vmat, cvref, cvgray, medpl, meanpl,jfrm,hpmd,hpmn,hiom):

    if ofmethod == 'hs':
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
        cv.CalcOpticalFlowHS(cvref, cvgray, False,
                             umat, vmat,
                             hssmooth,
                             (cv.CV_TERMCRIT_ITER | cv.CV_TERMCRIT_EPS, 8, 0.1))
        flow = dstack((asarray(umat), asarray(vmat)))

    elif ofmethod == 'farneback':
        flow = cv2.calcOpticalFlowFarneback(frameref, framegray,
                                           pyr_scale=0.5,
                                           levels=1,
                                           winsize=3,
                                           iterations=5,
                                           poly_n = 3,
                                           poly_sigma=1.5,
                                           flags=1)
    else:
        exit('*** OF method ' + ofmethod + ' not implemented')

    # zero out edges of image (which have very high flow, unnaturally)

    '''
    maybe this can be done more elegantly, maybe via pad or take?
    http://stackoverflow.com/questions/13525266/multiple-slice-in-list-indexing-for-numpy-array
    '''
    flow[:trimedge,...] = 0.; flow[-trimedge:,...] = 0.
    flow[:,:trimedge,:] = 0.; flow[:,-trimedge:,:] = 0.

    flow /= 255. #trying to make like matlab, which has normalized data input (opencv requires uint8)

#%% compute median and magnitude
    ofmag = hypot(flow[...,0], flow[...,1])
    ofmed = median(ofmag)
    if showofmag:
        medpl[jfrm] = ofmed
        meanpl[jfrm] = ofmag.mean()
        hpmd[0].set_ydata(medpl)
        hpmn[0].set_ydata(meanpl)
        
    if showflowvec:
        cv2.imshow('flow vectors', draw_flow(framegray,flow) )
    if showflowhsv:
        cv2.imshow('flowHSV', draw_hsv(flow) )
    if showofmag:
        #cv2.imshow('flowMag', ofmag) #was only grayscale, I wanted color
        hiom.set_data(ofmag)
       
    return flow,ofmag, ofmed, medpl, meanpl

def dothres(ofmag,medianflow,thresmode,thmin,thmax,svh):
    if thresmode == 'median':
        if medianflow>1e-6:  #median is scalar
            lowthres = thmin * medianflow #median is scalar!
            hithres = thmax * medianflow #median is scalar!
        else: #median ~ 0
            lowthres = 0
            hithres = inf

    elif thresmode == 'runningmean':
        exit('*** ' + thresmode + ' not yet implemented')
    else:
        exit('*** ' + thresmode + ' not yet implemented')
    
    thres = logical_and(ofmag < hithres, ofmag > lowthres).astype(uint8) * 255
        
    if svh['thres'] is not None:
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
    #return (ofmag > lowthres).astype(uint8) * 255
    
def dodespeck(thres,medfiltsize,svh):
    despeck = cv2.medianBlur(thres,ksize=medfiltsize)
    if svh['despeck'] is not None:
        svh['despeck'].write(despeck)
    if showthres:
        cv2.imshow('despeck', despeck)
    return despeck
    
def domorph(despeck,erodekernel,closekernel,svh):
    """
    http://docs.opencv.org/master/doc/py_tutorials/py_imgproc/py_morphological_ops/py_morphological_ops.html
    """
   # opened = cv2.morphologyEx(despeck, cv2.MORPH_OPEN, openkernel)
    eroded = cv2.erode(despeck,erodekernel)
    closed = cv2.morphologyEx(eroded, cv2.MORPH_CLOSE, closekernel)
    
    if svh['erode'] is not None:
        svh['erode'].write(eroded)
    if svh['close'] is not None:
        svh['close'].write(closed)
    
    if showmorph:
        #cv2.imshow('opened', opened)
        cv2.imshow('morphed',closed)
    
    return closed
    
def doblob(morphed,blobdetect,framegray,detect,jfrm,svh,hpdt,savevideo):
    """
    http://docs.opencv.org/master/modules/features2d/doc/drawing_function_of_keypoints_and_matches.html
    """
    keypoints = blobdetect.detect(morphed)
    nkey = len(keypoints)
    final = cv2.drawKeypoints(framegray, keypoints, flags=cv2.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS)
    cv2.putText(final, text=str(nkey), org=(10,510),
                fontFace=cv2.FONT_HERSHEY_PLAIN, fontScale=5,
                color=(0,255,0), thickness=2)
                
    if showfinal:
        cv2.imshow('final',final)
    if savevideo:
        #print('saving frame',ifrm)
        svh['detect'].write(final)

    if plotdet or savedet:
        detect[jfrm] = nkey
        hpdt[0].set_ydata(detect)

def setupfigs(showmeanmedian,showofmag,plotdet,savedet,finf,fn):
    hiom = None
    if showofmag:
        figure(30).clf()
        figom = figure(30)
        axom = figom.gca()
        hiom = axom.imshow(zeros((finf['supery'],finf['superx'])),vmin=1e-4, vmax=0.1,
                           origin='bottom', norm=LogNorm())#, cmap=lcmap) #arbitrary limits
        axom.set_title('optical flow magnitude')
        figom.colorbar(hiom,ax=axom)

    hpmn = None; hpmd = None; medpl = None; meanpl = None
    if showmeanmedian:
        medpl = zeros(finf['frameind'].size, dtype=float) #don't use nan, it won't plot
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
        detect = zeros(finf['frameind'].size, dtype=int)
        figure(40).clf()
        fgdt = figure(40)
        axdt = fgdt.gca()
        axdt.set_title('Detections of Aurora: ' + fn,fontsize =10)
        axdt.set_xlabel('frame index #')
        axdt.set_ylabel('number of detections')
        axdt.set_ylim((0,10))
        hpdt = axdt.plot(detect)

    return hiom, hpmn, hpmd,medpl,meanpl,fgdt,hpdt,detect

def draw_flow(img, flow, step=16):
    """
    this came from opencv/examples directory
    another way: http://docs.opencv.org/trunk/doc/py_tutorials/py_gui/py_drawing_functions/py_drawing_functions.html
    """
    #scaleFact = 1. #arbitary factor to make flow visible
    canno = (0, 65535, 0)  # 65535 since it's 16-bit images
    h, w = img.shape[:2]
    y, x = mgrid[step//2:h:step, step//2:w:step].reshape(2,-1)
    fx, fy =  flow[y,x].T
    #create line endpoints
    lines = vstack([x, y, (x+fx), (y+fy)]).T.reshape(-1, 2, 2)
    lines = int32(lines + 0.5)
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
    ang = arctan2(fy, fx) + pi
    v = sqrt(fx*fx+fy*fy)
    hsv = zeros((h, w, 3), uint8)
    hsv[...,0] = ang*(180/pi/2)
    hsv[...,1] = 255
    hsv[...,2] = minimum(v*4, 255)
    bgr = cv2.cvtColor(hsv, cv2.COLOR_HSV2BGR)
    return bgr

def loadvid(fn,cparam,params,verbose):
    print('using ' + cparam['ofmethod'] + ' for ' + fn)
    if verbose:
        print('minBlob='+str(cparam['minblobarea']) + ' maxBlob='+
          str(cparam['maxblobarea']) + ' maxNblob=' +
          str(cparam['maxblobcount']) )

    xypix=(cparam['xpix'],cparam['ypix'])
    xybin=(cparam['xbin'],cparam['ybin'])
    if params['startstop'][0] is None:
        finf = getDMCparam(fn,xypix,xybin,params['framestep'],verbose)
    else:
        finf = getDMCparam(fn,xypix,xybin,(params['startstop'][0], params['startstop'][1], params['framestep']))
    return finf

def getserialnum(flist):
    sn = []
    for f in flist:
        sn.append(int(search(r'(?<=CamSer)\d{3,6}',f).group()))
    return sn

def getcamparam(paramfn):
    camser = getserialnum(flist)
    camparam = read_excel(paramfn,index_col=0,header=0)
    return camser, camparam

if __name__=='__main__':
    from argparse import ArgumentParser
    p = ArgumentParser(description='detects aurora in raw video files')
    p.add_argument('indir',help='top directory over which to recursively find video files',type=str)
    p.add_argument('vidext',help='extension of raw video file',nargs='?',type=str,default='DMCdata')
    p.add_argument('-p','--framebyframe',help='space bar toggles play/pause', action='store_true')
    p.add_argument('-s','--savevideo',help='save video at each step (can make enormous files)',action='store_true')
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

    params = {'rejvid':a.rejectvid,
              'framestep':a.step,
              'startstop':a.frames,
              'montstep':a.ms,'clim':a.contrast,
              'paramfn':a.paramfn,'rejdet':a.rejectdet,'outdir':a.outdir,
              }
    try:
        flist = walktree(a.indir,'*.' + a.vidext)

        if a.profile:
            import cProfile
            from profilerun import goCprofile
            profFN = 'profstats.pstats'
            print('saving profile results to ' + profFN)
            cProfile.run('main(flist, params, a.savevideo, a.framebyframe, a.verbose)',profFN)
            goCprofile(profFN)
        else:
            final = main(flist, params, a.savevideo, a.framebyframe, a.verbose)
            #show()
    except KeyboardInterrupt:
        exit('aborting per user request')