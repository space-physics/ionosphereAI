#!/usr/bin/env python
from os import path
import sys
import numpy as np
import cv2
# local import
sys.path.append('../hist-utils')
sys.path.append('../cv-hst')
import rawDMCreader as rdr
from sixteen2eight import sixteen2eight

#------------------------------------
def main(BigFN,xyPix=(512,512),xyBin=(1,1),FrameInd=(0,0),playMovie=0.01,
         Clim=(None,None), rawFrameRate=None,startUTC=None):
    BigFN = path.expanduser(BigFN)
    BigExt = path.splitext(BigFN)[1]

    if rawFrameRate is None:
        startUTC=None

#get params
    if (BigExt == '.DMCdata'):
        finf = rdr.getDMCparam(
           BigFN,xyPix[0],xyPix[1],xyBin[0],xyBin[1],FrameInd)
        fid = open(BigFN, 'rb')
        (grayRef,rawIndRef) = rdr.getDMCframe(fid,0,BytesPerFrame,PixelsPerImage,Nmetadata,SuperX,SuperY)
        grayRef = sixteen2eight(grayRef,Clim)
    elif (BigExt == '.avi'):
        fid = cv2.VideoCapture(BigFN)
        #print(fid.isOpened())
        nFrame = fid.get(cv2.cv.CV_CAP_PROP_FRAME_COUNT)
        if all(FrameInd == 0):
            FrameInd = np.arange(0,nFrame,1,dtype=np.uint64) # has to be numpy for > comparison
        grayRef = cv2.cvtColor(fid.read()[1], cv2.COLOR_BGR2GRAY)
    else: sys.exit('unknown file type: ' + BigExt)

    print('processing '+str(nFrame) + ' frames from ' + BigFN)

#setup figures
    cv2.namedWindow('image', cv2.WINDOW_NORMAL)
    cv2.namedWindow('flowHSV', cv2.WINDOW_NORMAL)

    for iFrm in FrameInd:
        if (BigExt == '.DMCdata'):
            (gray,rawIndGray) = rdr.getDMCframe(fid,iFrm+1,BytesPerFrame,PixelsPerImage,Nmetadata,SuperX,SuperY)
            gray = sixteen2eight(gray,Clim)
        elif (BigExt == '.avi'):
            gray = cv2.cvtColor(fid.read()[1], cv2.COLOR_BGR2GRAY)

        flow = cv2.calcOpticalFlowFarneback(grayRef, gray,
               pyr_scale=0.5, levels=1, winsize=3, iterations=5,
               poly_n = 3, poly_sigma=1.5,flags=1)
# plotting
        cv2.imshow('image', draw_flow(gray,flow) )
        cv2.imshow('flowHSV', draw_hsv(flow) )
        if cv2.waitKey(1) == 27: # MANDATORY FOR PLOTTING TO WORK!
            break
        grayRef = gray; #rawIndRef = rawIndGray
#------------
    if (BigExt == '.DMCdata'):
        fid.close()
    elif (BigExt == '.avi'):
        fid.release()

    cv2.destroyAllWindows()
#-----------------------------------------------------------


def draw_flow(img, flow, step=16):
    scaleFact = 10 #arbitary factor to make flow visible
    h, w = img.shape[:2]
    y, x = np.mgrid[step/2:h:step, step/2:w:step].reshape(2,-1)
    fx, fy = scaleFact * flow[y,x].T
    #create line endpoints
    lines = np.vstack([x, y, x+fx, y+fy]).T.reshape(-1, 2, 2)
    lines = np.int32(lines) #+ 0.5)
    #create image and draw
    vis = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)
    cv2.polylines(vis, lines, 0, (0, 255, 0))
    for (x1, y1), (x2, y2) in lines:
        cv2.circle(vis, (x1, y1), 1, (0, 255, 0), -1)
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

def warp_flow(img, flow):
    h, w = flow.shape[:2]
    flow = -flow
    flow[:,:,0] += np.arange(w)
    flow[:,:,1] += np.arange(h)[:,np.newaxis]
    res = cv2.remap(img, flow, None, cv2.INTER_LINEAR)
    return res

if __name__ == '__main__':
    from argparse import ArgumentParser
    p = ArgumentParser(description='crude optical flow with OpenCV')
    p.add_argument('videofile',help='video data file to process',type=str,default=None)
    p.add_argument('--xypix',help='number of x,y pixels',type=int,default=(512,512))
    p.add_argument('--xybin',help='pixel binning x,y',type=int,default=(1,1))
    p.add_argument('--frameind',help='frame indices to process (start,stop)',type=int,default=(0,0))
    p.add_argument('--play',help='frame time in seconds for playback (arbitrary)',type=float,default=0.01)
    p.add_argument('--clim',help='arbitrary pixel value limits for playback display (low,high)',type=float,default=(None,None))
    p.add_argument('--framerate',help='frame rate of rate data (at time camera ran)',type=float,default=None)
    p.add_argument('--tstart',help='time camera started that night yyyy-mm-ddTHH:MM:SS.fffZ',type=str,default=None)
    a=p.parse_args()

    main(a.infile,a.xypix,a.xybin,a.frameind,a.play,a.clim.a.framerate,a.tstart)