#!/usr/bin/python2
"""
Michael Hirsch Dec 2014
This program detects aurora in multi-terabyte raw video data files
It is also used for the Haystack passive FM radar ionospheric activity detection
"""
from __future__ import division, absolute_import
try:
    import cv2
except ImportError as e:
    print('This program requires OpenCV2 or OpenCV3 installed into your Python')
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
print('OpenCV '+str(cv2.__version__)) #some installs of OpenCV don't give a consistent version number, just a build number and I didn't bother to parse this.
#
from pandas import read_excel
from os.path import join,isfile, splitext
import numpy as np
from scipy.signal import wiener
from scipy.misc import bytescale
from time import time
from tempfile import gettempdir
from warnings import warn
#
try:
    from .cvops import dooptflow,dothres,dodespeck,domorph,doblob
    from .getpassivefm import getfmradarframe
    from .histutils.walktree import walktree
    from .histutils.rawDMCreader import getDMCparam,getDMCframe,getserialnum
except:
    from cvops import dooptflow,dothres,dodespeck,domorph,doblob
    from getpassivefm import getfmradarframe
    from histutils.walktree import walktree
    from histutils.rawDMCreader import getDMCparam,getDMCframe,getserialnum

#plot disable
pshow = ('thres','final')
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
         
savedet = False
complvl = 4 #tradeoff b/w speed and filesize for TIFF

#only import matplotlib if needed to save time
if savedet or np.in1d(('det','hist','ofmag','meanmedian'),pshow).any():
    from matplotlib.pylab import draw, pause, figure, hist
    from matplotlib.colors import LogNorm
    #from matplotlib.cm import get_cmap

try:
    import h5py
except ImportError as e:
    warn('h5py not working. Wont be able to save detections to disk')
    print(str(e))
    savedet=False

def loopaurorafiles(flist, up, savevideo, framebyframe, verbose):
    if not flist:
        warn('no files specified')
        return

    camser,camparam = getcamparam(up['paramfn'],flist)

    for f,s in zip(flist,camser): #iterate over files in list
        result = procaurora(f,s,camparam,up,savevideo,framebyframe,verbose)

def procaurora(f,s,camparam,up,savevideo,framebyframe,verbose=False):
    tic = time()
    #setup output file
    stem = splitext(f)[0]
    detfn = join(up['outdir'],f +'_detections.h5')
    if isfile(detfn):
        print('** overwriting existing ' + detfn)

    try:
        cp = camparam[s] #pick the parameters for this camara from pandas DataFrame
    except KeyError:
        print('* using first column of '+up['paramfn'] + ' as I didnt find '+str(s)+' in it.')
        cp = camparam.iloc[:,s] #fallback to first column

    finf,ap,dfid = getvidinfo(f,cp,up,verbose)
    if finf is None: return
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
#%% start main loop
    for ifrm in finf['frameind'][:-1]:
#%% load and filter
        framegray,frameref,ap = getraw(dfid,ifrm,finf,svh,ap,cp,savevideo,verbose)
        if framegray is None: break
#%% compute optical flow or Background/Foreground
        if lastflow is not None: #very fast way to check mode
            flow,ofmaggmm,ofmed,pl = dooptflow(framegray,frameref,lastflow,uv,
                                               ifrm, ap,cp,pl,pshow)
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
        final = doblob(morphed,blobdetect,framegray,ifrm,svh,pl,pshow) #lint:ok
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
    try:
        if finf['reader'] == 'raw':
            dfid.close()
        elif finf['reader'] == 'cv2':
            dfid.release()
    except Exception as e:
        print(str(e))

    print('{:0.1f}'.format(time()-tic) + ' seconds to process ' + f)
    if savedet:
        detfn = stem + '_det.h5'
        detpltfn = stem + '_det.png'
        try:
            print('saving detections to ' + detfn)
            with h5py.File(detfn,'w',libver='latest') as h5fid:
                h5fid["/det"] = pl['detect']
            print('saving detection plot to ' + detpltfn)
            pl['fdet'].savefig(detpltfn,dpi=100,bbox_inches='tight')
        except Exception as e:
            print('** trouble saving detection result')
            print(str(e))

    svrelease(svh,savevideo)
    return pl


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
    if savevideo:
        print('dumping video output to '+tdir)
    svh = {'video':None, 'wiener':None,'thres':None,'despeck':None,
           'erode':None,'close':None,'detect':None,'save':savevideo,'complvl':complvl}
    if savevideo == 'tif':
        #complvl = 6 #0 is uncompressed
        try:
            from tifffile import TiffWriter  #pip install tifffile
        except ImportError as e:
            print('** I cannot save iterated video results due to missing tifffile module')
            print('try pip install tifffile')
            print(str(e))
            return svh

        if dowiener:
            svh['wiener'] = TiffWriter(join(tdir,'wiener.tif'))
        else:
            svh['wiener'] = None

        svh['video']  = TiffWriter(join(tdir,'video.tif')) if 'rawscaled' in pshow else None
        svh['thres']  = TiffWriter(join(tdir,'thres.tif')) if 'thres' in pshow else None
        svh['despeck']= TiffWriter(join(tdir,'despk.tif')) if 'thres' in pshow else None
        svh['erode']  = TiffWriter(join(tdir,'erode.tif')) if 'morph' in pshow else None
        svh['close']  = TiffWriter(join(tdir,'close.tif')) if 'morph' in pshow else None
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

        svh['video']  = cv2.VideoWriter(join(tdir,'video.avi'), cc4,wfps, (ypix,xpix),False) if 'rawscaled' in pshow else None
        svh['thres']  = cv2.VideoWriter(join(tdir,'thres.avi'), cc4,wfps, (ypix,xpix),False) if 'thres' in pshow else None
        svh['despeck']= cv2.VideoWriter(join(tdir,'despk.avi'), cc4,wfps, (ypix,xpix),False) if 'thres' in pshow else None
        svh['erode']  = cv2.VideoWriter(join(tdir,'erode.avi'), cc4,wfps, (ypix,xpix),False) if 'morph' in pshow else None
        svh['close']  = cv2.VideoWriter(join(tdir,'close.avi'), cc4,wfps, (ypix,xpix),False) if 'morph' in pshow else None
        svh['detect'] = cv2.VideoWriter(join(tdir,'detct.avi'), cc4,wfps, (ypix,xpix),True)  if 'final' in pshow else None

        for k,v in svh.items():
            if v is not None and not v.isOpened():
                exit('*** trouble writing video for ' + k)

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
            umat =   cv.CreateMat(ypix, xpix, cv.CV_32FC1)
            vmat =   cv.CreateMat(ypix, xpix, cv.CV_32FC1)
            lastflow = np.nan #nan instead of None to signal to use OF instead of GMM
        except NameError as e:
            print("*** OpenCV 3 doesn't have legacy cv functions such as {}. You're using OpenCV {}  Please use another CV method".format(ap['ofmethod'],cv2.__version__))
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
    """ this function reads the reference frame too--which makes sense if youre
       only reading every Nth frame from the multi-TB file instead of every frame
    """
    frameref = None #just in case not used
    dowiener = np.isfinite(cp['wienernhood'])
#%% reference frame

    if finf['reader'] == 'raw':
        if ap['twoframe']:
            frameref = getDMCframe(dfid,ifrm,finf,verbose)[0]
            frameref = bytescale(frameref, ap['rawlim'][0], ap['rawlim'][1])
            if dowiener:
                frameref = wiener(frameref,cp['wienernhood'])

        frame16,rfi = getDMCframe(dfid,ifrm+1,finf)
        if frame16 is None or rfi is None: #FIXME accidental end of file, smarter way to detect beforehand?
            ap['rawframeind'] = np.delete(ap['rawframeind'], np.s_[ifrm:])
            return None, None, ap
        framegray = bytescale(frame16, ap['rawlim'][0], ap['rawlim'][1])

    elif finf['reader'] == 'cv2':
        if ap['twoframe']:
            retval,frameref = dfid.read()
            if not retval:
                if ifrm==0:
                    print('*** could not read video file, sorry')
                print('done reading video.')
                return None, None, ap
            frameref = cv2.cvtColor(frameref, cv2.COLOR_BGR2GRAY)
            if dowiener:
                frameref = wiener(frameref,cp['wienernhood'])

        retval,frame16 = dfid.read() #TODO this is skipping every other frame!
        rfi = ifrm
        if not retval:
            print('*** could not read video from file!')
            return None, None, ap
        #FIXME what if the original avi is gray? didn't try this yet.
        framegray = cv2.cvtColor(frame16, cv2.COLOR_BGR2GRAY)
    elif finf['reader'] == 'h5':   #one frame per file
        if ap['twoframe']:
            frameref = getfmradarframe(dfid[ifrm])[2]
            frameref = bytescale(frameref, ap['rawlim'][0], ap['rawlim'][1])
        frame16 = getfmradarframe(dfid[ifrm+1])[2]
        rfi = ifrm
        framegray = bytescale(frame16, ap['rawlim'][0], ap['rawlim'][1])


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

def setupfigs(finf,fn):


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
    if 'det' in pshow or savedet:
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

def getvidinfo(fn,cp,up,verbose):
    print('using {} for {}'.format(cp['ofmethod'],fn))
    if verbose:
        print('minBlob='+str(cp['minblobarea']) + ' maxBlob='+
          str(cp['maxblobarea']) + ' maxNblob=' + str(cp['maxblobcount']) )

    if fn.endswith('.DMCdata'):
        xypix=(cp['xpix'],cp['ypix'])
        xybin=(cp['xbin'],cp['ybin'])
        if up['startstop'][0] is None:
            finf = getDMCparam(fn,xypix,xybin,up['framestep'],verbose)
        else:
            finf = getDMCparam(fn,xypix,xybin,
                     (up['startstop'][0], up['startstop'][1], up['framestep']))
        finf['reader']='raw'

        dfid = open(fn,'rb') #I didn't use the "with open(f) as ... " because I want to swap in other file readers per user choice

    elif fn.lower().endswith(('.h5','.hdf5')):
        finf = {'reader':'h5'}
        print('attempting to read HDF5 ' + str(fn))
        dfid = flist
        finf['nframe'] = len(dfid) # currently the passive radar uses one file per frame

        range_km,vel_mps = getfmradarframe(fn)[:2] #assuming all frames are the same size
        finf['superx'] = range_km.size
        finf['supery'] = vel_mps.size
        finf['frameind'] = np.arange(finf['nframe'],dtype=np.int64)
    else:
        #FIXME start,stop,step is not yet implemented, simply uses every other frame
        print('attempting to read ' + str(fn) + ' with OpenCV.')
        finf = {'reader':'cv2'}
        dfid = cv2.VideoCapture(fn)
        nframe = np.int64(dfid.get(cv.CV_CAP_PROP_FRAME_COUNT))
        xpix = int(dfid.get(cv.CV_CAP_PROP_FRAME_WIDTH))
        ypix = int(dfid.get(cv.CV_CAP_PROP_FRAME_HEIGHT))
        if nframe<1 or xpix<1 or ypix<1:
            print('** I may not be reading {} correctly, trying anyway by reading an initial frame..'.format(fn))
            retval, frame =dfid.read()
            if not retval:
                print('*** could not succeed in any way to read '+str(fn))
                return None, None, None
            ypix,xpix = frame.shape
            finf['nframe'] = 100000 #FIXME guessing how many frames in file
        else:
            finf['nframe'] = nframe
        finf['superx'] = xpix
        finf['supery'] = ypix

        finf['frameind']=np.arange(finf['nframe'],dtype=np.int64)


#%% extract analysis parameters
    ap = {'twoframe':bool(cp['twoframe']), # note this should be 1 or 0 input, not the word, because even the word 'False' will be bool()-> True!
          'ofmethod':cp['ofmethod'].lower(),
          'rawframeind': np.empty(finf['nframe'],dtype='int64'), #int64 for very large files on Windows Python 2.7, long is not available on Python3
          'rawlim': (cp['cmin'], cp['cmax']),
          'xpix': finf['superx'], 'ypix':finf['supery'],
          'thresmode':cp['thresholdmode'].lower()}



    return finf, ap, dfid

def getcamparam(paramfn,flist):
    #uses pandas and xlrd to parse the spreadsheet parameters
    if flist[0].endswith('.DMCdata'):
        camser = getserialnum(flist)
    else:
        #FIXME add your own criteria to pick which spreadsheet paramete column to use.
        # for now I tell it to just use the first column (same criteria for all files)
        print('* using first column of spreadsheet only for camera parameters')
        camser = [0] * len(flist) #we don't need to bother with itertools.repeat, we have a short list

    camparam = read_excel(paramfn,index_col=0,header=0) #returns a nicely indexable DataFrame
    return camser, camparam

if __name__=='__main__':
    from argparse import ArgumentParser
    p = ArgumentParser(description='detects aurora in raw video files')
    p.add_argument('indir',help='specify file, OR top directory over which to recursively find video files',type=str,nargs='+')
    p.add_argument('-e','--vidext',help='extension of raw video file',type=str,default='DMCdata')
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
            import cProfile,pstats
            profFN = 'profstats.pstats'
            cProfile.run('main(flist, uparams, savevideo, a.framebyframe, a.verbose)',profFN)
            pstats.Stats(profFN).sort_stats('time','cumulative').print_stats(50)
        else:
            loopaurorafiles(flist, uparams, savevideo, a.framebyframe, a.verbose)
            #show()
    except KeyboardInterrupt:
        exit('aborting per user request')
