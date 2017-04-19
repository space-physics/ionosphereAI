from configparser import ConfigParser
from numpy import arange,empty
from pandas import DataFrame
import h5py
from pathlib import Path
#
from .getpassivefm import getfmradarframe
from morecvutils.getaviprop import getaviprop
from histutils.rawDMCreader import getDMCparam,getNeoParam
from dmcutils.neospool import spoolparam

SPOOLINI = 'acquisitionmetadata.ini' # for Solis spool files

def getvidinfo(fn,cp,up,verbose=False):
    #print('using {} for {}'.format(cp['main']['ofmethod'],fn))
    if verbose:
        print(f'minBlob={cp["blob"]["minblobarea"]}'
              f'maxBlob={cp["blob"]["maxblobarea"]}'
              f'maxNblob={cp["blob"]["maxblobcount"]}')

    try: # for spool case
        fn = fn[0]
    except TypeError:
        pass

    if fn.suffix.lower() in ('.dmcdata',): # HIST
        xypix=(cp.getint('main','xpix'), cp.getint('main','ypix'))
        xybin=(cp.getint('main','xbin'), cp.getint('main','ybin'))
        if up['startstop'][0] is None:
            finf = getDMCparam(fn,xypix,xybin,up['framestep'],verbose=verbose)
        else:
            finf = getDMCparam(fn,xypix,xybin,
                     (up['startstop'][0], up['startstop'][1], up['framestep']),
                      verbose=verbose)
        finf['reader']='raw'
    elif fn.suffix.lower() in ('.dat',): # Andor Solis spool file
            finf = spoolparam(fn.parent/SPOOLINI)
            finf['reader'] = 'spool'
            finf['nframe'] = up['nfile'] * finf['nframefile']
            # FIXME should we make this general to all file types?
            if up['nfile'] > 1 and finf['nframe'] > 10* up['framestep']:
                finf['frameind'] = arange(0,finf['nframe'], up['framestep'], dtype=int)
            else:  # revert to all frames because there aren't many, rather than annoying with zero result
                finf['frameind'] = arange(finf['nframe'], dtype=int)
            finf['kinetic'] = None # FIXME blank for now
    elif fn.suffix.lower() in ('.h5','.hdf5'):
#%% determine if optical or passive radar
        with h5py.File(str(fn), 'r', libver='latest') as f:
            if 'rawimg' in f: #hst image/video file
                finf = {'reader':'h5vid'}
                finf['nframe'] = f['rawimg'].shape[0]
                finf['superx'] = f['rawimg'].shape[2]
                finf['supery'] = f['rawimg'].shape[1]
                #print('HDF5 video file detected {}'.format(fn))
            elif 'ambiguity' in f: # Haystack passive FM radar file
                finf = {'reader':'h5fm'}
                finf['nframe'] = 1 # currently the passive radar uses one file per frame
                range_km,vel_mps = getfmradarframe(fn)[:2] #assuming all frames are the same size
                finf['superx'] = range_km.size
                finf['supery'] = vel_mps.size
                #print('HDF5 passive FM radar file detected {}'.format(fn))
            else:
                raise NotImplementedError('unknown HDF5 file type')

        finf['frameind'] = arange(0,finf['nframe'], up['framestep'], dtype=int)
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

        #dfid = fits.open(str(fn),mode='readonly',memmap=False)

        #finf['frameind'] = arange(0,finf['nframe'],up['framestep'],dtype=int)
    else: #assume video file
        #TODO start,stop,step is not yet implemented, simply uses every other frame
        print(f'attempting to read {fn} with OpenCV.')
        finf = {'reader':'cv2'}

        vidparam = getaviprop(fn)
        finf['nframe'] = vidparam['nframe']
        finf['superx'] = vidparam['xpix']
        finf['supery'] = vidparam['ypix']

        finf['frameind'] = arange(finf['nframe'], dtype=int)
#%% extract analysis parameters
    ap = {'twoframe': cp.getboolean('main','twoframe'),
          'ofmethod': cp.get('main','ofmethod').lower(),
          'rawframeind': empty(finf['nframe'], int),
          'rawlim': [cp.getfloat('main','cmin'), #list not tuple for auto
                     cp.getfloat('main','cmax')],
          'xpix': finf['superx'], 'ypix':finf['supery'],
          'thresmode': cp.get('filter','thresholdmode').lower()}

#%% concatenate dicts
    up = {**up, **ap}

    return finf, up

def getparam(pfn):
    pfn = Path(pfn).expanduser()
    P = ConfigParser(allow_no_value=True)
    P.read(pfn)

    return P

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
    if len(stat) == 0:
        return
    assert isinstance(stat,DataFrame)
    print('saving detections & statistics to {}'.format(fn))

    with h5py.File(str(fn),'w',libver='latest') as f:
        f['/detect']  = stat['detect']
        f['/mean']    = stat['mean']
        f['/median']  = stat['median']
        f['/variance']= stat['variance']
