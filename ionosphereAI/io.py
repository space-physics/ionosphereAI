from configparser import ConfigParser
from numpy import arange
from pandas import DataFrame
import h5py
from pathlib import Path
from typing import Dict, Any, Tuple, Sequence
import logging
#
from .getpassivefm import getfmradarframe

try:
    from morecvutils.getaviprop import getaviprop
except ImportError as e:
    logging.debug(e)
    getaviprop = None
try:
    from histutils.rawDMCreader import getDMCparam
    from histutils.solis import getNeoParam
except ImportError as e:
    logging.debug(e)
    getDMCparam = getNeoParam = None

try:
    from dmcutils.neospool import spoolparam
except ImportError as e:
    logging.debug(e)
    spoolparam = None

try:
    import imageio
except ImportError as e:
    logging.debug(e)
    imageio = None

SPOOLINI = 'acquisitionmetadata.ini'  # for Solis spool files


def getvidinfo(files: Sequence[Path],
               P,
               U: Dict[str, Any]) -> Tuple[Dict[str, Any], Dict[str, Any]]:

    def _spoolcase(fn, P, U, f0):
        if spoolparam is None:
            raise ImportError('pip install dmcutils')

        finf = spoolparam(fn.parent/SPOOLINI,
                          P.getint('main', 'xpix', fallback=None), P.getint('main', 'ypix', fallback=None),
                          P.getint('main', 'stride', fallback=0))
        finf = {**finf, **f0}

        print('using spool file, config', fn.parent/SPOOLINI)
        finf['reader'] = 'spool'
        finf['nframe'] = U['nfile']  # first frame pair of each spool file

        # FIXME should we make this general to all file types?
        if U['nfile'] > 1 and finf['nframe'] > 10 * U['framestep']:
            finf['frameind'] = arange(0, finf['nframe'], U['framestep'], dtype=int)
        else:  # revert to all frames because there aren't many, rather than annoying with zero result
            finf['frameind'] = arange(finf['nframe'], dtype=int)
        finf['kinetic'] = None  # FIXME blank for now

        finf['path'] = fn.parent

        return finf

    # print('using {} for {}'.format(P['main']['ofmethod'],fn))
    logging.debug(f'minBlob={P["blob"]["minblobarea"]}'
                  f'maxBlob={P["blob"]["maxblobarea"]}'
                  f'maxNblob={P["blob"]["maxblobcount"]}')

    if isinstance(files, Path):
        files = [files]
    fn = files[0]

    if fn.suffix.lower() == '.dmcdata':  # HIST
        if getDMCparam is None:
            raise ImportError('pip install histutils')
        xypix = (P.getint('main', 'xpix'), P.getint('main', 'ypix'))
        xybin = (P.getint('main', 'xbin'), P.getint('main', 'ybin'))
        if U['startstop'] is None:
            finf = getDMCparam(fn, xypix, xybin, U['framestep'])
        elif len(U['startstop']) == 2:
            finf = getDMCparam(fn, xypix, xybin,
                               (U['startstop'][0], U['startstop'][1], U['framestep']))
        else:
            raise ValueError('start stop must both or neither be specified')

        finf['reader'] = 'raw'
        finf['nframe'] = finf['nframeextract']
        finf['frameind'] = finf['frameindrel']
    elif fn.suffix.lower() == '.dat':  # Andor Solis spool file
        finf = _spoolcase(fn, P, U, {})
    elif fn.suffix.lower() in ('.h5', '.hdf5'):
        finf = {}
        try:  # can't read inside context
            with h5py.File(fn, 'r') as f:
                finf['flist'] = f['fn'][:]
            # finf['flist'] = read_hdf(fn,'filetick')
            if U['startstop'] is not None:
                finf['flist'] = finf['flist'][U['startstop'][0]:U['startstop'][1]]

            U['nfile'] = len(finf['flist'])
            logging.info(f'taking {U["nfile"]} files from index {fn}')
        except KeyError:
            pass
# %% determine if optical or passive radar
        with h5py.File(fn, 'r') as f:
            if 'rawimg' in f:  # hst image/video file
                finf = {'reader': 'h5vid'}
                finf['nframe'] = f['rawimg'].shape[0]
                finf['superx'] = f['rawimg'].shape[2]
                finf['supery'] = f['rawimg'].shape[1]
                # print('HDF5 video file detected {}'.format(fn))
            elif 'ambiguity' in f:  # Haystack passive FM radar file
                finf = {'reader': 'h5fm'}
                finf['nframe'] = 1  # currently the passive radar uses one file per frame
                range_km, vel_mps = getfmradarframe(fn)[:2]  # assuming all frames are the same size
                finf['superx'] = range_km.size
                finf['supery'] = vel_mps.size
                # print('HDF5 passive FM radar file detected {}'.format(fn))
            elif 'ticks' in f:  # Andor Solis spool file index from dmcutils/Filetick.py
                finf = _spoolcase(fn, P, U, finf)
                finf['path'] = fn.parent
            else:
                raise ValueError(f'{fn}: unknown input HDF5 file type')

        if 'frameind' not in finf:
            finf['frameind'] = arange(0, finf['nframe'], U['framestep'], dtype=int)
    elif fn.suffix.lower() in ('.fit', '.fits'):
        if getNeoParam is None:
            raise ImportError('pip install histutils')
        finf = getNeoParam(fn, U['framestep'])
        finf['reader'] = 'fits'
    elif fn.suffix.lower() in ('.tif', '.tiff'):
        if getNeoParam is None:
            raise ImportError('pip install histutils')
        finf = getNeoParam(fn, U['framestep'])
        finf['reader'] = 'tiff'
    else:  # assume video file
        if getaviprop is None:
            raise ImportError('pip install morecvutils')
        # TODO start,stop,step is not yet implemented, simply uses every other frame
        logging.info(f'attempting to read {fn} with OpenCV.')
        finf = {'reader': 'cv2'}

        vidparam = getaviprop(fn)
        finf['nframe'] = vidparam['nframe']
        finf['superx'] = vidparam['xpix']
        finf['supery'] = vidparam['ypix']

        finf['frameind'] = arange(finf['nframe'], dtype=int)

        U['h_read'] = imageio.get_reader(f'imageio:{fn}')

# %% extract analysis parameters
    U.update({'twoframe': P.getboolean('main', 'twoframe'),
              'ofmethod': P.get('main', 'ofmethod').lower(),
              #          'rawframeind': empty(finf['nframe'], int),
              'rawlim': [P.getfloat('main', 'cmin'),  # list not tuple for auto
                         P.getfloat('main', 'cmax')],
              'xpix': finf['superx'], 'ypix': finf['supery'],
              'thresmode': P.get('filter', 'thresholdmode').lower()})

    return finf, U


def getparam(pfn):
    pfn = Path(pfn).expanduser()

    if not pfn.is_file():
        raise FileNotFoundError(f'{pfn} not found!')

    P = ConfigParser(allow_no_value=True, inline_comment_prefixes=[';'])
    P.read(pfn)

    return P


def keyhandler(keypressed, framebyframe):
    if keypressed == 255:  # no key pressed  (used to be -1)
        return (framebyframe, False)
    elif keypressed == 32:  # space  (used to be 1048608)
        return (not framebyframe, False)
    elif keypressed == 27:  # escape (used to be 1048603)
        return (None, True)
    else:
        print('keypress code: ', keypressed)
        return (framebyframe, False)


def savestat(stat: DataFrame, fn: Path, idir: Path, U: dict):
    assert isinstance(stat, DataFrame)
    print('saving detections & statistics to', fn)

    writemode = 'r+' if fn.is_file() else 'w'

    with h5py.File(fn, writemode, libver='latest') as f:
        f['/input'] = str(idir)
        f['/detect'] = stat['detect'].values.astype(int)  # FIXME: why is type 'O'?

        f['/nfile'] = U['nfile']
        f['framestep'] = U['framestep']
        f['previewDecim'] = U['previewdecim']

        if stat['mean'].nonzero()[0].any():
            f['/mean'] = stat['mean'].values
        if stat['median'].nonzero()[0].any():
            f['/median'] = stat['median'].values
        if stat['variance'].nonzero()[0].any():
            f['/variance'] = stat['variance'].values
