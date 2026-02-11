from configparser import ConfigParser
from pathlib import Path
from typing import Any
import logging

from pandas import DataFrame
import h5py
import numpy as np
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


def get_file_info(files: list[Path],
                  U: dict[str, Any]) -> dict[str, Any]:

    # keeping type hints consistent
    if isinstance(files, Path):
        files = [files]
    fn = files[0]

    if fn.suffix.lower() == '.dmcdata':  # HIST
        finf = read_dmc(fn, U)
    elif fn.suffix.lower() == '.dat':  # Andor Solis spool file
        finf = read_spool(fn, U, {})
    elif fn.suffix.lower() in {'.h5', '.hdf5'}:
        finf = {}
        try:  # can't read inside context
            with h5py.File(fn, 'r') as f:
                finf['flist'] = f['fn'][:]
            # finf['flist'] = read_hdf(fn,'filetick')
            if U['startstop'] is not None:
                finf['flist'] = finf['flist'][U['startstop'][0]:U['startstop'][1]]

            logging.info(f'taking {len(finf["flist"])} files from index {fn}')
        except KeyError:
            pass
# %% determine if optical or passive radar
        with h5py.File(fn, 'r') as f:
            if 'rawimg' in f:  # hst image/video file
                finf = {'reader': 'h5vid'}
                finf['nframe'] = f['rawimg'].shape[0]
                finf['super_x'] = f['rawimg'].shape[2]
                finf['super_y'] = f['rawimg'].shape[1]
                # print('HDF5 video file detected {}'.format(fn))
            elif 'ambiguity' in f:  # Haystack passive FM radar file
                finf = {'reader': 'h5fm'}
                finf['nframe'] = 1  # currently the passive radar uses one file per frame
                range_km, vel_mps = getfmradarframe(fn)[:2]  # assuming all frames are the same size
                finf['super_x'] = range_km.size
                finf['super_y'] = vel_mps.size
                # print('HDF5 passive FM radar file detected {}'.format(fn))
            elif 'ticks' in f:  # Andor Solis spool file index from dmcutils/Filetick.py
                finf = read_spool(fn, U, finf)
                finf['path'] = fn.parent
            else:
                raise ValueError(f'{fn}: unknown input HDF5 file type')

        if 'frameind' not in finf:
            finf['frameind'] = np.arange(0, finf['nframe'], U['framestep'], dtype=int)
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
        finf = read_cv2(fn)

    return finf


def read_cv2(fn: Path) -> dict[str, Any]:
    if getaviprop is None:
        raise ImportError('pip install morecvutils')
    # TODO start,stop,step is not yet implemented, simply uses every other frame
    logging.info(f'attempting to read {fn} with OpenCV.')

    vidparam = getaviprop(fn)

    finf = {'reader': 'cv2',
            'nframe': vidparam['nframe']}

    finf['super_x'], finf['super_y'] = vidparam['xy_pixel']

    finf['frameind'] = np.arange(finf['nframe'], dtype=int)

    finf['h_read'] = imageio.get_reader(f'imageio:{fn}')

    return finf


def read_dmc(fn: Path, U: dict[str, Any]) -> dict[str, Any]:
    if getDMCparam is None:
        raise ImportError('pip install histutils')

    if not U.get('startstop'):
        U['frame_request'] = U.get('framestep', 1)
    elif len(U['startstop']) == 2:
        U['frame_request'] = (U['startstop'][0], U['startstop'][1], U['framestep'])
    else:
        raise ValueError('unknown start, stop, step frame request')

    finf = getDMCparam(fn, U)

    finf['reader'] = 'raw'
    finf['nframe'] = finf['nframeextract']
    finf['frameind'] = finf['frameindrel']

    return finf


def read_spool(fn: Path,
               U: dict[str, Any],
               f0: dict[str, Any]) -> dict[str, Any]:
    if spoolparam is None:
        raise ImportError('pip install dmcutils')

    finf = spoolparam(fn.parent/SPOOLINI,
                      U['super_x'], U['super_y'],
                      U.get('cmos_stride', 0))
    finf = {**finf, **f0}

    print('using spool file, config', fn.parent/SPOOLINI)
    finf['reader'] = 'spool'
    finf['nframe'] = U['nfile']  # first frame pair of each spool file

    # FIXME should we make this general to all file types?
    if U['nfile'] > 1 and finf['nframe'] > 10 * U['framestep']:
        finf['frameind'] = np.arange(0, finf['nframe'], U['framestep'], dtype=int)
    else:  # revert to all frames because there aren't many, rather than annoying with zero result
        finf['frameind'] = np.arange(finf['nframe'], dtype=int)
    finf['kinetic'] = None  # FIXME blank for now

    finf['path'] = fn.parent

    return finf


def get_sensor_config(fn: Path) -> ConfigParser:
    fn = Path(fn).expanduser()
    if not fn.is_file():
        raise FileNotFoundError(fn)

    P = ConfigParser(allow_no_value=True, inline_comment_prefixes=[';'])
    P.read(fn)

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


def savestat(stat: DataFrame, fn: Path, idir: Path, U: dict[str, Any]):
    assert isinstance(stat, DataFrame)
    print('saving detections & statistics to', fn)

    writemode = 'r+' if fn.is_file() else 'w'

    with h5py.File(fn, writemode) as f:
        f['/input'] = str(idir)
        f['/detect'] = stat['detect'].values.astype(int)  # FIXME: why is type 'O'?

        f['/nfile'] = U['nfile']
        f['framestep'] = U['framestep']
        f['previewDecim'] = U['previewdecim']

        if stat['mean'].to_numpy().nonzero()[0].any():
            f['/mean'] = stat['mean'].values
        if stat['median'].to_numpy().nonzero()[0].any():
            f['/median'] = stat['median'].values
        if stat['variance'].to_numpy().nonzero()[0].any():
            f['/variance'] = stat['variance'].values
