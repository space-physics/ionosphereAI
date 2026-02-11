import logging
import h5py
import numpy as np
from typing import Any
from pathlib import Path
from scipy.signal import wiener
from .getpassivefm import getfmradarframe
from .utils import sixteen2eight
# import fitsio  # so much faster than Astropy.io.fits

try:
    import cv2
except ImportError as e:
    logging.debug(e)
    cv2 = None

try:
    from matplotlib.pyplot import figure, hist
except (ImportError, RuntimeError) as e:
    logging.debug(e)
    figure = hist = None


try:
    import imageio  # tifffile is excruciatingly slow on each file access
except ImportError as e:
    logging.debug(e)
    pass

try:
    from dmcutils.neospool import readNeoSpool
except ImportError as e:
    logging.debug(e)
    readNeoSpool = None

try:
    from histutils.rawDMCreader import getDMCframe
except ImportError as e:
    logging.debug(e)
    getDMCframe = None

try:
    from astropy.io import fits
except ImportError as e:
    logging.debug(e)
    fits = None


def setscale(fn: Path,
             up: dict[str, Any],
             finf: dict[str, Any]) -> dict[str, Any]:
    """
    if user has not set fixed upper/lower bounds for 16-8 bit conversion, do it automatically,
    since not specifying fixed contrast destroys the working of auto CV detection
    """
    pt = [0.01, 0.99]  # percentiles
    mod = False

    if not isinstance(up['rawlim'][0], (float, int)):
        mod = True
        prc = samplepercentile(fn, pt[0], up, finf)
        up['rawlim'][0] = prc

    if not isinstance(up['rawlim'][1], (float, int)):
        mod = True
        prc = samplepercentile(fn, pt[1], up, finf)
        up['rawlim'][1] = prc

# %%
    if mod:
        cdiff = up['rawlim'][1] - up['rawlim'][0]

        assert cdiff > 0, f'raw limits do not make sense  lower: {up["rawlim"][0]}   upper: {up["rawlim"][1]}'

        if cdiff < 20:
            raise ValueError('your video may have very poor contrast and not work for auto detection')

        print(f'data number lower,upper  {up["rawlim"][0]}  {up["rawlim"][1]}')

    return up


def samplepercentile(fn: Path,
                     pct: float,
                     up: dict[str, Any],
                     finf: dict[str, Any]):
    """
    for 16-bit files mainly
    """
    isamp = (finf['nframe'] * np.array([.1, .25, .5, .75, .9])).astype(int)  # pick a few places in the file to touch

    isamp = isamp[:finf['nframe']-1]  # for really small files

    dat = np.empty((isamp.size, finf['supery'], finf['superx']), float)

    tmp = get_frames(fn, ifrm=0, finf=finf, up=up)
    if tmp.dtype.itemsize < 2:
        logging.warning(f'{fn}: usually we use autoscale with 16-bit video, not 8-bit.')

    for j, i in enumerate(isamp):
        dat[j, ...] = get_frames(fn, ifrm=i, finf=finf, up=up)

    return np.percentile(dat, pct).astype(int)


def get_frames(fn: Any,
               ifrm: int,
               finf: dict[str, Any],
               up: dict[str, Any],
               svh: dict[str, Any] = {}, *,
               ifits: int = None):
    """
    this function reads the reference frame too--which makes sense if you're
       only reading every Nth frame from the multi-TB file instead of every frame
    """
# %% reference frame
    reader = finf.get('reader')

    if reader == 'raw':
        frame = read_dmc(fn, ifrm, up['twoframe'], finf)
    elif reader == 'spool':
        frame = read_spool(fn, ifrm, up['twoframe'], finf, up.get('zerocols', 0))
    elif reader == 'h5vid':
        frame = read_hdf(fn, ifrm, up['twoframe'])
    elif reader == 'h5fm':
        frame = read_h5fm(fn, ifrm, up['twoframe'])
    elif reader == 'fits':
        frame = read_fits(fn, ifits, up['twoframe'])  # ifits not ifrm
    elif reader == 'tiff':
        frame = read_tiff(fn, ifrm, up['twoframe'])
    elif reader == 'cv2':
        frame = read_cv2(finf['h_read'], up['twoframe'])
    else:
        frame = read_cv2(finf['h_read'], up['twoframe'])
# %% current frame
#    if 'rawframeind' in up:
#        up['rawframeind'][ifrm] = rfi

    if up.get('wienernhood') and up['twoframe']:
        frame = wiener(frame, up['wienernhood'])

    # image histograms (to help verify proper scaling to uint8)
    if 'hist' in up.get('pshow', []):
        ax = figure().gca()
        hist(frame.flatten(), bins=128, fc='w', ec='k', log=True)
        ax.set_title('raw uint16 values')

# %% scale to 8bit
    if reader != 'cv2' and up.get('rawlim'):
        frame = sixteen2eight(frame, up['rawlim'])

    if 'hist' in up.get('pshow', []):
        ax = figure().gca()
        hist(frame[0, :, :].flatten(), bins=128, fc='w', ec='k', log=True)
        ax.set_xlim((0, 255))
        ax.set_title('normalized video into opt flow')

    if 'raw' in up.get('pshow', []):
        # cv2.imshow just divides by 256, NOT autoscaled!
        # http://docs.opencv.org/modules/highgui/doc/user_interface.html
        cv2.imshow('video', frame[0, :, :])
# %% plotting
    if 'rawscaled' in up.get('pshow', []):
        cv2.imshow('raw video, scaled to 8-bit', frame[0, :, :])

    if svh.get('video'):
        if up['savevideo'] == 'tif':
            svh['video'].save(frame[0, :, :], compress=up['complvl'])
        elif up['savevideo'] == 'vid':
            svh['video'].write(frame[0, :, :])

    return frame


def read_dmc(fn: Path, ifrm: int, twoframe: bool, finf: dict[str, Any]):
    if getDMCframe is None:
        raise ImportError('pip install histutils')

    if twoframe:
        frame0 = getDMCframe(fn, ifrm, finf)[0]
    frame, iraw = getDMCframe(fn, ifrm+1, finf)

    if twoframe:
        frame = np.stack((frame0, frame), axis=0)

    return frame


def read_fits(fn: Path, ifrm: int, twoframe: bool):
    """
    ifits not ifrm for fits!
    """
    if fits is None:
        raise ImportError('Need Astropy for FITS')
    # memmap = False required thru at least Astropy 1.3.2 due to BZERO used...
    with fits.open(fn, mode='readonly', memmap=False) as f:
        if twoframe:
            frame = f[0][ifrm:ifrm+2, :, :]
        else:
            frame = f[0][ifrm+1, :, :]

    return frame


def read_h5fm(files: list[Path], ifrm: int, twoframe: bool):
    """
      one frame per file
    """
    if twoframe:
        frame0 = getfmradarframe(files[ifrm])[2]
    frame = getfmradarframe(files[ifrm+1])[2]

    if twoframe:
        frame = np.stack((frame0, frame), axis=0)

    return frame


def read_hdf(fn: Path, ifrm: int, twoframe: bool):

    with h5py.File(fn, 'r') as f:
        if twoframe:
            frame = f['/rawimg'][ifrm:ifrm+2, ...]
        else:
            frame = f['/rawimg'][ifrm+1, ...]

    return frame


def read_spool(fn: Path, ifrm: int, twoframe: bool,
               finf: dict[str, Any], zerocols: int):

    """
    Read only the first frame pair from each spool file,
    as each spool file is generally less than about 10 frames.
    To skip further in time, skip more files.
    """
    iread = (ifrm, ifrm+1) if twoframe else ifrm+1

    frames, ticks, tsec = readNeoSpool(fn, finf, iread, zerocols=zerocols)

    if twoframe:
        frame = frames[:2, :, :]
    else:
        frame = frames[0, :, :]

    return frame


def read_cv2(h, twoframe: bool):
    """
    uses ImageIO to read video and cv2 to scale--could use non-cv2 method to scale.

    h is handle from imageio.get_reader()
    """
    if cv2 is None:
        raise ImportError('FIXME: can we use non-opencv rgb2gray?')
    if twoframe:
        frame0 = h.read()

        if frame0.ndim > 2:
            frame0 = cv2.cvtColor(frame0, cv2.COLOR_RGB2GRAY)

    frame16 = h.read()  # NOTE: this is skipping every other frame

    if frame16.ndim > 2:
        frame = cv2.cvtColor(frame16, cv2.COLOR_RGB2GRAY)
    else:
        frame = frame16  # copy NOT needed

    if twoframe:
        frame = np.stack((frame0, frame), axis=0)

    return frame


def read_tiff(fn: Path, ifrm: int, twoframe: bool):
    """
    TODO: do we pass in ifrm or do we open a handle like read_cv2
    """

    if twoframe:
        frame0 = imageio.imread(fn, ifrm)

    frame = imageio.imread(fn, ifrm+1)

    if twoframe:
        frame = np.stack((frame0, frame), axis=0)

    return frame
