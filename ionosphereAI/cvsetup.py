import logging
from pathlib import Path
import pandas
from typing import Dict, Any, Tuple
from datetime import datetime
import numpy as np

try:
    import cv2
    from cv2 import VideoWriter_fourcc as fourcc
except ImportError:
    cv2 = fourcc = None

try:
    from matplotlib.pylab import figure
    from matplotlib.colors import LogNorm
except (ImportError, RuntimeError):
    figure = LogNorm = None


def setupkern(up: Dict[str, Any]) -> Dict[str, Any]:
    if cv2 is None:
        raise ImportError('OpenCV is needed')

    openrad = up['open_radius']
    if not openrad % 2:
        raise ValueError('openRadius must be ODD')

    up['open'] = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (openrad, openrad))

    up['erode'] = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (openrad, openrad))
    up['close'] = cv2.getStructuringElement(cv2.MORPH_RECT, (up['close_width'], up['close_height']))

    return up


def svsetup(up: Dict[str, Any], finf: Dict[str, Any]) -> Dict[str, Any]:
    savevideo = up['savevideo']
    x = finf['super_x']
    y = finf['super_y']
    pshow = up['pshow']

    if savevideo:
        logging.info(f'dumping video output to {up["odir"]}')
    svh = {'save': savevideo, 'complvl': up['complvl']}
    if savevideo == 'tif':
        # complvl = 6 #0 is uncompressed
        try:
            from tifffile import TiffWriter  # pip install tifffile
        except ImportError:
            logging.error(f'I cannot save iterated video results due to missing tifffile module \n'
                          'try   pip install tifffile \n {e}')
            return svh

        if up.get('wienernhood'):
            svh['wiener'] = TiffWriter(str(up['odir'] / 'wiener.tif'))
        else:
            svh['wiener'] = None

        svh['video'] = TiffWriter(str(up['odir'] / 'video.tif')) if 'rawscaled' in pshow else None
        svh['thres'] = TiffWriter(str(up['odir'] / 'thres.tif')) if 'thres' in pshow else None
        svh['despeck'] = TiffWriter(str(up['odir'] / 'despk.tif')) if 'thres' in pshow else None
        svh['erode'] = TiffWriter(str(up['odir'] / 'erode.tif')) if 'morph' in pshow else None
        svh['close'] = TiffWriter(str(up['odir'] / 'close.tif')) if 'morph' in pshow else None
        # next line makes big file
        svh['detect'] = None  # TiffWriter(join(tdir,'detect.tif')) if showfinal else None

    elif savevideo == 'vid':
        if cv2 is None:
            raise ImportError('OpenCV is needed')

        wfps = up['fps']
        if wfps < 3:
            logging.warning('VLC media player had trouble with video slower than about 3 fps')

        """ if grayscale video, isColor=False
        http://stackoverflow.com/questions/9280653/writing-numpy-arrays-using-cv2-videowriter

        These videos are casually dumped to the temporary directory.
        """
        # cc4 = fourcc(*'FFV1')
        cc4 = fourcc(*'FMP4')
        """
        try 'MJPG' 'XVID' 'FMP4' if FFV1 doesn't work.

        https://github.com/scivision/pyimagevideo
        """
        if up.get('wienernhood'):
            svh['wiener'] = cv2.VideoWriter(str(up['odir'] / 'wiener.avi'), cc4, wfps, (y, x), False)
        else:
            svh['wiener'] = None

        svh['video'] = cv2.VideoWriter(str(up['odir'] / 'video.avi'), cc4, wfps, (y, x), False) if 'rawscaled' in pshow else None
        svh['thres'] = cv2.VideoWriter(str(up['odir'] / 'thres.avi'), cc4, wfps, (y, x), False) if 'thres' in pshow else None
        svh['despeck'] = cv2.VideoWriter(str(up['odir'] / 'despk.avi'), cc4, wfps, (y, x), False) if 'thres' in pshow else None
        svh['erode'] = cv2.VideoWriter(str(up['odir'] / 'erode.avi'), cc4, wfps, (y, x), False) if 'morph' in pshow else None
        svh['close'] = cv2.VideoWriter(str(up['odir'] / 'close.avi'), cc4, wfps, (y, x), False) if 'morph' in pshow else None
        svh['detect'] = cv2.VideoWriter(str(up['odir'] / 'detct.avi'), cc4, wfps, (y, x), True) if 'final' in pshow else None

        for k, v in svh.items():
            try:
                if not v.isOpened():
                    logging.error(f'trouble writing video for {k}')
            except AttributeError:  # not a cv2 object, duck typing
                pass

    return svh


def svrelease(svh, savevideo: str):
    try:
        if savevideo == 'tif':
            for k, v in svh.items():
                if v is not None:
                    v.close()
        elif savevideo == 'vid':
            for k, v in svh.items():
                try:
                    v.release()
                except AttributeError:
                    pass
    except Exception as e:
        logging.error(e)


def setupof(U: Dict[str, Any],
            finf: Dict[str, Any]) -> Tuple[np.ndarray, Any]:

    gmm = None
    lastflow = None  # if it stays None, signals to use GMM
    if not isinstance(U['ofmethod'], str):
        raise TypeError('expected type str for ofmethod')
    if U['ofmethod'] == 'hs':
        pass
    elif U['ofmethod'] == 'farneback':
        lastflow = np.zeros((finf['super_y'], finf['super_x'], 2))
# %% GMM
    elif U['ofmethod'] == 'mog':
        if cv2 is None:
            raise ImportError('OpenCV is needed')
        # http://docs.opencv.org/3.2.0/d7/d7b/classcv_1_1BackgroundSubtractorMOG2.html
        gmm = cv2.createBackgroundSubtractorMOG2(history=U['gmm_nhistory'],
                                                 varThreshold=U['gmm_varthreshold'],
                                                 detectShadows=False)
        gmm.setNMixtures(U['gmm_nmixtures'])
        gmm.setComplexityReductionThreshold(U['gmm_compresthres'])
    elif U['ofmethod'] == 'knn':
        if cv2 is None:
            raise ImportError('OpenCV is needed')
        gmm = cv2.createBackgroundSubtractorKNN(history=U['gmm_nhistory'],
                                                detectShadows=True)
    elif U['ofmethod'] == 'gmg':
        if cv2 is None:
            raise ImportError('OpenCV is needed')

        try:
            gmm = cv2.createBackgroundSubtractorGMG(initializationFrames=U['gmm_nhistory'])
        except AttributeError as e:
            raise ImportError(f'GMG is part of opencv_contrib. {e}')
    else:
        raise TypeError(f'unknown method {U["ofmethod"]}')

    return lastflow, gmm


def setupfigs(finf: Dict[str, Any],
              fn: Path,
              U: Dict[str, Any]) -> Tuple[Dict[str, Any], pandas.DataFrame]:
    # %% optical flow magnitude plot

    if (figure is not None and
        'threscolor' in U['pshow'] and
            U.get('ofmethod') in ('hs', 'farneback')):

        fg = figure()
        axom = fg.gca()
        hiom = axom.imshow(np.zeros((finf['super_y'], finf['super_x'])),
                           vmin=1e-5, vmax=1,  # arbitrary limits
                           origin='top',  # origin=top like OpenCV
                           norm=LogNorm())  # cmap=lcmap)
        axom.set_title('optical flow magnitude')
        fg.colorbar(hiom, ax=axom)
        U['iofm'] = hiom
# %% stat plot
    try:
        dt = [datetime.utcfromtimestamp(t) for t in finf['ut1'][:-1]]
        ut = finf['ut1'][:-1]
    except (TypeError, KeyError):
        dt = ut = finf['frameind'][:-1]

    stat = pandas.DataFrame(index=ut, columns=['mean', 'median', 'variance', 'detect'])
    stat['detect'] = np.zeros(finf['frameind'].size-1, dtype=int)
    stat[['mean', 'median', 'variance']] = np.zeros((finf['frameind'].size-1, 3), dtype=float)

    hpmn, hpmd, hpdt, fgdt = statplot(dt, stat, U, fn)

#    draw(); pause(0.001) #catch any plot bugs

    U.update({'pmean': hpmn, 'pmed': hpmd, 'pdet': hpdt, 'fdet': fgdt})

    return U, stat


def statplot(dt, stat: pandas.DataFrame, U: Dict[str, Any], fn: Path):
    if figure is None:
        raise ImportError('pip install matplotlib')
    hpmn = None
    hpmd = None
    hpdt = None
    fg = None

    def _timelbl(ax, x, y, lbl=None):
        if x is None:
            hpl = ax.plot(stat.index, y, label=lbl)
            ax.set_xlabel('frame index #')
        elif isinstance(x[0], (int, np.int64)):
            hpl = ax.plot(x, y, label=lbl)
            ax.set_xlabel('Spool File index # (row of index.h5)')
        elif isinstance(x[0], (datetime, np.datetime64)):
            hpl = ax.plot(x, y, label=lbl)
            ax.set_xlabel('Time [UTC]')
        else:
            hpl = None

        return hpl

    if 'stat' in U['pshow']:

        if U.get('ofmethod') in ('hs', 'farneback'):
            Np = 2
            fg = figure(figsize=(12, 5))
            ax = fg.add_subplot(Np, 1, 1)
            ax.set_title('optical flow statistics')
            ax.set_xlabel('frame index #')
            ax.set_ylim((0, 0.1))

            hpmn = _timelbl(ax, dt, stat['mean'],  'mean')
            hpmd = _timelbl(ax, dt, stat['median'], 'median')
            ax.legend(loc='best')
# %% detections
        else:
            fg = figure()
            Np = 1

        ax = fg.add_subplot(Np, 1, Np)
        ax.set_title(f'Detections of Aurora {fn.name}')
        ax.set_ylabel('number of detections')
        ax.set_ylim((0, 10))

        hpdt = _timelbl(ax, dt, stat['detect'])

    return hpmn, hpmd, hpdt, fg
