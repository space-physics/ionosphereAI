import numpy as np
from typing import Tuple
import logging


def sixteen2eight(I: np.ndarray, Clim: Tuple[int, int]) -> np.ndarray:
    """
    scipy.misc.bytescale had bugs

    inputs:
    ------
    I: 2-D Numpy array of grayscale image data
    Clim: length 2 of tuple or numpy 1-D array specifying lowest and highest expected values in grayscale image
    Michael Hirsch, Ph.D.
    """
    Q = normframe(I, Clim)
    Q *= 255  # stretch to [0,255] as a float
    return Q.round().astype(np.uint8)  # convert to uint8


def normframe(I: np.ndarray, Clim: tuple) -> np.ndarray:
    """
    inputs:
    -------
    I: 2-D Numpy array of grayscale image data
    Clim: length 2 of tuple or numpy 1-D array specifying lowest and highest expected values in grayscale image
    """
    Vmin = Clim[0]
    Vmax = Clim[1]

    # stretch to [0,1]
    return (I.astype(np.float32).clip(Vmin, Vmax) - Vmin) / (Vmax - Vmin)


def saturation_check(frame: np.ndarray,
                     minmaxcount: Tuple[int, int],
                     minmax: Tuple[int, int] = (0, 255)) -> bool:
    """
    Check for excessive saturation of 8-bit image

    Parameters
    ----------

    frame: np.ndarray of uint8
        image data downscaled from 16-bit
    minmaxcount: tuple of int
        count of minimum, maximum values that are OK to saturate e.g. for stars
    minmax: tuple of int
        minimum, maximum values that are OK

    Results
    -------
    bad: bool
        too many pixels saturated
    """
    bad = False

    if (frame == minmax[1]).sum() > minmaxcount[1]:
        logging.warning('video saturated at 255')
        bad = True

    if (frame == minmax[0]).sum() > minmaxcount[0]:
        logging.warning('video saturated at 0')
        bad = True

    return bad
