from __future__ import division

"""
inputs:
-------
I: 2-D Numpy array of grayscale image data
Clim: length 2 of tuple or numpy 1-D array specifying lowest and highest expected values in grayscale image
"""

def normframe(I,Clim):
    Vmin = Clim[0]; Vmax = Clim[1]

    Q = I.astype('float64').copy()

    return (Q.clip(Vmin, Vmax) - Vmin) / (Vmax - Vmin) #stretch to [0,1]