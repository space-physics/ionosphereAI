from __future__ import division
from numpy import around
#from pdb import set_trace
"""
inputs:
-------
I: 2-D Numpy array of grayscale image data
Clim: length 2 of tuple or numpy 1-D array specifying lowest and highest expected values in grayscale image
"""

def sixteen2eight(I,Clim):

    Vmin = Clim[0]; Vmax = Clim[1]
    Q = I.astype('float64').copy()
    Q[I > Vmax] = Vmax #clip high end
    Q[I < Vmin] = Vmin #boost low end

    Q = (Q - Vmin) / (Vmax - Vmin) #stretch to [0,1]
    Q *= 255. # stretch to [0,255] as a float
    return around(Q).astype('uint8') # convert to uint8