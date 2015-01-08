from __future__ import division

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
     #clip high end, boost low end

    Q = (Q.clip(Vmin, Vmax) - Vmin) / (Vmax - Vmin) #stretch to [0,1]
    Q *= 255. # stretch to [0,255] as a float
    return Q.round().astype('uint8') # convert to uint8