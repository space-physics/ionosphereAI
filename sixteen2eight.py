from __future__ import division
"""
inputs:
-------
I: 2-D Numpy array of grayscale image data
Clim: length 2 of tuple or numpy 1-D array specifying lowest and highest expected values in grayscale image
"""

def sixteen2eight(I,Clim):
    Vmin = Clim[0]; Vmax = Clim[1]
    Q = I.astype('float32').copy() #could also be float64, I did float32 to save RAM, but is is slower than float64 like in Matlab?
    Q[I > Vmax] = Vmax #clip high end
    Q[I < Vmin] = Vmin #boost low end

    Q = (Q - Vmin) / (Vmax - Vmin) #stretch to [0,1]
    return Q.astype('uint8') * 255 # stretch to [0,255]