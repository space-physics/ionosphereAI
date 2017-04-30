#!/usr/bin/env python
"""
model for picking how many frames before/after detection to keep
"""
import numpy as np

#x = np.random.uniform(size=100)
#x = x>0.5  # now a boolean coin flip vector

x = np.zeros(20,dtype=int)
x[[5,6,8]] = [1,5,1]  # this method give equal weighting regardless of number of hits
x[16] = 1  # isolated strike

# Lkeep should be ODD in length for symmetry of result
# FIXME use trapezoid or triangle to emphasize groups of detections even more.
Lkeep = np.ones(5)   # Retain Lkeep/2-0.5 frames/files before/after first/last detection

y = np.convolve(x,Lkeep,'same')#.astype(bool)