#!/usr/bin/env python
"""
model for picking how many frames before/after detection to keep
"""
import numpy as np

#x = np.random.uniform(size=100)
#x = x>0.5  # now a boolean coin flip vector

x = np.zeros(20,dtype=bool)
x[[5,6,8,16]] = True

# Lkeep should be ODD in length for symmetry of result
Lkeep = np.ones(5,dtype=bool)   # Retain Lkeep/2-0.5 frames/files before/after first/last detection

y = np.convolve(x,Lkeep,'same')