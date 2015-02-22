# skeltonizing using Python


from skimage.morphology import skeletonize
from skimage import draw
import numpy as np
import matplotlib.pyplot as plt
from oct2py import octave




data = octave.call('RunGenOFtestPattern',0,'','horizslide','vertbar',12,256,256,0.5,0.5,1,8)

print(data.shape)
