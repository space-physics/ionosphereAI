cv-hst
======
Computer Vision functions made for working with HiST optical data

```detectaurora.py``` is the main program, the others are only helper functions

Usage:
------
python detectaurora.py ~/directoryOfVideos/optionalSpecificFilename(s)

This program will process several video files if given a directory, or specific file(s) if given the filename(s)
The outputs include:
* PNG figure plot of the number of auroral detections per video frame sampled
* HDF5 file of the data in the PNG plot (so that you can use another program to extract the GB of interesting data from TB file

This program is specifically designed for .DMCdata video files in a proprietary binary format from the HiST project. It is trivial to adapt the program to ingest AVI, NetCDF, HDF5 and many other formats--just contact me.

Prerequisities:
---------------
Python 3.4 or 2.7 with OpenCV

### Anaconda Python 2.7 
```
conda install matplotlib opencv h5py pandas xlrd
```

### Anaconda Python 3.4
see http://blogs.bu.edu/mhirsch/2015/03/anaconda-python-opencv3/

Installation:
-------------
```
mkdir ~/code
cd ~/code
git clone https://github.com/scienceopen/cv-hst.git
git clone https://github.com/scienceopen/hist-utils.git
```

Thanks:
-------
Many thanks to Amber Baurley for testing and improving the original, messy Matlab code I came up with.

Tested with:
------------
Python 3.4.3 / 2.7.9 

OpenCV 3.0 beta / 2.4.{8-10}

Numpy 1.9.2

Matplotlib 1.4.3

Pandas 0.15.2

h5py 2.4.0
