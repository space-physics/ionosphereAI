cv-hst
======
Computer Vision functions made for working with HiST optical data

```detectaurora.py``` is the main program, the others are only helper functions

Usage:
python detectaurora.py ~/directoryOfVideos/optionalSpecificFilename(s)

This program will process several video files if given a directory, or specific file(s) if given the filename(s)
The outputs include:
* PNG figure plot of the number of auroral detections per video frame sampled
* HDF5 file of the data in the PNG plot (so that you can use another program to extract the GB of interesting data from TB file

This program is specifically designed for .DMCdata video files in a proprietary binary format from the HiST project. It is trivial to adapt the program to ingest AVI, NetCDF, and many other formats--just email mhirsch@bu.edu.

Prerequisities:
Python 2.7, suggest Anaconda or Miniconda for ease of installing prereqs, but any Python 2.7 should be fine on any operating system.
```conda install matplotlib opencv h5py pandas xlrd```

Installation:
```
mkdir ~/code
cd ~/code
git clone https://github.com/scienceopen/cv-hst.git
git clone https://github.com/scienceopen/hist-utils.git
```

Many thanks to Amber Baurley for testing and improving the original, messy Matlab code I came up with.

Tested with:
Python 2.7.8 (Anaconda)
OpenCV 2.4.10
Numpy 1.9.1
Matplotlib 1.4.2
Pandas 0.15.2
h5py 2.4.0
