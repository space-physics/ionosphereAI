cv-hst
======
Computer Vision functions made for working with auroral video

Usage:
------
#### Load and process all .AVI in a directory
``` 
python detectaurora.py ~/mydir --ext avi 
```
The line above will find all the .avi files in directory ~/mydir and play them back with analysis.

#### Load and process a specific fle
``` 
python detectaurora.py ~/mydir/myvideo.avi
```
The line above will read a specific file(s)

This program will process several video files if given a directory, or specific file(s) if given the filename(s)
The outputs include:
* PNG figure plot of the number of auroral detections per video frame sampled
* HDF5 file of the data in the PNG plot (so that you can use another program to extract the GB of interesting data from TB file

This program reads any video format available to OpenCV cv2.VideoCapture(), such as most AVI, MPG, MOV, OGV, etc. depending on how your particular OpenCV was compiled. 
The program also reads our multi-terabyte .DMCdata video files in a proprietary binary format from the [HiST project](https://github.com/scienceopen/hist-feasibility).

It is trivial to adapt the program to ingest NetCDF, HDF5 and many other formats--just contact me.

Prerequisities:
---------------
Python 3.4 or 2.7 with OpenCV

```
pip install -r requirements.txt
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

OpenCV 3.0 beta / 2.4.10

Numpy 1.9.2

Matplotlib 1.4.3

Pandas 0.15.2

h5py 2.4.0
