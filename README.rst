======
cv-hst
======
Computer Vision functions made for working with auroral video

Usage
------
* Load and process all .AVI in a directory::

    python detectaurora.py ~/mydir --ext avi

The line above will find all the .avi files in directory ~/mydir and play them back with analysis.

* Load and process a specific file::

    python detectaurora.py ~/mydir/myvideo.avi

The line above will read a specific file(s)

-------------

Program disk outputs include:

* PNG figure plot of the number of auroral detections per video frame sampled
* HDF5 file of the data in the PNG plot (so that you can use another program to extract the GB of interesting data from TB file

This program reads any video format available to OpenCV cv2.VideoCapture(), such as most AVI, MPG, MOV, OGV, etc. depending on how your particular OpenCV was compiled.
The program also reads our multi-terabyte .DMCdata video files in a proprietary binary format from the `HiST project <https://github.com/scienceopen/hist-feasibility>`_.

It is trivial to adapt the program to ingest NetCDF, HDF5 and many other formats--just contact me.

Install
---------------
from Terminal::

  git clone --recursive --depth 1 https://github.com/scienceopen/cv-hst.git
  conda install --file requirements.txt
  pip install -r piprequirements.txt

If using Python 3, see https://scivision.co/anaconda-python-opencv3/

Tested with
------------
Python 3.4 / 2.7

OpenCV 3.0 / 2.4
