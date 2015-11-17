(c) Copyright 2012-2015 Michael Hirsch

This is non-public code. Do not publish or share this code without prior written approval from Michael Hirsch.

======
cv-hst
======

:Author: Michael Hirsch

Computer Vision functions made for working with auroral video


.. contents::

Usage
=====
A few common uses:

Load and process all .AVI in a directory
----------------------------------------
::

    python RunDetectAurora.py ~/mydir -e avi

This will find all the .avi files in directory ~/mydir and play them back with analysis.

Load and process a specific file
--------------------------------
::

    python RunDetectAurora.py ~/mydir/myvideo.avi

Hard disk outputs
-----------------
* PNG figure plot of the number of auroral detections per video frame sampled
* HDF5 file of the data in the PNG plot (so that you can use another program to extract the GB of interesting data from TB file

This program reads any video format available to OpenCV cv2.VideoCapture(), such as most AVI, MPG, MOV, OGV, etc. depending on how your particular OpenCV was compiled.
The program also reads our multi-terabyte .DMCdata video files in a proprietary binary format from the `HiST project <https://github.com/scienceopen/hist-feasibility>`_.

It is trivial to adapt the program to ingest NetCDF, HDF5 and many other formats--just contact me.

Install
=======
::

  git clone --depth 1 https://github.com/scienceopen/cvhst.git
  python setup.py develop

`If using Python 3 <https://scivision.co/anaconda-python-opencv3/>`_

Tested with
------------
* Python 3.5 / 3.4 / 2.7
* OpenCV 3.0 / 2.4
