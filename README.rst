.. image:: https://travis-ci.com/scienceopen/cvhst.svg?token=qSpy37WAwefTyjRcouQS&branch=master
    :target: https://travis-ci.com/scienceopen/cvhst
    
Copyright 2012-2016 Michael Hirsch


======
cv-hst
======

:Author: Michael Hirsch

Computer Vision functions made for working with auroral video


.. contents::

Examples
========
A few common uses:

Process all .AVI in a directory
----------------------------------------
::

    python RunDetectAurora.py ~/mydir -e avi

This will find all the .avi files in directory ~/mydir and play them back with analysis.

Process a specific file
--------------------------------
::

    python RunDetectAurora.py ~/mydir/myvideo.avi

Process DMC sCMOS video
-----------------------
::

    python RunDetectAurora.py "~/X/data/DMC2015-11/2015-11-15" -e .fits -p dmc.xlsx

Hard disk outputs
=================
* PNG figure plot of the number of auroral detections per video frame sampled
* HDF5 file of the data in the PNG plot (so that you can use another program to extract the GB of interesting data from TB file

This program reads any video format available to OpenCV cv2.VideoCapture(), such as most AVI, MPG, MOV, OGV, etc. depending on how your particular OpenCV was compiled.
The program also reads our multi-terabyte .DMCdata video files in a proprietary binary format from the `HiST project <https://github.com/scienceopen/hist-feasibility>`_.

It is trivial to adapt the program to ingest NetCDF, HDF5 and many other formats--just contact me.

Install
=======
::

  python setup.py develop


Note on installing OpenCV 3
===========================
`Install OpenCV 3 on Python 3 <https://scivision.co/anaconda-python-opencv3/>`_

Tested with
------------
* Python 3.5 / 2.7
* OpenCV 3.1 / 2.4
