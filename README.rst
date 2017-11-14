.. image:: https://zenodo.org/badge/DOI/10.5281/zenodo.168226.svg
   :target: https://doi.org/10.5281/zenodo.168226

=============
cv-ionosphere
=============

:Author: Michael Hirsch

Computer Vision functions made for working with auroral video


.. contents::

Examples
========
A few common uses:

Process Incoherent Scatter Radar data
-------------------------------------
Using `raw ISR data <https://github.com/scivision/isrutils>`_::

    python Detect.py ~/data/2013-05-01/isr -e .dt3.h5

Process all .AVI in a directory
---------------------------------
::

    python Detect.py ~/mydir -e avi

This will find all the .avi files in directory ~/mydir and play them back with analysis.

Process a specific file
--------------------------------
::

    python Detect.py ~/mydir/myvideo.avi

Process DMC sCMOS video
-----------------------
::

    python Detect.py "~/data/DMC2015-11/2015-11-15" -e .fits -p dmc.ini

Hard disk outputs
=================
* PNG figure plot of the number of auroral detections per video frame sampled
* HDF5 file of the data in the PNG plot (so that you can use another program to extract the GB of interesting data from TB file

This program reads any video format available to OpenCV cv2.VideoCapture(), such as most AVI, MPG, MOV, OGV, etc. depending on how your particular OpenCV was compiled.
The program also reads our multi-terabyte .DMCdata video files in a proprietary binary format from the `HiST project <https://github.com/scivision/hist-feasibility>`_.

It is trivial to adapt the program to ingest NetCDF, HDF5 and many other formats--just contact me.

Install
=======
`Install OpenCV on Python <https://scivision.co/anaconda-python-opencv3/>`_
::

  pip install -e .

