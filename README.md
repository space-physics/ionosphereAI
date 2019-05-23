[![image](https://zenodo.org/badge/DOI/10.5281/zenodo.168226.svg)](https://doi.org/10.5281/zenodo.168226)
[![image](https://travis-ci.org/scivision/ionosphereAI.svg?branch=master)](https://travis-ci.org/scivision/ionosphereAI)
[![image](https://coveralls.io/repos/github/scivision/ionosphereAI/badge.svg?branch=master)](https://coveralls.io/github/scivision/ionosphereAI?branch=master)

# Ionosphere AI

CV / ML / AI for working with auroral video, passive FM radar and more

## Examples

A few common uses:

### Incoherent Scatter Radar

Using [raw ISR data](https://github.com/scivision/isrutils):

    python Detect.py ~/data/2013-05-01/isr -e .dt3.h5

### Process all .AVI in a directory

    python Detect.py ~/mydir -e avi

This will find all the .avi files in directory \~/mydir and play them
back with analysis.

### Process a specific file

    python Detect.py ~/mydir/myvideo.avi

### DMC sCMOS video

    python Detect.py "~/data/DMC2015-11/2015-11-15" -e .fits -p dmc.ini

## Hard disk outputs

-   PNG figure plot of the number of auroral detections per video frame sampled
-   HDF5 file of the data in the PNG plot (so that you can use another
    program to extract the GB of interesting data from TB file

This program reads any video format available to OpenCV cv2.VideoCapture(), such as most AVI, MPG, MOV, OGV, etc. depending on how your particular OpenCV was compiled.
The program also reads our
multi-terabyte .DMCdata video files in a proprietary binary format from
the [HiST project](https://github.com/scivision/hist-feasibility).

It is trivial to adapt the program to ingest NetCDF, HDF5 and many other
formats--just contact me.

## Install

1. [Install OpenCV on Python](https://scivision.co/anaconda-python-opencv3/)
2. install this program:
   ```sh
   python -m pip install -e .
   ```
