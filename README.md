[![image](https://zenodo.org/badge/DOI/10.5281/zenodo.168226.svg)](https://doi.org/10.5281/zenodo.168226)

[![Build Status](https://travis-ci.org/space-physics/ionosphereAI.svg?branch=master)](https://travis-ci.org/space-physics/ionosphereAI)
[![Coverage Status](https://coveralls.io/repos/github/space-physics/ionosphereAI/badge.svg?branch=master)](https://coveralls.io/github/space-physics/ionosphereAI?branch=master)
[![Build status](https://ci.appveyor.com/api/projects/status/w2vi0awovp9e1t4r?svg=true)](https://ci.appveyor.com/project/scivision/ionosphereai)

# Ionosphere AI

Machine learning and computer vision techniques for auroral video, passive FM radar, incoherent scatter radar and other geoscience data using collective behavior detection.
The programs are OS/platform-agnostic.


## Examples


### Incoherent Scatter Radar

Using [raw ISR data](https://github.com/space-physics/isrutils):

    python Detect.py ~/data/2013-05-01/isr -e .dt3.h5

### Process all .AVI in a directory

    python Detect.py ~/mydir -e avi

This will find all the .avi files in directory ~/mydir and play them
back with analysis.

### Process a specific file

    python Detect.py ~/mydir/myvideo.avi

### DMC sCMOS video

    python Detect.py ~/data/DMC2015-11/2015-11-15 -e .fits -p dmc.ini

## Hard disk outputs

* PNG figure plot of the number of auroral detections per video frame sampled
* HDF5 file of the data in the PNG plot (so that you can use another program to extract the GB of interesting data from TB file

This program reads any video format available to OpenCV
[cv2.VideoCapture()](https://docs.opencv.org/master/d8/dfe/classcv_1_1VideoCapture.html) and
[imageio.imread()](https://imageio.readthedocs.io/en/stable/userapi.html#imageio.imread)
-- essentially anything FFmpeg can read, such as most AVI, MPG, MOV, OGV, etc. depending on how your particular OpenCV was compiled.
The program also reads our multi-terabyte .DMCdata video files in a proprietary binary format from the
[HiST project](https://github.com/space-physics/histfeasf).

It is usually straightforward to adapt the program to ingest NetCDF, HDF5 and many other formats.

## Install

```sh
python -m pip install -e .[opencv,cv]
```

If you don't have OpenCV:

```sh
python -m pip install opencv-python
```

or

```sh
conda install opencv
```
