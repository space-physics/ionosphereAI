# Ionosphere AI

[![image](https://zenodo.org/badge/DOI/10.5281/zenodo.168226.svg)](https://doi.org/10.5281/zenodo.168226)
[![ci](https://github.com/space-physics/ionosphereAI/actions/workflows/ci.yml/badge.svg)](https://github.com/space-physics/ionosphereAI/actions/workflows/ci.yml)


Machine learning and computer vision techniques for auroral video, passive FM radar, incoherent scatter radar and other geoscience data using collective behavior detection.
The programs are OS/platform-agnostic.


## Examples


### Incoherent Scatter Radar

Using [raw ISR data](https://github.com/space-physics/isrutils):

```sh
python -m ionosphereAI.detect ~/data/2013-05-01/isr -e .dt3.h5
```

### Process all .AVI in a directory

```sh
python -m ionosphereAI.detect ~/mydir ~~/mydetections
```

This will find all the .avi files in directory ~/mydir and play them
back with analysis.

### Process a specific file

```sh
python -m ionosphereAI.detect ~/mydir/myvideo.avi
```

### DMC sCMOS video

```sh
python -m ionosphereAI.detect ~/data/DMC2015-11/2015-11-15 ~/data/detect_dmc dmc-gmm.ini
```

## Hard disk outputs

* PNG figure plot of the number of auroral detections per video frame sampled
* HDF5 file of the data in the PNG plot so that you can use another program to extract the GB of interesting data from TB file

This program reads any video format available to OpenCV
[cv2.VideoCapture()](https://docs.opencv.org/master/d8/dfe/classcv_1_1VideoCapture.html) and
[imageio.imread()](https://imageio.readthedocs.io/en/stable/userapi.html#imageio.imread)
-- essentially anything FFmpeg can read, such as most AVI, MPG, MOV, OGV, etc. depending on how your particular OpenCV was compiled.
The program also reads our multi-terabyte .DMCdata video files in a proprietary binary format from the
[HiST project](https://github.com/space-physics/histfeas).

It is usually straightforward to adapt the program to ingest NetCDF, HDF5 and many other formats.

## Install

```sh
python -m pip install -e ./ionosphereAI
```

If you don't have OpenCV:

```sh
python -m pip install opencv-python
```

or

```sh
conda install opencv
```
