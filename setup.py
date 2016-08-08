#!/usr/bin/env python

from setuptools import setup
import subprocess

try:
    subprocess.call(['conda','install','--file','requirements.txt'])
except Exception as e:
    pass

setup(name='cviono',
      packages=['cviono'],
  	  description='OpenCV auroral detection and passive FM radar ionosphere turbulence detector',
	   url='https://github.com/scienceopen/cv_ionosphere',
       dependency_links = [
            'https://github.com/scienceopen/histutils/tarball/master#egg=histutils',
            'https://github.com/scienceopen/cvutils/tarball/master#egg=cvutils'],
	   install_requires=['histutils','cvutils',
                        'tifffile','pathlib2',
                        ],
	  )

try:
    import cv2
except Exception:
    print('you need to install OpenCV for Python. see:')
    print('https://scivision.co/install-opencv-3-0-x-for-python-on-windows/')
