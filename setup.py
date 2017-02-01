#!/usr/bin/env python
from setuptools import setup
try:
    import conda.cli
    conda.cli.main('install','--file','requirements.txt')
except Exception as e:
    print(e)

setup(name='cviono',
      packages=['cviono'],
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
