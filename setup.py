#!/usr/bin/env python
from setuptools import setup

req = ['nose','scipy','pandas','numpy','matplotlib','h5py','astropy',
        'histutils','cvutils']


setup(name='cviono',
      packages=['cviono'],
       dependency_links = [
            'https://github.com/scienceopen/cvutils/tarball/master#egg=cvutils'],
	   install_requires=req,
	   extras_require={'tifffile':['tifffile']}
	  )

try:
    import cv2
except ImportError:
    print('you need to install OpenCV for Python. see:')
    print('https://scivision.co/install-opencv-3-0-x-for-python-on-windows/')
