#!/usr/bin/env python
from setuptools import setup

req = ['nose','scipy','pandas','numpy','matplotlib','h5py','astropy',
        'histutils','morecvutils']


setup(name='cviono',
      packages=['cviono'],
      author='Michael Hirsch, Ph.D.',
      description='detect ionospheric phenomena using computer vision algorithms',
      version='0.9',
      classifiers=[
      'Intended Audience :: Science/Research',
      'Development Status :: 4 - Beta',
      'License :: OSI Approved :: MIT License',
      'Topic :: Scientific/Engineering :: Visualization',
      'Programming Language :: Python :: 3.6',
      ],
	   install_requires=req,
	   extras_require={'tifffile':['tifffile'],'opencv':['opencv']}
	  )

try:
    import cv2
except ImportError:
    print('you need to install OpenCV for Python. see: \n https://www.scivision.co/install-opencv-python-windows/')
