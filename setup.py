#!/usr/bin/env python
req = ['nose','pillow','scipy','pytables','pandas','numpy','matplotlib','h5py','astropy']
pipreq=['tables','histutils','dmcutils','morecvutils','fitsio','pyoptflow',]
# %%
import pip
try:
    import conda.cli
    conda.cli.main('install',*req)
except Exception as e:
    pip.main(['install'] + req)
pip.main(['install'] + pipreq)
# %%
from setuptools import setup
from sys import stderr

setup(name='cviono',
      packages=['cviono'],
      author='Michael Hirsch, Ph.D.',
      description='detect ionospheric phenomena using computer vision algorithms',
      version='0.9.1',
      classifiers=[
      'Intended Audience :: Science/Research',
      'Development Status :: 4 - Beta',
      'License :: OSI Approved :: MIT License',
      'Topic :: Scientific/Engineering :: Visualization',
      'Programming Language :: Python :: 3.6',
      ],
	   extras_require={'tifffile':['tifffile']}
	  )

try:
    import cv2
    print(f'\nOpenCV {cv2.__version__} detected')
except ImportError:
    print('you need to install OpenCV for Python. see: \n https://www.scivision.co/install-opencv-python-windows/',file=stderr)
