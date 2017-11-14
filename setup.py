#!/usr/bin/env python
req = ['nose','pillow','scipy','pandas','numpy','matplotlib','h5py',
       'histutils','dmcutils','morecvutils','pyoptflow']
# %%
from setuptools import setup

setup(name='cviono',
      packages=['cviono'],
      author='Michael Hirsch, Ph.D.',
      description='detect ionospheric phenomena using machine learning algorithms',
      version='0.9.1',
      classifiers=[
      'Intended Audience :: Science/Research',
      'Development Status :: 4 - Beta',
      'License :: OSI Approved :: MIT License',
      'Topic :: Scientific/Engineering :: Visualization',
      'Programming Language :: Python :: 3.6',
      ],
	   extras_require={'plot':['tifffile'],
	                    'fits':['fitsio','astropy',]},
	   python_requires='>=3.6',
      install_requires=req,
	  )

try:
    import cv2
    print(f'\nOpenCV {cv2.__version__} detected')
except ImportError:
    raise ImportError('Need to install OpenCV for Python. \n https://www.scivision.co/install-opencv-python-windows/')
