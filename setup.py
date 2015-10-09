#!/usr/bin/env python3

from setuptools import setup

with open('README.rst','r') as f:
	long_description = f.read()

setup(name='cvhst',
      version='0.1',
	  description='OpenCV auroral detection for the HiST auroral tomography system',
	  long_description=long_description,
	  author='Michael Hirsch',
	  url='https://github.com/scienceopen/cvhst',
      dependency_links = ['https://github.com/scienceopen/histutils/tarball/master#egg=histutils',
                          'https://github.com/scienceopen/cvutils/tarball/master#egg=cvutils'],
	  install_requires=['histutils','CVutils',
                        'tifffile',
                        'scipy','scikit-image','numpy','h5py','six','nose'],
      packages=['cvhst'],
	  )

