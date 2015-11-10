#!/usr/bin/env python3

from setuptools import setup
import subprocess,os

with open('README.rst','r') as f:
	long_description = f.read()

try:
    subprocess.call(['conda','install','--file','requirements.txt'],env={'PATH': os.environ['PATH']},shell=False)
    ok = True
except Exception as e:
    ok = False

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
                        'pathlib2>=2.1.0'],
      packages=['cvhst'],
	  )

if not ok:
    print('you will need to install packages in requirements.txt  {}'.format(e))