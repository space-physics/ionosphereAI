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
                        'tifffile',
                        ],
	  )


