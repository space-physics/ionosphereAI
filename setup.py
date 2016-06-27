#!/usr/bin/env python

from setuptools import setup
import subprocess

try:
    subprocess.call(['conda','install','--yes','--file','requirements.txt'])
except Exception as e:
    pass

setup(name='cvhst',
      packages=['cvhst'],
  	  description='OpenCV auroral detection for the HiST auroral tomography system',
	   url='https://github.com/scienceopen/cvhst',
       dependency_links = [
            'https://github.com/scienceopen/histutils/tarball/master#egg=histutils',
            'https://github.com/scienceopen/cvutils/tarball/master#egg=cvutils'],
	   install_requires=['histutils','cvutils',
                        'tifffile',
                        ],
	  )


