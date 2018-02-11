#!/usr/bin/env python
install_requires = ['pillow','scipy','pandas','numpy','matplotlib','h5py',
       'opencv-python',
       'histutils','dmcutils','morecvutils','pyoptflow']
tests_require=['nose','coveralls']
# %%
from setuptools import setup,find_packages

setup(name='cviono',
      packages=find_packages(),
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
	                    'fits':['fitsio','astropy',],
                      'tests':tests_require,},
	   python_requires='>=3.6',
      install_requires=install_requires,
      tests_require=tests_require,
	  )

try:
    import cv2
    print(f'\nOpenCV {cv2.__version__} detected')
except ImportError:
    raise ImportError('Need to install OpenCV for Python. \n https://www.scivision.co/install-opencv-python-windows/')
