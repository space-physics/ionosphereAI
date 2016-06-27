Copyright 2013 Michael Hirsch

======================
passive-fm-utils
======================

Functions used with data from MIT Haystack Intercepted Signals for Ionospheric Science 
P. I. Frank Lind

.. contents::

data available from
http://atlas.haystack.mit.edu/isis/fermi/events/

Running the Code
================
There is one main program, and three informational programs. All other files are meant for internal use only.

Main Program
------------
``RunCV``  runs process on an HDF5 file, with parameters on lines 3-5

Informational Programs
----------------------
These programs don't actually do any processing, just for user convenience of plotting.

``RawPlayer0`` plays raw audio of HDF5 file, for user convenience (understanding what is the stimulus for the response seen in RunCV)

``PlotSCR`` plots Scatter to Clutter ratio of an HDF5 file.

``FMbesselSidebands`` theoretical sideband relative amplitude

all other functions are meant to be called from main programs.

Getting the Data
================
You will need ``wget``. This is built-in to every Linux system, on Cgywin under Windows, and via HomeBrew or MacPorts on Mac.

Example Download Commands
-------------------------
::
  
  mkdir -p ~/data
  
  wget --quota=10000m -r -np -nc -nH --cut-dirs=3 --random-wait --wait 0.5 --reject index.html* -e robots=off -P ~/data/ http://atlas.haystack.mit.edu/isis/fermi/events/2010-08-03/rx40rx51/
  
  wget --quota=10000m -r -np -nc -nH --cut-dirs=3 --random-wait --wait 0.5 --reject index.html* -e robots=off -P ~/data/ http://atlas.haystack.mit.edu/isis/fermi/events/2010-08-13/rx40rx51/
  
  wget --quota=10000m -r -np -nc -nH --cut-dirs=3 --random-wait --wait 0.5 --reject index.html* -e robots=off -P ~/data/ http://atlas.haystack.mit.edu/isis/fermi/events/2010-08-05/rx40rx51/
  
  wget --quota=10000m -r -np -nc -nH --cut-dirs=3 --random-wait --wait 0.5 --reject index.html* -e robots=off -P ~/data/ http://atlas.haystack.mit.edu/isis/fermi/events/2010-08-04/rx40rx51/


Note
====
There are a couple trivial mistakes in this code. They were fixed in the Python adaptation of this code, compared side-by-side with this old Matlab version, and seen to be insignificant (same conclusions, same performance, etc.)
