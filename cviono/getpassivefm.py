#!/usr/bin/env python
import h5py
from datetime import datetime as DT
from numpy import log10,absolute,median,ascontiguousarray
from pytz import UTC
"""
Michael Hirsch
Read Haystack Passive FM radar frame, one frame per file
"""

def getfmradarframe(fn):
    with h5py.File(str(fn),'r',libver='latest') as f:
        ambiguity    = ascontiguousarray(f['/ambiguity/ambiguity'].value.T) #transpose makes it Fortran order, which cv2.cv.fromarray doesn't like
        range_km     = f['/ambiguity/range_axis'].value/1e3
        velocity_mps = f['/ambiguity/velocity_axis'].value
        dtime = DT.utcfromtimestamp(f['/ambiguity'].attrs.get('utc_second')).replace(tzinfo=UTC) # replace is required for tzaware
        integration_time = f['/ambiguity'].attrs.get('integration_time')

    logamb = log10(absolute(ambiguity))
    SCRdb = 10*log10(absolute(ambiguity/median(ambiguity)))

    return range_km,velocity_mps,SCRdb,dtime,integration_time,logamb
