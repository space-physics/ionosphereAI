#!/usr/bin/env python3
import pstats
from os.path import expanduser

def goCprofile(profFN):
    profFN = expanduser(profFN)
    p = pstats.Stats(profFN)

#p.strip_dirs() #strip path names

#p.sort_stats('cumulative').print_stats(10) #print 10 longest function
#p.print_stats()

    p.sort_stats('time','cumulative').print_stats(50)
#p.print_stats()

if __name__ == '__main__':
    goCprofile('profstats')