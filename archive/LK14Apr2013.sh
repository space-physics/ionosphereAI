#!/bin/bash

s1='~/HSTdata/DataField/2014-04-17/2014-04-17T07-00-CamSer1387_frames_22000-1-22836.DMCdata' 
n1=(1000 1200)

s2='~/HSTdata/DataField/2013-04-11/2013-04-11T07-00-CamSer7196_frames_713380-1-716560.DMCdata'
n2=(700 6000)

s3=('~/Prospectus/vid/X1387_032307_112005.36_full.avi' 0 0)

this=$s3 #change this

python LKtest.py \
 "${this[0]}" \
 512 512 1 1 \
 0 0 \
 0.01 \
 ${this[1]} ${this[2]}


