# Copyright (c) 2023, University of Cambridge, all rights reserved. Written by Andrew Wheeler, University of Cambridge


import numpy as np
import os
    
def read_profile(file_name,inorm):
    f = open(file_name,'r')     
    
    N = int(f.readline())
    
    #display(N)
    
    xprof = []
    yprof = []
    
    pitch=float(f.readline())

    contents=f.readlines()

    for line in contents:
        spl=line.split()
        xprof.append(float(spl[0]))
        yprof.append(float(spl[1]))
    
    f.close()
    
    iLE = np.argmin(xprof)
    iTE = np.argmax(xprof)
    
    xLE = xprof[iLE]
    yLE = yprof[iLE]
    
    xTE = xprof[iTE]
    yTE = yprof[iTE]
    
    cax = xTE - xLE
    
    stag= np.arctan((yTE - yLE) / (xTE - xLE))*180.0 / np.pi
      
    xprof = np.ndarray.flatten(np.asarray(xprof))
    yprof = np.ndarray.flatten(np.asarray(yprof))
    
    if(inorm==True):
       yprof = (yprof - yLE)/cax
       xprof = (xprof - xLE)/cax
       pitch = pitch/cax
    
    
    return xprof,yprof,pitch,stag


if __name__ == '__main__':
    main()
