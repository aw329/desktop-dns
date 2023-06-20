# Copyright (c) 2023, University of Cambridge, all rights reserved. Written by Andrew Wheeler, University of Cambridge



import numpy as np

def area(x,y):
    
    ni,nj = np.shape(x)
    
    pv = np.zeros([ni-1,nj-1,3])
    qv = np.zeros([ni-1,nj-1,3])
    
    xx11 = x[:-1,:-1]
    xx12 = x[:-1,1:] 
    xx22 = x[1:,1:] 
    xx21 = x[1:,:-1]
    
    yy11 = y[:-1,:-1]
    yy12 = y[:-1,1:] 
    yy22 = y[1:,1:] 
    yy21 = y[1:,:-1]
                       
    pv[:,:,0] = xx22 - xx11          
    pv[:,:,1] = yy22 - yy11          
    pv[:,:,2] = 0.0
    qv[:,:,0] = xx21 - xx12          
    qv[:,:,1] = yy21 - yy12          
    qv[:,:,2] = 0.0
    
    a = (np.cross(pv,qv))*0.5
    ar  = np.abs(a[:,:,2])
    
    return ar
