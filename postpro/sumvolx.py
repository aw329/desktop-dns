# Copyright (c) 2023, University of Cambridge, all rights reserved. Written by Andrew Wheeler, University of Cambridge

import numpy as np
from .area import *

def sumvolx(prop,geom,sum_var,xpos):

    
    nx = len(xpos)
    totx = np.zeros([nx,1])
    
    for ib in range(len(geom)):
        
        x = geom[ib]['x']
        y = geom[ib]['y']
        a = prop[ib]['area']
    
        f = prop[ib][sum_var]    
        
        if(len(np.shape(f))==3):    
           f = np.mean(f,axis=2)
           
        xav = (x[:-1,:-1] + x[1:,:-1] + x[:-1,1:] + x[1:,1:])*0.25
        fav = (f[:-1,:-1] + f[1:,:-1] + f[:-1,1:] + f[1:,1:])*0.25
        
        for n in range(nx):
            I = (xav < xpos[n]) & (xav > xpos[0])
            totx[n] = totx[n] + np.sum(fav[I]*a[I])       
            
    return totx
    
    
    
