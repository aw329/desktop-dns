# Copyright (c) 2023, University of Cambridge, all rights reserved. Written by Andrew Wheeler, University of Cambridge

import os
import numpy as np
from scipy.interpolate import RegularGridInterpolator
from .read_flo import *

def interp_flo(old_case,new_case):

    flo_new = {}

    path = os.getcwd()
      
    # read old block dimensions  
    nijk_old = np.genfromtxt(os.path.join(path,old_case,'blockdims.txt'),dtype=int)
    
    # read new block dimensions  
    nijk_new = np.genfromtxt(os.path.join(path,new_case,'blockdims.txt'),dtype=int)
    
    flo_old,_ = read_flo(old_case)
   
    NB=len(flo_old)
    for ib in range(NB):
    
        flo_new[ib] = {}
    
        ni_old = nijk_old[ib,0]
        nj_old = nijk_old[ib,1]
        nk_old = nijk_old[ib,2]
        
        ni_new = nijk_new[ib,0]
        nj_new = nijk_new[ib,1]
        nk_new = nijk_new[ib,2]
        
        fic=np.linspace(0,1,ni_old)
        fjc=np.linspace(0,1,nj_old)
        fkc=np.linspace(0,1,nk_old)
        
        fi=np.linspace(0,1,ni_new)
        fj=np.linspace(0,1,nj_new)
        fk=np.linspace(0,1,nk_new)
        
        points = (fic,fjc,fkc)
        fi_new,fj_new,fk_new = np.meshgrid(fi, fj, fk, indexing='ij')
        
        for var in flo_old[ib]: 
            values =  flo_old[ib][var]  
            try:
              fn = RegularGridInterpolator(points, values)
              flo_new[ib][var] = fn((fi_new,fj_new,fk_new))
            except:
              print('Not interpolating ' + var)
            
    
    return flo_new
    
    
    
