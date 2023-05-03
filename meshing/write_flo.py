import os
import numpy as np
from numpy import ndarray
from .read_case import *

def write_flo(flo,casename):
    
        
    path = os.getcwd()
      
    ## read block dimensions  
    #nijk = np.genfromtxt(os.path.join(path,casename,'blockdims.txt'),dtype=int)
    
    # get case details  
    case = read_case(casename)
    
    # unpack case
    blk = case['blk']
    
    # write flow from each block
    for ib in range(len(blk)):
        
        ni,nj = np.shape(blk[ib]['x'])
        nk = case['solver']['nk']
        M = ni*nj*nk 
        
        q = np.zeros([5,M],dtype=float)        
        
        if(case['solver']['version']=='gpu'):
           flow_name = 'flow_' + str(ib+1)
           q[0,:] = np.reshape(flo[ib]['ro'],[ni*nj*nk],order='F')
           q[1,:] = np.reshape(flo[ib]['ru'],[ni*nj*nk],order='F')
           q[2,:] = np.reshape(flo[ib]['rv'],[ni*nj*nk],order='F')
           q[3,:] = np.reshape(flo[ib]['rw'],[ni*nj*nk],order='F')
           q[4,:] = np.reshape(flo[ib]['Et'],[ni*nj*nk],order='F')
           flow_file = os.path.join(path,casename,flow_name)
           f = open(flow_file,'wb')
           # convert to 1d array
           q   = np.reshape(q,[5*M],order='F')
           # write to file
           q.tofile(f)
           # close files        
           f.close()

        else: 
           flow_name = 'flo2_' + str(ib+1)
           ind_name =  'nod2_' + str(ib+1)
            
           ind = np.zeros([3,M],dtype=int)
           
           n = 0
           
           for k in range(nk):
               for j in range(nj):
                   for ii in range(ni):
           
                       
                       ind[0,n] = ii + 1
                       ind[1,n] = j + 1
                       ind[2,n] = k + 1
                     
                       q[0,n] = flo[ib]['ro'][ii,j,k]
                       q[1,n] = flo[ib]['ru'][ii,j,k]
                       q[2,n] = flo[ib]['rv'][ii,j,k]
                       q[3,n] = flo[ib]['rw'][ii,j,k]
                       q[4,n] = flo[ib]['Et'][ii,j,k]
               
                       n = n + 1
                    
           flow_file = os.path.join(path,casename,flow_name)
           ind_file = os.path.join(path,casename,ind_name)
       
           f = open(flow_file,'wb')
           g = open(ind_file,'wb')
           
           # convert to 1d array
           ind = np.reshape(ind,[3*M],order='F')
           q   = np.reshape(q,[5*M],order='F')
           
           # write to file
           ind.tofile(g)
           q.tofile(f)
                   
           # close files        
           f.close()
           g.close()
         
    
    return 