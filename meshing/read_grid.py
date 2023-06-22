# Copyright (c) 2023, University of Cambridge, all rights reserved. Written by Andrew Wheeler, University of Cambridge

import os
import numpy as np

def read_grid(casename):

    grid = {} 
    blk = {}
    
    path = os.path.join(os.getcwd(),casename)
    blockdims = os.path.join(path,'blockdims.txt')
    bijk = np.loadtxt(blockdims,dtype=np.int32)
    #print(len(np.shape(bijk)))
    if( len(np.shape(bijk))==1 ):
        NB = 1
    else:
        NB,_ = np.shape(bijk)

    for ib in range(NB):
        blk[ib] = {}
        if(NB==1):
            ni,nj,nk = bijk[:]
        else:
            ni,nj,nk = bijk[ib,:]
        
        grid_file = 'grid_' + str(ib+1) + '.txt'        
        grid_file_path = os.path.join(path,grid_file)        
        
        grid = np.loadtxt(grid_file_path)
        #print(np.shape(grid))
        
        x=np.reshape(grid[:,0],[ni,nj],order='F')
        y=np.reshape(grid[:,1],[ni,nj],order='F')
        
        blk[ib]['x'] = x
        blk[ib]['y'] = y
        


    return  blk
    
