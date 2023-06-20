# Copyright (c) 2023, University of Cambridge, all rights reserved. Written by Andrew Wheeler, University of Cambridge

import os
import numpy as np
from scipy.interpolate import interp1d,interp2d,griddata

from meshing.read_case import *


# prop1 and geom1 contain the time-average flow from previous simulation
def write_mut(prop1,geom1,casename):


    path = os.getcwd()
        
    # read new case files and geometry
    case = read_case(casename)
    
    # get new mesh
    geom2 = case['blk']


    # group data points for interpolant
    Nb = len(geom1)
    xx = []
    yy = []
    Mu = []
    Sr = []
    for nb in range(Nb):
        
        mut = prop1[nb]['mut_opt'] # optimum eddy viscosity   
        S_ = prop1[nb]['S_'] # traceless strain    
        
        x   = geom1[nb]['x']
        y   = geom1[nb]['y']
        
        ni,nj=np.shape(x)
        
        # apply limits to mut
        mut[mut>200.0] = 200.0
        mut[mut<0.0]   = 0.0
        
        # smooth mut
        for ms in range(10):
            mut[1:-1,1:-1] = (mut[0:-2,1:-1] + mut[2:,1:-1] + mut[1:-1,0:-2] + mut[1:-1,2:])*0.25
        
        
        xx   = np.append(xx,np.reshape(x,ni*nj))
        yy   = np.append(yy,np.reshape(y,ni*nj))
        Mu   = np.append(Mu,np.reshape(mut,ni*nj))
        Sr   = np.append(Sr,np.reshape(S_,ni*nj))
        
   
    # eliminate regions of very low strain where mut is not well defined
    strain_limit = np.max(Sr)*0.0001
    sig = strain_limit
    fblend = 1.0 - np.exp( -Sr*Sr/(sig*sig) )
    Mu = fblend*Mu    
    
   
    xin = np.min(xx)    
    xout = np.max(xx)    
    xlen = xout-xin    
    
    ## eliminate regions near inlet and exit
    #sig = xlen*0.01
    #fblend = 1.0 - np.exp( -(xx-xin)*(xx-xin)/(sig*sig) )
    #Mu = fblend*Mu    
    #fblend = 1.0 - np.exp( -(xx-xout)*(xx-xout)/(sig*sig) )
    #Mu = fblend*Mu    
    

    points = (xx,yy)
       
    for nb in range(Nb):
    
        xi = geom2[nb]['x']
        yi = geom2[nb]['y']
        
        ni,nj = np.shape(xi)
        pointsi = (np.reshape(xi,ni*nj,order='F'),np.reshape(yi,ni*nj,order='F'))
        
        muti = griddata(points,Mu,pointsi,'cubic',fill_value=0.0)        
        
        # apply limits to mut
        muti[muti>200.0] = 200.0
        muti[muti<0.0]   = 0.0
      
                    
        # open file                     
        flow_name = 'rans_'+ str(nb+1)
        flow_file = os.path.join(path,casename,flow_name)
        f = open(flow_file,'wb')
       
        # write to file
        muti.tofile(f)
                
        # close file        
        f.close()
       
    
    return






