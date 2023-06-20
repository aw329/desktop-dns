# Copyright (c) 2023, University of Cambridge, all rights reserved. Written by Andrew Wheeler, University of Cambridge


import numpy as np

def grad(f,x,y): 

    dfi = np.gradient(f,axis=0,edge_order=2)
    dfj = np.gradient(f,axis=1,edge_order=2)
    
    dxi = np.gradient(x,axis=0,edge_order=2)
    dxj = np.gradient(x,axis=1,edge_order=2)
    
    dyi = np.gradient(y,axis=0,edge_order=2)
    dyj = np.gradient(y,axis=1,edge_order=2)
    
    xj = dxi*dyj - dxj*dyi
    dix = dyj/xj
    djx =-dyi/xj
    diy =-dxj/xj
    djy = dxi/xj
 
    dfx = dfi*dix + dfj*djx
    dfy = dfi*diy + dfj*djy
    
    return dfx,dfy
