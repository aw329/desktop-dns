import numpy as np

# A. Wheeler 2023, 2nd order curvilinear gradient 

def grad(f,x,y): 

    dfi = np.gradient(f,axis=0)
    dfj = np.gradient(f,axis=1)
    
    dxi = np.gradient(x,axis=0)
    dxj = np.gradient(x,axis=1)

    dyi = np.gradient(y,axis=0)
    dyj = np.gradient(y,axis=1)
    
       
    xj = dxi*dyj - dxj*dyi
    dix = dyj/xj
    djx =-dyi/xj
    diy =-dxj/xj
    djy = dxi/xj
 
    dfx = dfi*dix + dfj*djx
    dfy = dfi*diy + dfj*djy
    
    
    return dfx,dfy
