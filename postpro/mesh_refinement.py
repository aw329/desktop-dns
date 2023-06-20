import numpy as np
from scipy.interpolate import interp2d

def mesh_refinement(blk,refine_fac,npp):

    NB=len(blk)
    for ib in range(NB):
    
        x=blk[ib]['x']
        y=blk[ib]['y']
        
        ni,nj=np.shape(x)
        ni_new=np.int(npp*np.ceil(refine_fac*ni/ npp))
        nj_new=np.int(npp*np.ceil(refine_fac*nj/ npp))
        
        fi=np.linspace(0,1,ni_new)
        fj=np.linspace(0,1,nj_new)
        fic=np.linspace(0,1,ni)
        fjc=np.linspace(0,1,nj)
        
        fx=interp2d(fjc,fic,x,kind='cubic')
        fy=interp2d(fjc,fic,y,kind='cubic')
        
        blk[ib]['x'] = fx(fj,fi)
        blk[ib]['y'] = fy(fj,fi) 
    
    return blk