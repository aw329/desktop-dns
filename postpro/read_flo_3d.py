import os
import numpy as np

def read_flo_3d(casename):
    
    flo = {}
        
    path = os.getcwd()
      
    # read block dimensions  
    nijk = np.genfromtxt(os.path.join(path,casename,'blockdims.txt'),dtype=int)
    
    # read in flow from each block
    for i in range(len(nijk)):
        
        ni = nijk[i,0]
        nj = nijk[i,1]
        nk = nijk[i,2]
        
        ib = i + 1
    
        flow_name = 'flo2_' + str(ib)
        ind_name =  'nod2_' + str(ib)
        
        flow_file = os.path.join(path,casename,flow_name)
        ind_file = os.path.join(path,casename,ind_name)
    
        f = open(flow_file,'rb')
        g = open(ind_file,'rb')
        
        q   = np.fromfile(f,dtype='float64',count=-1)
        ind = np.fromfile(g,dtype='uint32',count=-1)
        
        f.close()
        g.close
    
        N = np.int(len(q)/5)
        M = np.int(len(ind)/3)
        
        q = np.reshape(q,[5,N],order='F') # make sure to reshape with fortran rule!
        ind = np.reshape(ind,[3,M],order='F') # make sure to reshape with fortran rule!
        
        ro = np.zeros([ni,nj,nk])
        ru = np.zeros([ni,nj,nk])
        rv = np.zeros([ni,nj,nk])
        rw = np.zeros([ni,nj,nk])
        Et = np.zeros([ni,nj,nk])
        
        for n in range(M):
            i = ind[0,n]-1
            j = ind[1,n]-1
            k = ind[2,n]-1
            #print(i,j,k,n)
            ro[i,j,k] = q[0,n]
            ru[i,j,k] = q[1,n]
            rv[i,j,k] = q[2,n]
            rw[i,j,k] = q[3,n]
            Et[i,j,k] = q[4,n]
            
        
        flo[ib] = {}   
        flo[ib]['ro'] = ro
        flo[ib]['ru'] = ru
        flo[ib]['rv'] = rv
        flo[ib]['rw'] = rw
        flo[ib]['Et'] = Et
    
    return flo