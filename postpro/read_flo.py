import os
import numpy as np
from meshing.read_case import *
from .grad import *

def read_flo(casename):
    
    
    flo = {}
        
    
    path = os.getcwd()
    
    # get case details  
    case = read_case(casename)
    
    # unpack case
    blk = case['blk']
      
    # get solver version
    version = case['solver']['version']    

    # get gas props
    gam     = case['gas']['gamma']
    cp      = case['gas']['cp']
    mu_ref  = case['gas']['mu_ref']
    mu_tref = case['gas']['mu_tref']
    mu_cref = case['gas']['mu_cref']
    pr      = case['gas']['pr']
    cv = cp/gam
    rgas = cp-cv     

    for ib in range(len(blk)):
        
        x = blk[ib]['x']
        y = blk[ib]['y']
        flo[ib] = {}
          
        ni,nj = np.shape(blk[ib]['x'])
        nk = case['solver']['nk']

        if(nk>1):
           ro = np.zeros([ni,nj,nk])
           ru = np.zeros([ni,nj,nk])
           rv = np.zeros([ni,nj,nk])
           rw = np.zeros([ni,nj,nk])
           Et = np.zeros([ni,nj,nk])
        else:
           ro = np.zeros([ni,nj])
           ru = np.zeros([ni,nj])
           rv = np.zeros([ni,nj])
           rw = np.zeros([ni,nj])
           Et = np.zeros([ni,nj])
        
    
        if version == 'cpu':
            flow_name = 'flo2_' + str(ib+1)
            ind_name =  'nod2_' + str(ib+1)
            
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
            if(nk>1):
                for n in range(M):
                    i = ind[0,n]-1
                    j = ind[1,n]-1
                    k = ind[2,n]-1
                    ro[i,j,k] = q[0,n]
                    ru[i,j,k] = q[1,n]
                    rv[i,j,k] = q[2,n]
                    rw[i,j,k] = q[3,n]
                    Et[i,j,k] = q[4,n]
            else:
                for n in range(M):
                    i = ind[0,n]-1
                    j = ind[1,n]-1
                    ro[i,j] = q[0,n]
                    ru[i,j] = q[1,n]
                    rv[i,j] = q[2,n]
                    rw[i,j] = q[3,n]
                    Et[i,j] = q[4,n]
                    
        elif version == 'gpu':
        
            flow_name = 'flow_' + str(ib+1)
            rans_name = 'rans_' + str(ib+1)
            
            flow_file = os.path.join(path,casename,flow_name)
            rans_file = os.path.join(path,casename,rans_name)
            
            f = open(flow_file,'rb')
            q   = np.fromfile(f,dtype='float64',count=ni*nj*nk*5)
            f.close()

            if(os.path.exists(rans_file)): 
                f = open(rans_file,'rb')
                v   = np.fromfile(f,dtype='float64',count=ni*nj)
                f.close()
                mut_model = np.reshape(v,[ni,nj,1],'F')
            else:
                mut_model = np.zeros([ni,nj,1])
                        
            q = np.reshape(q,[5,ni,nj,nk],order='F') # make sure to reshape with fortran rule!
            ro = q[0,:,:,:]
            ru = q[1,:,:,:]
            rv = q[2,:,:,:]
            rw = q[3,:,:,:]
            Et = q[4,:,:,:]
            
        
        # get derived quantities
        u = ru/ro
        v = rv/ro
        w = rw/ro
        p = (gam-1.0)*(Et - 0.5*(u*u + v*v + w*w)*ro)
        T = p/(ro*rgas)
        mu = (mu_ref)*( ( mu_tref + mu_cref )/( T + mu_cref ) )*((T/mu_tref)**1.5)  
        alpha = np.arctan2(v,u)*180.0/np.pi
        s = cp*np.log(T/300) - rgas*np.log(p/1e5)
        vel = np.sqrt(u*u + v*v + w*w)
        mach = vel/np.sqrt(gam*rgas*T)
        To = T*(1.0 + (gam-1)*0.5*mach*mach)
        po = p*((To/T)**(gam/(gam-1.0)))
        
        dudx = np.zeros([ni,nj,nk])
        dudy = np.zeros([ni,nj,nk])
        
        dvdx = np.zeros([ni,nj,nk])
        dvdy = np.zeros([ni,nj,nk])
        
        for k in range(nk):
            dudx[:,:,k],dudy[:,:,k] = grad(u[:,:,k],x,y)
            dvdx[:,:,k],dvdy[:,:,k] = grad(v[:,:,k],x,y)
                
        vortz = dvdx - dudy
        

        flo[ib]['ro'] = ro
        flo[ib]['ru'] = ru
        flo[ib]['rv'] = rv
        flo[ib]['rw'] = rw
        flo[ib]['Et'] = Et

        flo[ib]['p'] = p
        flo[ib]['u'] = u
        flo[ib]['v'] = v
        flo[ib]['w'] = w
        flo[ib]['T'] = T
        flo[ib]['s'] = s
        flo[ib]['vortz'] = vortz

        flo[ib]['po'] = po
        flo[ib]['To'] = To
        flo[ib]['mach'] = mach
        flo[ib]['mu'] = mu
        flo[ib]['mut_model'] = mut_model

    return flo,blk
