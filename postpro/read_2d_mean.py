import os
import numpy as np
from meshing.read_case import *
from .grad import *

def read_2d_mean(casename,nfiles):
    
    nstats_prim = 11
    nstats_bud = 6
    nstats = nstats_prim + nstats_bud

    flo = {}
        
    
    path = os.getcwd()

    
    # read mean_time file
    f = os.path.join(path,casename,'mean_time.txt')
    mean_time = np.loadtxt(f,dtype={'names': ('nmean', 'time', 'dt'),'formats': ('i', 'f', 'f')})
    
    try:
       iter(mean_time['time'])
       nfmax = len(mean_time['time'])
    except TypeError:
       nfmax = len([mean_time['time']])
       
    if(len(nfiles)==1):
       if (nfiles[0] >= nfmax) or (nfiles[0]==0):
           nfile = mean_time['nmean']
           time =  mean_time['time']           
           dt   =  mean_time['dt']
       elif nfiles[0] < 0 :
           nfnow = np.abs(nfiles)-1
           nfile = mean_time['nmean'][nfnow]
           time =  mean_time['time'][nfnow]
           dt   =  mean_time['dt'][nfnow]           
       else:    
           nfile = mean_time['nmean'][-nfiles[0]:]
           time  = mean_time['time'][-nfiles[0]:]
           dt    = mean_time['dt'][-nfiles[0]:]
    else:
       nfile = mean_time['nmean'][nfiles[0]-1:nfiles[1]]
       time  =  mean_time['time'][nfiles[0]-1:nfiles[1]]
       dt    =  mean_time['dt'][nfiles[0]-1:nfiles[1]]
       
    total_time = np.sum(dt)
    try:
       iter(nfile)
    except TypeError:
       nfile = list([nfile])
       time = list([time])
       dt = list([dt])
       
    
    nfiles = len(dt)
    
    
    print(dt,total_time,nfile)
    
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
    
    bcs = case['bcs']
    
    
    for ib in range(len(blk)):
        
        x = blk[ib]['x']
        y = blk[ib]['y']
        flo[ib] = {}
          
        ni,nj = np.shape(blk[ib]['x'])
        qq = np.zeros([ni*nj*nstats]) 
        
        for nt,nf in enumerate(nfile):  
         
            flow_name = 'mean2_' + str(ib+1) + '_' + str(nf)
            flow_file = os.path.join(path,casename,flow_name)
            f = open(flow_file,'rb')
            wgt = 1.0/total_time
            qq = qq + (np.fromfile(f,dtype='float64',count=ni*nj*nstats))*wgt 
            f.close()
            
        iend = ni*nj*nstats_prim                
        q = np.reshape(qq[:iend],[nstats_prim,ni,nj],order='F') # make sure to reshape with fortran rule!
        q2 = np.reshape(qq[iend-1:-1],[nstats_bud,ni,nj],order='F') # make sure to reshape with fortran rule!
        
        ro = q[0,:,:]
        ru = q[1,:,:]
        rv = q[2,:,:]
        rw = q[3,:,:]
        Et = q[4,:,:]        

        ruu = q[5,:,:]
        rvv = q[6,:,:]
        rww = q[7,:,:]
        
        ruv = q[8,:,:]
        ruw = q[9,:,:]
        rvw = q[10,:,:]
        
        rus = q2[0,:,:]
        rvs = q2[1,:,:]
        dissT=q2[2,:,:]
        
        qxT  =q2[3,:,:]
        qyT  =q2[4,:,:]
        qzT  =q2[5,:,:]
         
        # get derived quantities
        u = ru/ro
        v = rv/ro
        w = rw/ro
        p = (gam-1.0)*(Et - 0.5*(ruu + rvv + rww))
        T = p/(ro*rgas)
        mu = (mu_ref)*( ( mu_tref + mu_cref )/( T + mu_cref ) )*((T/mu_tref)**1.5)  
        alpha = np.arctan2(v,u)*180.0/np.pi
        s = cp*np.log(T/bcs['Toin']) - rgas*np.log(p/bcs['Poin'])
        vel = np.sqrt(u*u + v*v + w*w)
        mach = vel/np.sqrt(gam*rgas*T)
        To = T*(1.0 + (gam-1)*0.5*mach*mach)
        po = p*((To/T)**(gam/(gam-1.0)))
        
        # Rey stresses
        ruu = ruu - ru*u
        rvv = rvv - rv*v
        rww = rww - rw*w
                        
        ruv = ruv - ru*v
        ruw = ruw - ru*w
        rvw = rvw - rv*w
        
        
        
        tke = 0.5*(ruu + rvv + rww)/ro
        tu = 100*np.sqrt((2.0/3.0)*tke)/(bcs['vin'])

        
        dudx,dudy = grad(u,x,y)
        dvdx,dvdy = grad(v,x,y)
        dwdx,dwdy = grad(w,x,y)
        dTdx,dTdy = grad(T,x,y)
        drus,_    = grad(rus,x,y)
        _,drvs    = grad(rvs,x,y)
        d2qx,_    = grad(qxT,x,y)
        _,d2qy    = grad(qyT,x,y)

        
        # strain tensor (2D,w=0,d/dz=0)
        s11 = dudx
        s22 = dvdy                
        s12 = (dudy + dvdx)*0.5

        # traceless strain
        s11_ = s11*(2.0/3.0) - s22*(1.0/3.0)# - s33*(1.0/3.0) (s33=0 in 2D)
        s22_ =-s11*(1.0/3.0) + s22*(2.0/3.0)# - s33*(1.0/3.0) 
        s33_ =-s11*(1.0/3.0) - s22*(1.0/3.0)# + s33*(2.0/3.0) 
        
        # traceless strain magnitude
        S_ = np.sqrt(s11_*s11_ + s22_*s22_ + s33_*s33_ + 2.0*s12*s12 )
        
        # Reynolds stress tensor
        t11 = ruu
        t22 = rvv
        t33 = rww
        t12 = ruv
        
        # traceless stress
        t11_ = t11*(2.0/3.0) - t22*(1.0/3.0) - t33*(1.0/3.0)
        t22_ =-t11*(1.0/3.0) + t22*(2.0/3.0) - t33*(1.0/3.0)
        t33_ =-t11*(1.0/3.0) - t22*(1.0/3.0) + t33*(2.0/3.0)

        # optimum eddy viscosity  
        mut_opt = 0.5*np.abs( (t11_*s11_ + t22_*s22_ + t33_*s33_ 
                              + 2.0*t12*s12 )/(S_*S_) )/mu
                  
                  
        # dissipation due to time mean strain
        diss_av =(mu*(2.0*(s11*s11 + s22*s22) + 4.0*s12*s12 )
                  - (2.0/3.0)*mu*(s11 + s22)*(s11 + s22)) 
        
        # turbulence production
        turb =  -( ruv*(dudy + dvdx) 
               + rvw*dwdy + ruw*dwdx + ruu*dudx + rvv*dvdy )

        # vorticity 
        vortz = dvdx - dudy
        
        
        # entropy generation terms
        Ds = (drus + drvs)      # LHS (flux derivative)
        Dsr = (d2qx + d2qy)     # rev. term
        Ds_phi = diss_av/T      # dissipation due to time mean strain
        Ds_eps = dissT - Ds_phi # turbulent dissipation


        
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
        
        flo[ib]['Ds'] = Ds
        flo[ib]['Dsr'] = Dsr
        flo[ib]['Ds_phi'] = Ds_phi
        flo[ib]['Ds_eps'] = Ds_eps
        
        flo[ib]['dissT'] = dissT
        
        flo[ib]['tke'] = tke
        flo[ib]['tu'] = tu
        flo[ib]['turb_production'] = turb
        flo[ib]['mut_opt'] = mut_opt
        flo[ib]['S_'] = S_
        
        
        

    return flo,blk 



       
           