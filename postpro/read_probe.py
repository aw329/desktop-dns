# Copyright (c) 2023, University of Cambridge, all rights reserved. Written by Andrew Wheeler, University of Cambridge

import os
import numpy as np
from   meshing.read_case import *

def read_probe(casename):
   
    basedir = os.getcwd()
    path = os.path.join(basedir,casename)
    #if os.path.isdir(path) == False:
    #   os.mkdir(path) 
        

    # get case details  
    case = read_case(casename)

    # unpack mesh
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

    # probe input file     
    probe_file = 'probe.txt'   
       
    file_path = os.path.join(path,probe_file)
    
    # read number of probes from probe.txt 
    fp = open(file_path, 'r')
    temp = np.fromstring(fp.readline(),dtype=int,sep=' ')
    nprobe = temp[0]
    
    probe = {}
    
    for i in range(nprobe):
        
        temp = np.fromstring(fp.readline(),dtype=int,sep=' ')          
        nb = temp[0]
        ii = temp[1]
        jj = temp[2]
        
        
        probe[i] = {}
        
        ip = i+1       
        
        probe_file = 'probe_' + str(ip)       
        file_path = os.path.join(path,probe_file)        
            
        f = open(file_path,'rb')
        q   = np.fromfile(f,dtype='float64',count=-1)
        f.close()
        N = np.int(len(q)/6)
        L = N*6
        temp = np.reshape(q[:L],[N,6])
        #temp = np.loadtxt(file_path)

        time=temp[:,0]
        ro = temp[:,1]
        ru = temp[:,2]
        rv = temp[:,3]
        rw = temp[:,4]
        Et = temp[:,5]
  

        # get derived quantities
        u = ru/ro
        v = rv/ro
        w = rw/ro
        p = (gam-1.0)*(Et - 0.5*(ru*u + rv*v + rw*w))
        T = p/(ro*rgas)
        mu = (mu_ref)*( ( mu_tref + mu_cref )/( T + mu_cref ) )*((T/mu_tref)**1.5)  
        alpha = np.arctan2(v,u)*180.0/np.pi
        s = cp*np.log(T/300) - rgas*np.log(p/1e5)
        vel = np.sqrt(u*u + v*v + w*w)
        c = np.sqrt(gam*rgas*T)
        mach = vel/c
        To = T*(1.0 + (gam-1)*0.5*mach*mach)
        po = p*((To/T)**(gam/(gam-1.0))) 
 
        probe[i]['time'] = time
        probe[i]['ro']   = ro
        probe[i]['ru']   = ru
        probe[i]['rv']   = rv
        probe[i]['rw']   = rw
        probe[i]['Et']   = Et
        
        probe[i]['u']   = u
        probe[i]['v']   = v
        probe[i]['w']   = w
        probe[i]['p']   = p
        probe[i]['T']   = T
        probe[i]['c']   = c
        
        probe[i]['po']    = po
        probe[i]['To']    = To
        probe[i]['mach']  = mach
        probe[i]['s']     = s
        probe[i]['vel']   = vel
        probe[i]['alpha'] = alpha
        probe[i]['mu']    = mu
        
        probe[i]['x']    = blk[nb-1]['x'][ii-1,jj-1]
        probe[i]['y']    = blk[nb-1]['y'][ii-1,jj-1]
                
     
    fp.close()
       
        
    return probe
    
    
