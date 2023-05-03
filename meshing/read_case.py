import os
import numpy as np
from .read_grid import *

def read_case(casename):
    
    NI = {}
    NJ = {}
    case = {}
    blk = {}
    next_block = {}
    next_patch = {}
    cor_type = {}
    corner = {}
    bcs = {}
    gas = {}
    solver = {}
   
    basedir = os.getcwd()
    path = os.path.join(basedir,casename)
    
    cpu = False
    gpu = False
    
    cpu_file = 'input_cpu.txt'          
    cpu_file_path = os.path.join(path,cpu_file)
    
    gpu_file = 'input_gpu.txt'          
    gpu_file_path = os.path.join(path,gpu_file)
       
    if(os.path.exists(cpu_file_path)):
       print('cpu input file found') 
       cpu = True
       
    if(os.path.exists(gpu_file_path)):
       print('gpu input file found') 
       gpu = True
    
    if(cpu and gpu):
       print('defaulting to gpu version')
       cpu = False           
    
    
    if(cpu):
        # CPU input
        f = open(cpu_file_path, 'r')
        temp = np.fromstring(f.readline(),dtype=int,sep=' ')
        NB = temp[0]
        nprocs=0
        npoints=0
        for i in range(NB):
            ib = i + 1
        
            nijk = np.fromstring(f.readline(),dtype=int,sep=' ')
            nijk_procs = np.fromstring(f.readline(),dtype=int,sep=' ')
            ptch = np.fromstring(f.readline(),dtype=int,sep=' ')
            
            #print(nijk[2])
            
            ni = nijk[0]
            nj = nijk[1]
            nk = nijk[2]
            
            npoints = npoints + ni*nj*nk
            
            niproc = nijk_procs[0]
            njproc = nijk_procs[1]
            nkproc = nijk_procs[2]
            
            nprocs = nprocs + niproc*njproc*nkproc
            
            im = ptch[0]
            ip = ptch[1]
            jm = ptch[2]
            jp = ptch[3]
            
            next_block[ib] = {}
            next_patch[ib] = {}
                    
            next_block[ib]['im']=0
            next_block[ib]['ip']=0
            next_block[ib]['jm']=0
            next_block[ib]['jp']=0
                                 
            next_patch[ib]['im']=im
            next_patch[ib]['ip']=ip
            next_patch[ib]['jm']=jm
            next_patch[ib]['jp']=jp
                    
            if (im == 0):
                temp = np.fromstring(f.readline(),dtype=int,sep=' ')
                next_block[ib]['im']= temp[0]
                next_patch[ib]['im']= temp[1]            
            if (ip == 0):
                temp = np.fromstring(f.readline(),dtype=int,sep=' ')
                next_block[ib]['ip']= temp[0]
                next_patch[ib]['ip']= temp[1]            
            if (jm == 0):
                temp = np.fromstring(f.readline(),dtype=int,sep=' ')
                next_block[ib]['jm']= temp[0]
                next_patch[ib]['jm']= temp[1]            
            if (jp == 0):
                temp = np.fromstring(f.readline(),dtype=int,sep=' ')
                next_block[ib]['jp']= temp[0]
                next_patch[ib]['jp']= temp[1]            
                
            NI[ib]=ni
            NJ[ib]=nj
        
        # read corners
        temp = np.fromstring(f.readline(),dtype=int,sep=' ')
        ncorner = temp[0]
        for n in range(ncorner):
            cor_type[n]=0
            temp = np.fromstring(f.readline(),dtype=int,sep=' ')
            corner[n] = {}
            corner[n]['Nb'] = temp[0]
            cor_type[n]     = temp[1]
                                
            corner[n]['block'] = {}
            corner[n]['i'] = {}
            corner[n]['j'] = {}
            for m in range(corner[n]['Nb']):
                temp = np.fromstring(f.readline(),dtype=int,sep=' ')
                ib = temp[0]
                ic = temp[1]            
                jc = temp[2]
                
                corner[n]['block'][m] =ib
                corner[n]['i'][m]     =ic
                corner[n]['j'][m]     =jc
        
        solver['nk'] = nk
        
    # read rest of file
    # check these are ok for your case!
        
        #nsteps, nwrite, ncut    
        temp = np.fromstring(f.readline(),dtype=int,sep=' ') 
        solver['niter']  = temp[0]
        solver['nwrite'] = temp[1]
        solver['ncut']   = temp[2]
        
        #cfl, filter coefficient, ifsplit, ifsat, ifLES, [don't use rest]
        temp = np.fromstring(f.readline(),dtype=float,sep=' ')  
        solver['cfl']     = temp[0]
        solver['sigma']   = temp[1]
        solver['ifsplit'] =   int(temp[2])
        solver['ifsat']   =   int(temp[3])
        solver['ifLES']   =   int(temp[4])
            
        #bcs
        temp = np.fromstring(f.readline(),dtype=float,sep=' ')   
        bcs['Toin']    = temp[0]
        bcs['Poin']    = temp[1]
        bcs['pexit']   = temp[2]
        bcs['vin']     = temp[3]
        bcs['alpha']   = temp[4]
        bcs['cax']     = temp[5]
        bcs['aturb']   = temp[6]
        bcs['lturb']   = temp[7]
        bcs['ilength'] =   int(temp[8])
        bcs['radprof'] =   int(temp[9])
        bcs['gamma'] = 0.0    
        bcs['g_z'] = 0.0
        
        #gas props
        temp = np.fromstring(f.readline(),dtype=float,sep=' ')   
        gas['gamma']    = temp[0]
        gas['cp']       = temp[1]
        gas['mu_ref']   = temp[2]
        gas['mu_tref']  = temp[3]
        gas['mu_cref']  = temp[4]
        gas['pr']       = temp[5]
        
        #span, expan_factor
        temp = np.fromstring(f.readline(),dtype=float,sep=' ')   
        solver['span']   = temp[0]
        solver['fexpan'] = temp[1]
        
        # irestart, istats
        temp = np.fromstring(f.readline(),dtype=int,sep=' ')  
        solver['irestart'] = temp[0]
        solver['istats']   = temp[1]  
        
        # number of inlets
        temp = f.readline()  
        
        # number of blocks in inlet block
        temp = f.readline()  
        
        # blocks in inlet block
        temp = f.readline()  
        temp = f.readline()  
        
        # input for stability analysis    
        temp = f.readline()  
        
        # input for incoming boundary layer and adiabatic/isothermal wall
        temp = np.fromstring(f.readline(),dtype=float,sep=' ')   
        bcs['twall'] = temp[2]
        
        f.close()

        solver['version'] = 'cpu'
        
    if(gpu):
        # GPU input
        f = open(gpu_file_path, 'r')
        temp = np.fromstring(f.readline(),dtype=int,sep=' ')
        NB = temp[0]
        solver['nkproc'] = temp[1]
        npoints=0
        for ib in range(NB):
            nijk = np.fromstring(f.readline(),dtype=int,sep=' ')
            ptch = np.fromstring(f.readline(),dtype=int,sep=' ')
             
            ni = nijk[0]
            nj = nijk[1]
            nk = nijk[2]
            
            npoints = npoints + ni*nj*nk
             
            im = ptch[0]
            ip = ptch[1]
            jm = ptch[2]
            jp = ptch[3]
            
            next_block[ib] = {}
            next_patch[ib] = {}
                    
            next_block[ib]['im']=0
            next_block[ib]['ip']=0
            next_block[ib]['jm']=0
            next_block[ib]['jp']=0
                                 
            next_patch[ib]['im']=im
            next_patch[ib]['ip']=ip
            next_patch[ib]['jm']=jm
            next_patch[ib]['jp']=jp
                    
            if (im == 0):
                temp = np.fromstring(f.readline(),dtype=int,sep=' ')
                next_block[ib]['im']= temp[0]
                next_patch[ib]['im']= temp[1]            
            if (ip == 0):
                temp = np.fromstring(f.readline(),dtype=int,sep=' ')
                next_block[ib]['ip']= temp[0]
                next_patch[ib]['ip']= temp[1]            
            if (jm == 0):
                temp = np.fromstring(f.readline(),dtype=int,sep=' ')
                next_block[ib]['jm']= temp[0]
                next_patch[ib]['jm']= temp[1]            
            if (jp == 0):
                temp = np.fromstring(f.readline(),dtype=int,sep=' ')
                next_block[ib]['jp']= temp[0]
                next_patch[ib]['jp']= temp[1]            
                
            NI[ib]=ni
            NJ[ib]=nj
        
        # read corners
        temp = np.fromstring(f.readline(),dtype=int,sep=' ')
        ncorner = temp[0]
        for n in range(ncorner):
            cor_type[n]=0
            temp = np.fromstring(f.readline(),dtype=int,sep=' ')
            corner[n] = {}
            corner[n]['Nb'] = temp[0]
            cor_type[n]     = temp[1]
                                
            corner[n]['block'] = {}
            corner[n]['i'] = {}
            corner[n]['j'] = {}
            for m in range(corner[n]['Nb']):
                temp = np.fromstring(f.readline(),dtype=int,sep=' ')
                ib = temp[0]
                ic = temp[1]            
                jc = temp[2]
                
                corner[n]['block'][m] =ib
                corner[n]['i'][m]     =ic
                corner[n]['j'][m]     =jc
                
                
        # number of block groups
        temp = np.fromstring(f.readline(),dtype=int,sep=' ')    
        block_groups = temp[0]*[None]
        for n in range(len(block_groups)):
            # number of blocks in block group
            temp = np.fromstring(f.readline(),dtype=int,sep=' ')    
            block_groups[n] = {}
            # blocks in block group    
            temp = np.fromstring(f.readline(),dtype=int,sep=' ')    
            block_groups[n]['blocks'] = temp
        
        solver['nk'] = nk
        
    # read rest of file
        
        #nsteps, nwrite, ncut    
        temp = np.fromstring(f.readline(),dtype=int,sep=' ') 
        solver['niter']  = temp[0]
        solver['nwrite'] = temp[1]
        solver['ncut']   = temp[2]
        
        #cfl, filter coefficient, ifsplit, ifsat, ifLES, [don't use rest]
        temp = np.fromstring(f.readline(),dtype=float,sep=' ')  
        solver['cfl']     = temp[0]
        solver['sigma']   = temp[1]
        solver['block_groups'] = block_groups
            
        #bcs
        temp = np.fromstring(f.readline(),dtype=float,sep=' ')   
        bcs['Toin']    = temp[0]
        bcs['Poin']    = temp[1]
        bcs['pexit']   = temp[2]
        bcs['vin']     = temp[3]
        bcs['alpha']   = temp[4]
        bcs['gamma']   = temp[5]
        bcs['cax']     = 1.0
        bcs['aturb']   = temp[6]
        bcs['lturb']   = 1.0
        bcs['ilength'] =   int(temp[7])
        bcs['radprof'] =   int(temp[8])
        bcs['g_z'] = 0.0
        
        #gas props
        temp = np.fromstring(f.readline(),dtype=float,sep=' ')   
        gas['gamma']    = temp[0]
        gas['cp']       = temp[1]
        gas['mu_ref']   = temp[2]
        gas['mu_tref']  = temp[3]
        gas['mu_cref']  = temp[4]
        gas['pr']       = temp[5]
        
        #span, expan_factor
        temp = np.fromstring(f.readline(),dtype=float,sep=' ')   
        solver['span']   = temp[0]
        solver['fexpan'] = temp[1]
        
        # irestart, istats
        temp = np.fromstring(f.readline(),dtype=int,sep=' ')  
        solver['irestart'] = temp[0]
        solver['istats']   = temp[1]  
        
        bcs['twall'] = -1
        
        f.close()
       
        solver['npp'] = 1
        solver['version'] = 'gpu'
        
    # read grid
    blk = read_grid(casename)
    
    case = {}
    
    case['blk'] = blk
    case['next_block'] = next_block
    case['next_patch'] = next_patch
    case['corner'] = corner
    case['bcs'] = bcs
    case['gas'] = gas
    case['solver'] = solver
    case['casename'] = casename
   
    
    return case
    
