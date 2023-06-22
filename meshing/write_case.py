import os
import numpy as np

def write_case(case):
        
    blk = case['blk'] 
    next_block = case['next_block']
    next_patch = case['next_patch']
    corner = case['corner']
    bcs = case['bcs']
    gas = case['gas']
    solver = case['solver'] 
    casename = case['casename'] 
    version = solver['version']
    
    NB=len(blk)
    ncorner=len(corner)
    NI = {}
    NJ = {}
    cor_type = {}
   
    basedir = os.getcwd()
    path = os.path.join(basedir,casename)
    if os.path.isdir(path) == False:
       os.mkdir(path) 
       
       
    gpu_file = 'input_gpu.txt'   
    cpu_file = 'input_cpu.txt'   
       
    gpu_file_path = os.path.join(path,gpu_file)
    cpu_file_path = os.path.join(path,cpu_file)
    
    #if(os.path.exists(gpu_file_path)):
    #   print('removing old input file') 
    #   os.remove(gpu_file_path)
    #   
    #if(os.path.exists(cpu_file_path)):
    #   print('removing old input file') 
    #   os.remove(cpu_file_path)
    
    
    # Now write header for new input file
    if(version == 'gpu'):
        print('Writing inputs for 3dns_gpu')
        # GPU input file
        f = open(gpu_file_path, 'w')
        f.write('%d %d\n' %(NB,solver['nkproc']))
        
        nprocs=0
        for ib in range(NB):
               
            ni,nj=np.shape(blk[ib]['x'])
            im_next_block=next_block[ib]['im']
            ip_next_block=next_block[ib]['ip']
            jm_next_block=next_block[ib]['jm']
            jp_next_block=next_block[ib]['jp']
            im_next_patch=next_patch[ib]['im']
            ip_next_patch=next_patch[ib]['ip']
            jm_next_patch=next_patch[ib]['jm']
            jp_next_patch=next_patch[ib]['jp']
            
            f.write('%d %d %d\n' %(ni, nj, solver['nk']))
            
            im = (im_next_block == 0)*im_next_patch
            ip = (ip_next_block == 0)*ip_next_patch
            jm = (jm_next_block == 0)*jm_next_patch
            jp = (jp_next_block == 0)*jp_next_patch
            
            f.write('%d %d %d %d\n' %(im, ip, jm, jp))
                    
            if (im == 0):
                f.write('%d %d\n' %(im_next_block, im_next_patch))
            if (ip == 0):
                f.write('%d %d\n' %(ip_next_block, ip_next_patch))
            if (jm == 0):
                f.write('%d %d\n' %(jm_next_block, jm_next_patch))
            if (jp == 0):
                f.write('%d %d\n' %(jp_next_block, jp_next_patch))
                
            NI[ib]=ni
            NJ[ib]=nj
        
        # write corners
        f.write('%d\n' %(ncorner))
        for n in range(ncorner):
            cor_type[n]=0
            f.write('%d %d\n' %(corner[n]['Nb'],cor_type[n]))
            
            for m in range(corner[n]['Nb']):
                ib=corner[n]['block'][m]-1
                ic=corner[n]['i'][m]
                jc=corner[n]['j'][m]
                if (ic > 1):
                    ic=NI[ib]
                if (jc > 1):
                    jc=NJ[ib]
                
                f.write('%d %d %d\n' %(ib+1,ic,jc))         
                
                
        block_groups = solver['block_groups']
    
    # number of block groups
        f.write('%d' %(len(block_groups)))              
    
        for b in block_groups:
        # number of blocks in block group
            f.write('\n%d\n' %(len(b['blocks'])))    
        # blocks in block group    
            for i in range(len(b['blocks'])):
                f.write('%d ' %(b['blocks'][i]))  
             
    # write rest of file
    # check these are ok for your case!
    
        #nsteps, nwrite, ncut    
        f.write('\n%d %d %d\n' %(solver['niter'],solver['nwrite'],solver['ncut']))  
        
        #cfl, filter coefficient
        f.write('%f %f\n' %(solver['cfl'],solver['sigma']))  
        
        #bcs
        f.write('%f %f %f %f %f %f %f %d %d %f\n' \
        %(bcs['Toin'],bcs['Poin'],bcs['pexit'],bcs['vin'],\
          bcs['alpha'],bcs['gamma'],bcs['aturb'],bcs['ilength'],\
          bcs['radprof'],0.0))
        
        #gas props
        f.write('%f %f %12.5e %f %f %f\n'\
        %(gas['gamma'],gas['cp'],gas['mu_ref'],gas['mu_tref'],gas['mu_cref'],gas['pr']))
        
        #span, expan_factor
        f.write( '%f %f\n' %(solver['span'],solver['fexpan']))
        
        # irestart, istats
        f.write('%d %d\n' %(solver['irestart'],solver['istats']))
        
        f.close()
    else :    
        print('Writing inputs for 3dns_cpu')
        # CPU input
        f = open(cpu_file_path, 'w')
        f.write('%d\n' %(NB))
        nprocs=0
        for ib in range(NB):
            ni,nj=np.shape(blk[ib]['x'])
            nk = solver['nk']
            im_next_block=next_block[ib]['im']
            ip_next_block=next_block[ib]['ip']
            jm_next_block=next_block[ib]['jm']
            jp_next_block=next_block[ib]['jp']
            im_next_patch=next_patch[ib]['im']
            ip_next_patch=next_patch[ib]['ip']
            jm_next_patch=next_patch[ib]['jm']
            jp_next_patch=next_patch[ib]['jp']
            
            nkproc=np.int(np.ceil(nk/solver['npp']))
            niproc=np.int(np.ceil(ni/solver['npp']))
            njproc=np.int(np.ceil(nj/solver['npp']))
            nprocs=nprocs + niproc*njproc*nkproc
            
            f.write('%d %d %d\n' %(ni, nj, nk))
            f.write('%d %d %d\n' %(niproc,njproc,nkproc))
            
            im = (im_next_block == 0)*im_next_patch
            ip = (ip_next_block == 0)*ip_next_patch
            jm = (jm_next_block == 0)*jm_next_patch
            jp = (jp_next_block == 0)*jp_next_patch
            
            f.write('%d %d %d %d\n' %(im, ip, jm, jp))
                    
            if (im == 0):
                f.write('%d %d\n' %(im_next_block, im_next_patch))
            if (ip == 0):
                f.write('%d %d\n' %(ip_next_block, ip_next_patch))
            if (jm == 0):
                f.write('%d %d\n' %(jm_next_block, jm_next_patch))
            if (jp == 0):
                f.write('%d %d\n' %(jp_next_block, jp_next_patch))
                
            NI[ib]=ni
            NJ[ib]=nj
        
        # write corners
        f.write('%d\n' %(ncorner))
        for n in range(ncorner):
            cor_type[n]=0
            f.write('%d %d\n' %(corner[n]['Nb'],cor_type[n]))
            
            for m in range(corner[n]['Nb']):
                ib=corner[n]['block'][m]-1
                ic=corner[n]['i'][m]
                jc=corner[n]['j'][m]
                if (ic > 1):
                    ic=NI[ib]
                if (jc > 1):
                    jc=NJ[ib]
                
                f.write('%d %d %d\n' %(ib+1,ic,jc))         
    
    # write rest of file
    # check these are ok for your case!
    
        #nsteps, nwrite, ncut    
        f.write('%d %d %d\n' %(solver['niter'],solver['nwrite'],solver['ncut']))  
        
        #cfl, filter coefficient, ifsplit, ifsat, ifLES, [don't use rest]
        #f.write('%f %f %d %d %d %d %d\n' \
        #%(solver['cfl'],solver['sigma'],solver['ifsplit'],solver['ifsat'],solver['ifLES'],0,0))  
        f.write('%f %f %d %d %d %d %d\n' \
        %(solver['cfl'],solver['sigma'],1,2,0,0,0))  
        
        #bcs
        f.write('%f %f %f %f %f %f %f %f %d %d\n' \
        %(bcs['Toin'],bcs['Poin'],bcs['pexit'],bcs['vin'],\
          bcs['alpha'],bcs['cax'],bcs['aturb'],bcs['lturb'],bcs['ilength'],\
          bcs['radprof']))
    
        #gas props
        f.write('%f %f %12.5e %f %f %f\n'\
        %(gas['gamma'],gas['cp'],gas['mu_ref'],gas['mu_tref'],gas['mu_cref'],gas['pr']))
        
        #span, expan_factor
        f.write( '%f %f\n' %(solver['span'],solver['fexpan']))
        
        # irestart, istats
        f.write('%d %d\n' %(solver['irestart'],solver['istats']))
       
        # number of inlets
        f.write('%d\n' %(1))
        
        # number of blocks in inlet block
        f.write('%d\n' %(2))
        
        # blocks in inlet block
        f.write('%d\n' %(1))
        f.write('%d\n' %(2))
        
        # input for stability analysis    
        f.write('%d %d %d %d %d %f %f\n' %(15,150,177,183,2,5.0,10000.0))
        
        # input for incoming boundary layer and adiabatic/isothermal wall
        f.write('%d %f %f\n' %(0,100000.0,bcs['twall']))
        
        f.close()
        
    # write grid
    for ib in range(NB):
        x=blk[ib]['x']
        y=blk[ib]['y']
        ni,nj=np.shape(x)
        
        grid_file = 'grid_' + str(ib+1) + '.txt'
        
        grid_file_path = os.path.join(path,grid_file)
        
        f = open(grid_file_path,'w')
        
        for j in range(nj):
            for i in range(ni):
                f.write('%20.16e %20.16e\n' %(x[i,j],y[i,j]))
        f.close()
    
    # write blockdims file
    f = open(os.path.join(path,'blockdims.txt'),'w')
    total_ij_points = 0
    for ib in range(NB):
        x=blk[ib]['x']
        ni,nj=np.shape(x)     
        total_ij_points = total_ij_points + ni*nj
        f.write('%d %d %d\n' %(ni,nj,solver['nk']))
    
    f.close()
    
    print('Total ij points')
    print(total_ij_points)

    if(version=='cpu'):
       print('Total processors')
       print(nprocs)

    if(version=='gpu'):
       print('Total GPUs')
       print(solver['nkproc']*len(block_groups))
    
    
    return 
    
    
