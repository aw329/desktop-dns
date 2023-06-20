import os

def write_probe(casename,probe):
   
    basedir = os.getcwd()
    path = os.path.join(basedir,casename)
    #if os.path.isdir(path) == False:
    #   os.mkdir(path) 
         
    # probe input file     
    probe_file = 'probe.txt'   
       
    file_path = os.path.join(path,probe_file)
    
    nprobe = probe['nprobe']
    nwrite = probe['nwrite']
    
    f = open(file_path, 'w')
    f.write('%d %d\n' %(nprobe,nwrite))
    
    nprocs=0
    for n in range(nprobe):
        
        f.write('%d %d %d %d\n' %(probe[n]['block'],probe[n]['i'],probe[n]['j'],probe[n]['k']))
        
    f.close()
        
    return 
    
    