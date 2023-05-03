import os

def write_buffer(casename,inlet,outlet,inex):
   
    basedir = os.getcwd()
    path = os.path.join(basedir,casename)
    
    # inlet buffer file     
    inlet_file = 'in_buffer.txt'   
    file_path = os.path.join(path,inlet_file)
    f = open(file_path, 'w')
    f.write('%d %f\n' %(inlet['N'],inlet['Lwave']))
    f.close()
            
    # exit buffer file     
    exit_file = 'out_buffer.txt'   
    file_path = os.path.join(path,exit_file)
    f = open(file_path, 'w')
    f.write('%d %f\n' %(outlet['N'],outlet['Lwave']))
    f.close()
    
    
    # inlet/exit pitchwise smoothing     
    inex_file = 'inex.txt'   
    file_path = os.path.join(path,inex_file)
    f = open(file_path, 'w')
    f.write('%f' %(inex))
    f.close()
    
    return 
    
    