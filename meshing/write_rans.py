import os

def write_rans(casename,rans):
   
    basedir = os.getcwd()
    path = os.path.join(basedir,casename)
    
    # rans setttings file     
    file = 'rans.txt'   
    file_path = os.path.join(path,file)
    f = open(file_path, 'w')
    f.write('%d\n' %(rans['if_rans']))
    f.write('%d\n' %(rans['speed_up']))
 
    if(rans['if_rans']==1):
      f.write('%f\n' %(0.0))
    elif(rans['if_rans']==2):
      f.write('%f\n' %(rans['nrans_file']))
      f.write('%s\n' %(rans['path1']))
      f.write('%s\n' %(rans['path2']))
      f.write('%f\n' %(rans['fac']))
 
    f.close()
    
    return 
    
    
