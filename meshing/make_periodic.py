import numpy as np

def make_periodic(blk,next_block,next_patch,nbnow,pitch):
    ng=1
    up = {}
    dn = {}
    
    ib=nbnow
    im_next_block=next_block[ib]['im']
    ip_next_block=next_block[ib]['ip']
    jm_next_block=next_block[ib]['jm']
    jp_next_block=next_block[ib]['jp']
    im_next_patch=next_patch[ib]['im']
    ip_next_patch=next_patch[ib]['ip']
    jm_next_patch=next_patch[ib]['jm']
    jp_next_patch=next_patch[ib]['jp']

    if (im_next_block != 0):
        if (im_next_patch == 1):
            yoffset=0.0
            if ((blk[ib]['y'][0,0] - blk[im_next_block-1]['y'][0,0]) > 0.1*pitch):
                yoffset = pitch
            if ((blk[ib]['y'][0,0] - blk[im_next_block-1]['y'][0,0]) < -0.1*pitch):
                yoffset=- pitch
            xup=blk[im_next_block-1]['x'][0,:]
            yup=blk[im_next_block-1]['y'][0,:] + yoffset
            blk[ib]['x'][0,:]=(xup + blk[ib]['x'][0,:])*0.5
            blk[ib]['y'][0,:]=(yup + blk[ib]['y'][0,:])*0.5
            blk[im_next_block-1]['x'][0,:]=blk[ib]['x'][0,:]
            blk[im_next_block-1]['y'][0,:]=blk[ib]['y'][0,:] - yoffset
            
        if (im_next_patch == 2):
            yoffset=0.0
            if ((blk[ib]['y'][0,0] - blk[im_next_block-1]['y'][-1,0]) > 0.1*pitch):
                yoffset=pitch
            if ((blk[ib]['y'][0,0] - blk[im_next_block-1]['y'][-1,0]) < -0.1*pitch):
                yoffset=- pitch
            xup=blk[im_next_block-1]['x'][-1,:]
            yup=blk[im_next_block-1]['y'][-1,:] + yoffset
            blk[ib]['x'][0,:]=(xup + blk[ib]['x'][0,:])*0.5
            blk[ib]['y'][0,:]=(yup + blk[ib]['y'][0,:])*0.5
            blk[im_next_block-1]['x'][-1,:]=blk[ib]['x'][0,:]
            blk[im_next_block-1]['y'][-1,:]=blk[ib]['y'][0,:] - yoffset
            
        if (im_next_patch == 3):
            yoffset=0.0
            if ((blk[ib]['y'][0,0] - blk[im_next_block-1]['y'][0,0]) > 0.1*pitch):
                yoffset=pitch
            if ((blk[ib]['y'][0,0] - blk[im_next_block-1]['y'][0,0]) < -0.1*pitch):
                yoffset=- pitch
            xup=np.transpose(blk[im_next_block-1]['x'][:,0])
            yup=np.transpose(blk[im_next_block-1]['y'][:,0]) + yoffset
            blk[ib]['x'][0,:]=(xup + blk[ib]['x'][0,:])*0.5
            blk[ib]['y'][0,:]=(yup + blk[ib]['y'][0,:])*0.5
            blk[im_next_block-1]['x'][:,0]=np.transpose(blk[ib]['x'][0,:])
            blk[im_next_block-1]['y'][:,0]=np.transpose(blk[ib]['y'][0,:]) - yoffset
            
        if (im_next_patch == 4):
            yoffset=0.0
            if ((blk[ib]['y'][0,0] - blk[im_next_block-1]['y'][0,-1]) > 0.1*pitch):
                yoffset=pitch
            if ((blk[ib]['y'][0,0] - blk[im_next_block-1]['y'][0,-1]) < -0.1*pitch):
                yoffset=- pitch
            xup=np.transpose(blk[im_next_block-1]['x'][:,-1])
            yup=np.transpose(blk[im_next_block-1]['y'][:,-1]) + yoffset
            blk[ib]['x'][0,:]=(xup + blk[ib]['x'][0,:])*0.5
            blk[ib]['y'][0,:]=(yup + blk[ib]['y'][0,:])*0.5
            blk[im_next_block-1]['x'][:,-1]=np.transpose(blk[ib]['x'][0,:])
            blk[im_next_block-1]['y'][:,-1]=np.transpose(blk[ib]['y'][0,:]) - yoffset
    
    if (ip_next_block != 0):
        if (ip_next_patch == 1):
            yoffset=0.0
            if ((blk[ib]['y'][-1,0] - blk[ip_next_block-1]['y'][0,0]) > 0.1*pitch):
                yoffset = pitch
            if ((blk[ib]['y'][-1,0] - blk[ip_next_block-1]['y'][0,0]) < -0.1*pitch):
                yoffset=- pitch
            xup=blk[ip_next_block-1]['x'][0,:]
            yup=blk[ip_next_block-1]['y'][0,:] + yoffset
            blk[ib]['x'][-1,:]=(xup + blk[ib]['x'][-1,:])*0.5
            blk[ib]['y'][-1,:]=(yup + blk[ib]['y'][-1,:])*0.5
            blk[ip_next_block-1]['x'][0,:]=blk[ib]['x'][-1,:]
            blk[ip_next_block-1]['y'][0,:]=blk[ib]['y'][-1,:] - yoffset
            
        if (ip_next_patch == 2):
            yoffset=0.0
            if ((blk[ib]['y'][-1,0] - blk[ip_next_block-1]['y'][-1,0]) > 0.1*pitch):
                yoffset=pitch
            if ((blk[ib]['y'][-1,0] - blk[ip_next_block-1]['y'][-1,0]) < -0.1*pitch):
                yoffset=- pitch
            xup=blk[ip_next_block-1]['x'][-1,:]
            yup=blk[ip_next_block-1]['y'][-1,:] + yoffset
            blk[ib]['x'][-1,:]=(xup + blk[ib]['x'][-1,:])*0.5
            blk[ib]['y'][-1,:]=(yup + blk[ib]['y'][-1,:])*0.5
            blk[ip_next_block-1]['x'][-1,:]=blk[ib]['x'][-1,:]
            blk[ip_next_block-1]['y'][-1,:]=blk[ib]['y'][-1,:] - yoffset
            
        if (ip_next_patch == 3):
            yoffset=0.0
            if ((blk[ib]['y'][-1,0] - blk[ip_next_block-1]['y'][0,0]) > 0.1*pitch):
                yoffset=pitch
            if ((blk[ib]['y'][-1,0] - blk[ip_next_block-1]['y'][0,0]) < -0.1*pitch):
                yoffset=- pitch
            xup=np.transpose(blk[ip_next_block-1]['x'][:,0])
            yup=np.transpose(blk[ip_next_block-1]['y'][:,0]) + yoffset
            blk[ib]['x'][-1,:]=(xup + blk[ib]['x'][-1,:])*0.5
            blk[ib]['y'][-1,:]=(yup + blk[ib]['y'][-1,:])*0.5
            blk[ip_next_block-1]['x'][:,0]=np.transpose(blk[ib]['x'][-1,:])
            blk[ip_next_block-1]['y'][:,0]=np.transpose(blk[ib]['y'][-1,:]) - yoffset
            
        if (ip_next_patch == 4):
            yoffset=0.0
            if ((blk[ib]['y'][-1,0] - blk[ip_next_block-1]['y'][0,-1]) > 0.1*pitch):
                yoffset=pitch
            if ((blk[ib]['y'][-1,0] - blk[ip_next_block-1]['y'][0,-1]) < -0.1*pitch):
                yoffset=- pitch
            xup=np.transpose(blk[ip_next_block-1]['x'][:,-1])
            yup=np.transpose(blk[ip_next_block-1]['y'][:,-1]) + yoffset
            blk[ib]['x'][-1,:]=(xup + blk[ib]['x'][-1,:])*0.5
            blk[ib]['y'][-1,:]=(yup + blk[ib]['y'][-1,:])*0.5
            blk[ip_next_block-1]['x'][:,-1]=np.transpose(blk[ib]['x'][-1,:])
            blk[ip_next_block-1]['y'][:,-1]=np.transpose(blk[ib]['y'][-1,:]) - yoffset
    
    if (jm_next_block != 0):
        if (jm_next_patch == 1):
            yoffset=0.0
            if ((blk[ib]['y'][0,0] - blk[jm_next_block-1]['y'][0,0]) > 0.1*pitch):
                yoffset = pitch
            if ((blk[ib]['y'][0,0] - blk[jm_next_block-1]['y'][0,0]) < -0.1*pitch):
                yoffset=- pitch
            xup=np.transpose(blk[jm_next_block-1]['x'][0,:])
            yup=np.transpose(blk[jm_next_block-1]['y'][0,:]) + yoffset
            blk[ib]['x'][:,0]=(xup + blk[ib]['x'][:,0])*0.5
            blk[ib]['y'][:,0]=(yup + blk[ib]['y'][:,0])*0.5
            blk[jm_next_block-1]['x'][0,:]=np.transpose(blk[ib]['x'][:,0])
            blk[jm_next_block-1]['y'][0,:]=np.transpose(blk[ib]['y'][:,0]) - yoffset
            
        if (jm_next_patch == 2):
            yoffset=0.0
            if ((blk[ib]['y'][0,0] - blk[jm_next_block-1]['y'][-1,0]) > 0.1*pitch):
                yoffset=pitch
            if ((blk[ib]['y'][0,0] - blk[jm_next_block-1]['y'][-1,0]) < -0.1*pitch):
                yoffset=- pitch
            xup=np.transpose(blk[jm_next_block-1]['x'][-1,:])
            yup=np.transpose(blk[jm_next_block-1]['y'][-1,:]) + yoffset
            blk[ib]['x'][:,0]=(xup + blk[ib]['x'][:,0])*0.5
            blk[ib]['y'][:,0]=(yup + blk[ib]['y'][:,0])*0.5
            blk[jm_next_block-1]['x'][-1,:]=np.transpose(blk[ib]['x'][:,0])
            blk[jm_next_block-1]['y'][-1,:]=np.transpose(blk[ib]['y'][:,0]) - yoffset
            
        if (jm_next_patch == 3):
            yoffset=0.0
            if ((blk[ib]['y'][0,0] - blk[jm_next_block-1]['y'][0,0]) > 0.1*pitch):
                yoffset=pitch
            if ((blk[ib]['y'][0,0] - blk[jm_next_block-1]['y'][0,0]) < -0.1*pitch):
                yoffset=- pitch
            xup=blk[jm_next_block-1]['x'][:,0]
            yup=blk[jm_next_block-1]['y'][:,0] + yoffset
            blk[ib]['x'][:,0]=(xup + blk[ib]['x'][:,0])*0.5
            blk[ib]['y'][:,0]=(yup + blk[ib]['y'][:,0])*0.5
            blk[jm_next_block-1]['x'][:,0]=blk[ib]['x'][:,0]
            blk[jm_next_block-1]['y'][:,0]=blk[ib]['y'][:,0] - yoffset
            
        if (jm_next_patch == 4):
            yoffset=0.0
            if ((blk[ib]['y'][0,0] - blk[jm_next_block-1]['y'][0,-1]) > 0.1*pitch):
                yoffset=pitch
            if ((blk[ib]['y'][0,0] - blk[jm_next_block-1]['y'][0,-1]) < -0.1*pitch):
                yoffset=- pitch
            xup=blk[jm_next_block-1]['x'][:,-1]
            yup=blk[jm_next_block-1]['y'][:,-1] + yoffset
            blk[ib]['x'][:,0]=(xup + blk[ib]['x'][:,0])*0.5
            blk[ib]['y'][:,0]=(yup + blk[ib]['y'][:,0])*0.5
            blk[jm_next_block-1]['x'][:,-1]=blk[ib]['x'][:,0]
            blk[jm_next_block-1]['y'][:,-1]=blk[ib]['y'][:,0] - yoffset
    
    if (jp_next_block != 0):
        if (jp_next_patch == 1):
            yoffset=0.0
            if ((blk[ib]['y'][0,-1] - blk[jp_next_block-1]['y'][0,0]) > 0.1*pitch):
                yoffset = pitch
            if ((blk[ib]['y'][0,-1] - blk[jp_next_block-1]['y'][0,0]) < -0.1*pitch):
                yoffset=- pitch
            xup=np.transpose(blk[jp_next_block-1]['x'][0,:])
            yup=np.transpose(blk[jp_next_block-1]['y'][0,:]) + yoffset
            blk[ib]['x'][:,-1]=(xup + blk[ib]['x'][:,-1])*0.5
            blk[ib]['y'][:,-1]=(yup + blk[ib]['y'][:,-1])*0.5
            blk[jp_next_block-1]['x'][0,:]=np.transpose(blk[ib]['x'][:,-1])
            blk[jp_next_block-1]['y'][0,:]=np.transpose(blk[ib]['y'][:,-1]) - yoffset
            
        if (jp_next_patch == 2):
            yoffset=0.0
            if ((blk[ib]['y'][0,-1] - blk[jp_next_block-1]['y'][-1,0]) > 0.1*pitch):
                yoffset=pitch
            if ((blk[ib]['y'][0,-1] - blk[jp_next_block-1]['y'][-1,0]) < -0.1*pitch):
                yoffset=- pitch
            xup=np.transpose(blk[jp_next_block-1]['x'][-1,:])
            yup=np.transpose(blk[jp_next_block-1]['y'][-1,:]) + yoffset
            blk[ib]['x'][:,-1]=(xup + blk[ib]['x'][:,-1])*0.5
            blk[ib]['y'][:,-1]=(yup + blk[ib]['y'][:,-1])*0.5
            blk[jp_next_block-1]['x'][-1,:]=np.transpose(blk[ib]['x'][:,-1])
            blk[jp_next_block-1]['y'][-1,:]=np.transpose(blk[ib]['y'][:,-1]) - yoffset
            
        if (jp_next_patch == 3):
            yoffset=0.0
            if ((blk[ib]['y'][0,-1] - blk[jp_next_block-1]['y'][0,0]) > 0.1*pitch):
                yoffset=pitch
            if ((blk[ib]['y'][0,-1] - blk[jp_next_block-1]['y'][0,0]) < -0.1*pitch):
                yoffset=- pitch
            xup=blk[jp_next_block-1]['x'][:,0]
            yup=blk[jp_next_block-1]['y'][:,0] + yoffset
            blk[ib]['x'][:,-1]=(xup + blk[ib]['x'][:,-1])*0.5
            blk[ib]['y'][:,-1]=(yup + blk[ib]['y'][:,-1])*0.5
            blk[jp_next_block-1]['x'][:,0]=blk[ib]['x'][:,-1]
            blk[jp_next_block-1]['y'][:,0]=blk[ib]['y'][:,-1] - yoffset
            
        if (jp_next_patch == 4):
            yoffset=0.0
            if ((blk[ib]['y'][0,-1] - blk[jp_next_block-1]['y'][0,-1]) > 0.1*pitch):
                yoffset=pitch
            if ((blk[ib]['y'][0,-1] - blk[jp_next_block-1]['y'][0,-1]) < -0.1*pitch):
                yoffset=- pitch
            xup=blk[jp_next_block-1]['x'][:,-1]
            yup=blk[jp_next_block-1]['y'][:,-1] + yoffset
            blk[ib]['x'][:,-1]=(xup + blk[ib]['x'][:,-1])*0.5
            blk[ib]['y'][:,-1]=(yup + blk[ib]['y'][:,-1])*0.5
            blk[jp_next_block-1]['x'][:,-1]=blk[ib]['x'][:,-1]
            blk[jp_next_block-1]['y'][:,-1]=blk[ib]['y'][:,-1] - yoffset
            
    return blk