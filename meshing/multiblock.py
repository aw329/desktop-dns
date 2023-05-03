import numpy as np

def multiblock(blk,next_block,next_patch,nbnow,pitch):
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

            up['xi']=blk[im_next_block-1]['x'][1,:]
            up['yi']=blk[im_next_block-1]['y'][1,:] + yoffset

            
        if (im_next_patch == 2):
            yoffset=0.0
            if ((blk[ib]['y'][0,0] - blk[im_next_block-1]['y'][-1,0]) > 0.1*pitch):
                yoffset=pitch
            if ((blk[ib]['y'][0,0] - blk[im_next_block-1]['y'][-1,0]) < -0.1*pitch):
                yoffset=- pitch
            up['xi']=blk[im_next_block-1]['x'][-2,:]
            up['yi']=blk[im_next_block-1]['y'][-2,:] + yoffset

            
        if (im_next_patch == 3):
            yoffset=0.0
            if ((blk[ib]['y'][0,0] - blk[im_next_block-1]['y'][0,0]) > 0.1*pitch):
                yoffset=pitch
            if ((blk[ib]['y'][0,0] - blk[im_next_block-1]['y'][0,0]) < -0.1*pitch):
                yoffset=- pitch
            up['xi']=np.transpose(blk[im_next_block-1]['x'][:,1])
            up['yi']=np.transpose(blk[im_next_block-1]['y'][:,1]) + yoffset
 
            
        if (im_next_patch == 4):
            yoffset=0.0
            if ((blk[ib]['y'][0,0] - blk[im_next_block-1]['y'][0,-1]) > 0.1*pitch):
                yoffset=pitch
            if ((blk[ib]['y'][0,0] - blk[im_next_block-1]['y'][0,-1]) < -0.1*pitch):
                yoffset=- pitch
            up['xi']=np.transpose(blk[im_next_block-1]['x'][:,-1])
            up['yi']=np.transpose(blk[im_next_block-1]['y'][:,-1]) + yoffset

    
    if (ip_next_block != 0):
        if (ip_next_patch == 1):
            yoffset=0.0
            if ((blk[ib]['y'][-1,0] - blk[ip_next_block-1]['y'][0,0]) > 0.1*pitch):
                yoffset = pitch
            if ((blk[ib]['y'][-1,0] - blk[ip_next_block-1]['y'][0,0]) < -0.1*pitch):
                yoffset=- pitch
            dn['xi']=blk[ip_next_block-1]['x'][1,:]
            dn['yi']=blk[ip_next_block-1]['y'][1,:] + yoffset

            
        if (ip_next_patch == 2):
            yoffset=0.0
            if ((blk[ib]['y'][-1,0] - blk[ip_next_block-1]['y'][-1,0]) > 0.1*pitch):
                yoffset=pitch
            if ((blk[ib]['y'][-1,0] - blk[ip_next_block-1]['y'][-1,0]) < -0.1*pitch):
                yoffset=- pitch
            dn['xi']=blk[ip_next_block-1]['x'][-2,:]
            dn['yi']=blk[ip_next_block-1]['y'][-2,:] + yoffset
            
        if (ip_next_patch == 3):
            yoffset=0.0
            if ((blk[ib]['y'][-1,0] - blk[ip_next_block-1]['y'][0,0]) > 0.1*pitch):
                yoffset=pitch
            if ((blk[ib]['y'][-1,0] - blk[ip_next_block-1]['y'][0,0]) < -0.1*pitch):
                yoffset=- pitch
            dn['xi']=np.transpose(blk[ip_next_block-1]['x'][:,1])
            dn['yi']=np.transpose(blk[ip_next_block-1]['y'][:,1]) + yoffset
            
        if (ip_next_patch == 4):
            yoffset=0.0
            if ((blk[ib]['y'][-1,0] - blk[ip_next_block-1]['y'][0,-1]) > 0.1*pitch):
                yoffset=pitch
            if ((blk[ib]['y'][-1,0] - blk[ip_next_block-1]['y'][0,-1]) < -0.1*pitch):
                yoffset=- pitch
            dn['xi']=np.transpose(blk[ip_next_block-1]['x'][:,-1])
            dn['yi']=np.transpose(blk[ip_next_block-1]['y'][:,-1]) + yoffset

    if (jm_next_block != 0):
        if (jm_next_patch == 1):
            yoffset=0.0
            if ((blk[ib]['y'][0,0] - blk[jm_next_block-1]['y'][0,0]) > 0.1*pitch):
                yoffset = pitch
            if ((blk[ib]['y'][0,0] - blk[jm_next_block-1]['y'][0,0]) < -0.1*pitch):
                yoffset=- pitch
            up['xj']=np.transpose(blk[jm_next_block-1]['x'][1,:])
            up['yj']=np.transpose(blk[jm_next_block-1]['y'][1,:]) + yoffset
            
        if (jm_next_patch == 2):
            yoffset=0.0
            if ((blk[ib]['y'][0,0] - blk[jm_next_block-1]['y'][-1,0]) > 0.1*pitch):
                yoffset=pitch
            if ((blk[ib]['y'][0,0] - blk[jm_next_block-1]['y'][-1,0]) < -0.1*pitch):
                yoffset=- pitch
            up['xj']=np.transpose(blk[jm_next_block-1]['x'][-2,:])
            up['yj']=np.transpose(blk[jm_next_block-1]['y'][-2,:]) + yoffset

            
        if (jm_next_patch == 3):
            yoffset=0.0
            if ((blk[ib]['y'][0,0] - blk[jm_next_block-1]['y'][0,0]) > 0.1*pitch):
                yoffset=pitch
            if ((blk[ib]['y'][0,0] - blk[jm_next_block-1]['y'][0,0]) < -0.1*pitch):
                yoffset=- pitch
            up['xj']=blk[jm_next_block-1]['x'][:,1]
            up['yj']=blk[jm_next_block-1]['y'][:,1] + yoffset

            
        if (jm_next_patch == 4):
            yoffset=0.0
            if ((blk[ib]['y'][0,0] - blk[jm_next_block-1]['y'][0,-1]) > 0.1*pitch):
                yoffset=pitch
            if ((blk[ib]['y'][0,0] - blk[jm_next_block-1]['y'][0,-1]) < -0.1*pitch):
                yoffset=- pitch
            up['xj']=blk[jm_next_block-1]['x'][:,-2]
            up['yj']=blk[jm_next_block-1]['y'][:,-2] + yoffset

    
    if (jp_next_block != 0):
        if (jp_next_patch == 1):
            yoffset=0.0
            if ((blk[ib]['y'][0,-1] - blk[jp_next_block-1]['y'][0,0]) > 0.1*pitch):
                yoffset = pitch
            if ((blk[ib]['y'][0,-1] - blk[jp_next_block-1]['y'][0,0]) < -0.1*pitch):
                yoffset=- pitch
            dn['xj']=np.transpose(blk[jp_next_block-1]['x'][1,:])
            dn['yj']=np.transpose(blk[jp_next_block-1]['y'][1,:]) + yoffset

            
        if (jp_next_patch == 2):
            yoffset=0.0
            if ((blk[ib]['y'][0,-1] - blk[jp_next_block-1]['y'][-1,0]) > 0.1*pitch):
                yoffset=pitch
            if ((blk[ib]['y'][0,-1] - blk[jp_next_block-1]['y'][-1,0]) < -0.1*pitch):
                yoffset=- pitch
            dn['xj']=np.transpose(blk[jp_next_block-1]['x'][-2,:])
            dn['yj']=np.transpose(blk[jp_next_block-1]['y'][-2,:]) + yoffset
            
            
        if (jp_next_patch == 3):
            yoffset=0.0
            if ((blk[ib]['y'][0,-1] - blk[jp_next_block-1]['y'][0,0]) > 0.1*pitch):
                yoffset=pitch
            if ((blk[ib]['y'][0,-1] - blk[jp_next_block-1]['y'][0,0]) < -0.1*pitch):
                yoffset=- pitch
            dn['xj']=blk[jp_next_block-1]['x'][:,1]
            dn['yj']=blk[jp_next_block-1]['y'][:,1] + yoffset

            
        if (jp_next_patch == 4):
            yoffset=0.0
            if ((blk[ib]['y'][0,-1] - blk[jp_next_block-1]['y'][0,-1]) > 0.1*pitch):
                yoffset=pitch
            if ((blk[ib]['y'][0,-1] - blk[jp_next_block-1]['y'][0,-1]) < -0.1*pitch):
                yoffset=- pitch
            dn['xj']=blk[jp_next_block-1]['x'][:,-2]
            dn['yj']=blk[jp_next_block-1]['y'][:,-2] + yoffset

            
    return up,dn