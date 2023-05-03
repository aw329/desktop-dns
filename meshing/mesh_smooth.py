from .curve_length   import *
from .multiblock import *
from .make_periodic_smooth2 import *
from .make_periodic import *
from .fexpan import *
from .spacing import *


from joblib import Parallel, delayed

import numpy as np
from scipy.interpolate import interp1d



    
# define smooth command function to parallelize     
def smooth(next_block,next_patch,b,x_prof,y_prof,s_prof,x_prof2,y_prof2,s_prof2,ywall,up,dn):
# smooth routine

       s_len=s_prof[-1]
       
       im_next_block=next_block['im']
       ip_next_block=next_block['ip']
       jm_next_block=next_block['jm']
       jp_next_block=next_block['jp']
       im_next_patch=next_patch['im']
       ip_next_patch=next_patch['ip']
       jm_next_patch=next_patch['jm']
       jp_next_patch=next_patch['jp']
                   
       im_wall=im_next_block == 0 and im_next_patch == 3
       ip_wall=ip_next_block == 0 and ip_next_patch == 3
       jm_wall=jm_next_block == 0 and jm_next_patch == 3
       jp_wall=jp_next_block == 0 and jp_next_patch == 3
       im_in=im_next_block == 0 and im_next_patch == 1
       ip_ex=ip_next_block == 0 and ip_next_patch == 2
   
       xnew=b['x']
       ynew=b['y']
   
       ni_new,nj_new=np.shape(xnew)
       x=xnew
       y=ynew
   
       d_xi  = np.zeros([ni_new,nj_new])
       d_yi  = np.zeros([ni_new,nj_new])
       d_xj  = np.zeros([ni_new,nj_new])
       d_yj  = np.zeros([ni_new,nj_new])
       alpha = np.zeros([ni_new,nj_new])
       beta  = np.zeros([ni_new,nj_new])
       gamma = np.zeros([ni_new,nj_new])
       d2xij = np.zeros([ni_new,nj_new])
       d2yij = np.zeros([ni_new,nj_new])
       xiav  = x*2.0
       xjav  = x*2.0
       yiav  = y*2.0
       yjav  = y*2.0

       # get grid-line angles
       d_xi[1:-1,:] =  (x[2:,:] - x[0:-2,:])*0.5
       d_yi[1:-1,:] =  (y[2:,:] - y[0:-2,:])*0.5
       d_xi[0,:]  =  (x[1,:]    - x[0,:])
       d_yi[0,:]  =  (y[1,:]    - y[0,:])
       d_xi[-1,:] =  (x[-1,:]   - x[-2,:])
       d_yi[-1,:] =  (y[-1,:]   - y[-2,:])

       
       d_xj[:,1:-1] =  (x[:,2:] - x[:,0:-2])*0.5
       d_yj[:,1:-1] =  (y[:,2:] - y[:,0:-2])*0.5
       d_xj[:,0]  =  (x[:,1]    - x[:,0])
       d_yj[:,0]  =  (y[:,1]    - y[:,0])
       d_xj[:,-1] =  (x[:,-1]   - x[:,-2]);
       d_yj[:,-1] =  (y[:,-1]   - y[:,-2]);

       d2xij[1:-1,1:-1] = x[2:,2:] - x[2:,0:-2] + x[0:-2,0:-2] - x[0:-2,2:] 
       d2yij[1:-1,1:-1] = y[2:,2:] - y[2:,0:-2] + y[0:-2,0:-2] - y[0:-2,2:] 

       if (im_next_block != 0):               
           xiav[0,:]  = x[1,:] + up['xi']
           yiav[0,:]  = y[1,:] + up['yi']
           d_xi[0,:]  = x[1,:] - up['xi']
           d_yi[0,:]  = y[1,:] - up['yi']
           d2xij[0,1:-1]=x[1,2:] - x[1,0:-2] + up['xi'][0:-2] - up['xi'][2:]
           d2yij[0,1:-1]=y[1,2:] - y[1,0:-2] + up['yi'][0:-2] - up['yi'][2:]
           

       if (ip_next_block != 0):               
           xiav[-1,:]  = x[-2,:] + dn['xi']
           yiav[-1,:]  = y[-2,:] + dn['yi']
           d_xi[-1,:]  =-x[-2,:] + dn['xi']
           d_yi[-1,:]  =-y[-2,:] + dn['yi']
           d2xij[-1,1:-1]=-x[-2,2:] + x[-2,0:-2] - dn['xi'][0:-2] + dn['xi'][2:]
           d2yij[-1,1:-1]=-y[-2,2:] + y[-2,0:-2] - dn['yi'][0:-2] + dn['yi'][2:]
           

       if (jm_next_block != 0):               
           xjav[:,0]  = x[:,1] + up['xj']
           yjav[:,0]  = y[:,1] + up['yj']
           d_xj[:,0]  = x[:,1] - up['xj']
           d_yj[:,0]  = y[:,1] - up['yj']
           d2xij[1:-1,0]=x[2:,1] - x[0:-2,1] + up['xj'][0:-2] - up['xj'][2:]
           d2yij[1:-1,0]=y[2:,1] - y[0:-2,1] + up['yj'][0:-2] - up['yj'][2:]


       if (jp_next_block != 0):               
           xjav[:,-1]  = x[:,-2] + dn['xj']
           yjav[:,-1]  = y[:,-2] + dn['yj']
           d_xj[:,-1]  =-x[:,-2] + dn['xj']
           d_yj[:,-1]  =-y[:,-2] + dn['yj']
           d2xij[1:-1,-1]=-x[2:,-2] + x[0:-2,-2] - dn['xj'][0:-2] + dn['xj'][2:]
           d2yij[1:-1,-1]=-y[2:,-2] + y[0:-2,-2] - dn['yj'][0:-2] + dn['yj'][2:]
   
           
       gamma = d_xi*d_xi + d_yi*d_yi
       alpha = d_xj*d_xj + d_yj*d_yj
       beta  = d_xj*d_xi + d_yj*d_yi
       rotn  = d_xi*d_yj - d_yi*d_xj            
                   
                       
       xiav[1:-1,:] = x[0:-2,:] + x[2:,:]
       xjav[:,1:-1] = x[:,0:-2] + x[:,2:]
       
       yiav[1:-1,:] = y[0:-2,:] + y[2:,:]
       yjav[:,1:-1] = y[:,0:-2] + y[:,2:]
                       
       # solve for x and y             
       xnew=0.5*(xiav*alpha - beta*d2xij*0.5 + xjav*gamma) / (alpha + gamma)
       ynew=0.5*(yiav*alpha - beta*d2yij*0.5 + yjav*gamma) / (alpha + gamma)    
             
   
       # Apply boundary conditions
       # build o-grid
       if (jp_wall):
       
           xprof=x[:,-1]
           yprof=y[:,-1]
           sprof=curve_length(xprof,yprof)
           
           
           s_len_local = sprof[-1]-sprof[0]
           
           si=np.linspace(sprof[0]-s_len_local*0.1,sprof[-1]+s_len_local*0.1,ni_new*100)
           
           
           sprof,iu=np.unique(sprof,True)
           xprof=xprof[iu]
           yprof=yprof[iu]
           
           istart=np.argmin(np.sqrt( (x_prof - xprof[0]) ** 2 + (y_prof - yprof[0]) ** 2))
           rotn_max = np.amax(rotn[:,-1])
           rotn_min = np.amin(rotn[:,-1])
                        
           if (rotn_max > 0.0):
               istart=np.argmin(np.sqrt( (x_prof - xprof[-1]) ** 2 + (y_prof - yprof[-1]) ** 2))
                            
           sstart=s_prof[istart]
           
           si=si + sstart
           si[si > s_len]=si[si > s_len] - s_len
           si[si < 0]    =si[si < 0]     + s_len
           
           f_x = interp1d(s_prof2, x_prof2, kind='cubic')
           f_y = interp1d(s_prof2, y_prof2, kind='cubic')
           
           xi = f_x(si)
           yi = f_y(si)
           
           
           for i in range(ni_new):
               # find nearest wall point to set orthogonality
               d=np.sqrt((x[i,-2] - xi) ** 2 + (y[i,-2] - yi) ** 2)
               inorm=np.argmin(d)
               xnew[i,-1]=xi[inorm]
               ynew[i,-1]=yi[inorm]
               
           # now drive near wall distance to ywall
           for i in range(ni_new):
               ynorm=curve_length(xnew[i,:],ynew[i,:])
               fex=fexpan(ynorm[-1] / ywall,nj_new)
               fy=spacing(nj_new,1.0 / fex,0)
               ynorm=ynorm / ynorm[-1]
               yni=fy
               
               f_x = interp1d(ynorm, xnew[i,:], kind='cubic')
               f_y = interp1d(ynorm, ynew[i,:], kind='cubic')                   
               
               xnew[i,:]=f_x(yni)
               ynew[i,:]=f_y(yni)
               
           
   
       if (im_in):
           ynew[0,:]=ynew[1,:]
       if (ip_ex):
           ynew[-1,:]=ynew[-2,:]
           
       b['x'] = xnew
       b['y'] = ynew
           
       return b
   
    
    


def mesh_smooth(blk,next_block,next_patch,corner,pitch,msmooths,x_prof,y_prof,ywall,cor_fac):

    NB=len(blk)
    nib = {}
    njb = {}
    up = {}
    dn = {}
    

    
    for i in range(NB):
         nib[i],njb[i] = np.shape(blk[i]['x'])
    
    # ensure wall profile remains fixed
    s_prof=curve_length(x_prof,y_prof)
    s_len=s_prof[-1]
    N = len(s_prof)
    ii = np.array(range(1,N-1))
    
    s_prof2=np.concatenate((s_prof-s_len,s_prof[ii],s_prof+s_len),axis=None)
    x_prof2=np.concatenate((x_prof,x_prof[ii],x_prof),axis=None)
    y_prof2=np.concatenate((y_prof,y_prof[ii],y_prof),axis=None)


    for mm in range(msmooths):

        # call multiblock to find the block boundary interfaces
        for i in range(NB):
            up[i] = {}
            dn[i] = {}
            up[i],dn[i]=multiblock(blk,next_block,next_patch,i,pitch)
        
       
        # run smooth in parallel    
        res = Parallel(n_jobs=NB)\
        (delayed(smooth)(next_block[i],next_patch[i],blk[i],\
        x_prof,y_prof,s_prof,x_prof2,y_prof2,s_prof2,ywall,up[i],dn[i])\
        for i in range(NB)) 
        blk = res
        
        # run smooth in serial         
        #for i in range(NB):
        #    blk[i] = smooth(next_block[i],next_patch[i],blk[i],
        #    x_prof,y_prof,s_prof,x_prof2,y_prof2,s_prof2,ywall,up[i],dn[i])
            
         
       
        # enforce periodics
        for i in range(NB):
            blk=make_periodic(blk,next_block,next_patch,i,pitch)     
        
        # Now corner treatment
        # in this section the grid points where i=j are shrunk toward the corner
        # point
        ncorner=len(corner)
        xcor=np.zeros(ncorner)
        ycor=np.zeros(ncorner)
        xc=np.zeros(ncorner)
        yc=np.zeros(ncorner)
        yc_0=np.zeros(ncorner)
        for n in range(ncorner):
            dely=0.0
            for m in range(corner[n]['Nb']):
                ib=corner[n]['block'][m]-1
                ic=corner[n]['i'][m]
                jc=corner[n]['j'][m]
                
                if ic != 1:
                    ic=nib[ib]
                if jc != 1:
                    jc=njb[ib]
                if (m == 0):
                    yc0=blk[ib]['y'][ic-1,jc-1]
                yoffset=0
                if ((blk[ib]['y'][ic-1,jc-1] - yc0) > pitch*0.5):
                    yoffset= -pitch
                if ((blk[ib]['y'][ic-1,jc-1] - yc0) < -pitch*0.5):
                    yoffset= pitch
                    
                xcor[n]=xcor[n] + blk[ib]['x'][ic-1,jc-1]
                ycor[n]=ycor[n] + blk[ib]['y'][ic-1,jc-1] + yoffset
                
            xc[n]=xcor[n] / corner[n]['Nb']
            yc[n]=ycor[n] / corner[n]['Nb']
            yc_0[n]=yc0
            xx=xc[n]
            yy=yc[n]

            for m in range(corner[n]['Nb']):
                ib=corner[n]['block'][m]-1
                ic=corner[n]['i'][m]
                jc=corner[n]['j'][m]
                
                if ic != 1:
                    ic=nib[ib]
                if jc != 1:
                    jc=njb[ib]
                yoffset=0
                if ((blk[ib]['y'][ic-1,jc-1] - yc_0[n]) > pitch*0.5):
                    yoffset=- pitch
                if ((blk[ib]['y'][ic-1,jc-1] - yc_0[n]) < -pitch*0.5):
                    yoffset=pitch
                xx=xc[n]
                yy=yc[n] - yoffset
                ni_new,nj_new=np.shape(blk[ib]['x'])
                blk[ib]['x'][ic-1,jc-1]=xx
                blk[ib]['y'][ic-1,jc-1]=yy
                #
                nic=16
                if(nic>=ni_new):
                   nic = ni_new - 1
                if(nic>=nj_new):
                   nic = nj_new - 1   
                   
                fc=np.ones([nic,nic])
                if (corner[n]['Nb'] == 5):
                    f = np.array(range(1,nic+1))/nic
                    f = np.tanh(f*nic*cor_fac)
                    #f = np.tanh(f*nic/2.0)
                    #fc=fc*nic
                    #fc[:,0]=np.array(range(1,nic+1))
                    #fc[0,:]=np.array(range(1,nic+1))
                    #fc=fc / nic
                    #fc=fc ** 0.5
                    fc[:,0] = f
                    fc[0,:] = f
                    
                    # now smooth fc
                    fcnow = fc
                    for j in range(1,nic-1):
                        for i in range(1,nic-1):
                            fc[i,j] = (fcnow[i+1,j] + fcnow[i-1,j] \
                            +          fcnow[i,j+1] + fcnow[i,j-1] )*0.25  
                    
                    
                istart = 0
                iend = nic
                jstart = 0
                jend = nic
                iflag = 1
                jflag = 1
                
                if (ic != 1):
                    istart = ni_new-1
                    iend = ni_new-nic-1
                    iflag = -1

                if (jc != 1):
                    jstart = nj_new-1
                    jend = nj_new-nic-1
                    jflag = -1
                
                #print(istart,iend,iflag,jstart,jend,jflag)          
                
                
                blk[ib]['x'][istart:iend:iflag,jstart:jend:jflag]= \
                xx + (blk[ib]['x'][istart:iend:iflag,jstart:jend:jflag] - xx)*fc
                
                blk[ib]['y'][istart:iend:iflag,jstart:jend:jflag]= \
                yy + (blk[ib]['y'][istart:iend:iflag,jstart:jend:jflag] - yy)*fc                
                
                
    # enforce periodics
    for i in range(NB):
        blk=make_periodic(blk,next_block,next_patch,i,pitch)     
        
    return blk