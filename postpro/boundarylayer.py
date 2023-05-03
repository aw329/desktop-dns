import os
import numpy as np
import matplotlib.pyplot as plt

from scipy.interpolate import interp1d
from meshing.read_case import *
from .read_flo import *
from .grad import *

def boundarylayer(casename):
    
    flo = {}
        
    path = os.getcwd()
    
    # get case details  
    case = read_case(casename)
    
    # get flow and geom
    prop,blk = read_flo(casename)
      
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
    
    
    # patch info
    next_block = case['next_block']
    next_patch = case['next_patch']

    Nb = len(blk) 
    
    # initialize arrays
    IPS   = np.asarray([],dtype=float)
    ISS   = np.asarray([],dtype=float)
    volbl = np.asarray([],dtype=float)
    x0    = np.asarray([],dtype=float)
    y0    = np.asarray([],dtype=float)
    d0    = np.asarray([],dtype=float)
    d1    = np.asarray([],dtype=float)
    d2    = np.asarray([],dtype=float)
    d3    = np.asarray([],dtype=float)
    d1i   = np.asarray([],dtype=float)
    d2i   = np.asarray([],dtype=float)
    d3i   = np.asarray([],dtype=float)
    d4    = np.asarray([],dtype=float)
    d5    = np.asarray([],dtype=float)
    d6    = np.asarray([],dtype=float)
    d7    = np.asarray([],dtype=float)
    d8    = np.asarray([],dtype=float)
    d9    = np.asarray([],dtype=float)
    d10   = np.asarray([],dtype=float)
    yplus = np.asarray([],dtype=float)
    xplus = np.asarray([],dtype=float)
    zplus = np.asarray([],dtype=float)
    tauw  = np.asarray([],dtype=float)
    cf    = np.asarray([],dtype=float)
    XX    = np.asarray([],dtype=float)
    YY    = np.asarray([],dtype=float)
    
    
    
        
    ns = -1
    for nb in range(Nb):
        
        x = blk[nb]['x']
        y = blk[nb]['y']
        
        ro = prop[nb]['ro']
        mu = prop[nb]['mu']
        u  = prop[nb]['u']
        v  = prop[nb]['v']
        w  = prop[nb]['w']
        p  = prop[nb]['p']
        po = prop[nb]['po']
        To = prop[nb]['To']
        
        fspan = os.path.join(path,casename,'span_'+str(nb+1)+'.txt')
        span = np.loadtxt(fspan)
    
        try:
           mut = prop[nb]['mut_model']
        except:
           mut = np.zeros(np.shape(mu))
        
        try:
           delz = span[1,0]-span[0,0]
        except:
           delz = 0.0
        
        try:
           pr = prop[nb]['turb_production']
        except:
           pr = np.zeros(np.shape(mu))
         
        if(len(np.shape(ro))==3):
           ro = np.mean(ro, axis=2)
           mu = np.mean(mu, axis=2)
           u  = np.mean(u,  axis=2)
           v  = np.mean(v,  axis=2)
           w  = np.mean(w,  axis=2)
           p  = np.mean(p,  axis=2)
           po = np.mean(po, axis=2)
           To = np.mean(To, axis=2)
           pr = np.mean(pr, axis=2)
           mut= np.mean(mut,axis=2)
           
        
        im_wall = (next_block[nb]['im'] == 0 and next_patch[nb]['im'] >= 3)
        ip_wall = (next_block[nb]['ip'] == 0 and next_patch[nb]['ip'] >= 3)
        jm_wall = (next_block[nb]['jm'] == 0 and next_patch[nb]['jm'] >= 3)
        jp_wall = (next_block[nb]['jp'] == 0 and next_patch[nb]['jp'] >= 3)
        
        wall = im_wall or ip_wall or jm_wall or jp_wall
        
        if(wall):
        
            # transpose matrices so that wall is always on jm boundary
            if(im_wall):
               x  = x.T
               y  = y.T
               ro = ro.T
               mu = mu.T
               u  = u.T
               v  = v.T
               w  = w.T
               p  = p.T
               po = po.T
               To = To.T
               pr = pr.T   
               mut= mut.T
            
            if(ip_wall):
               x  = x[::-1,:]
               y  = y[::-1,:]
               ro = ro[::-1,:]
               mu = mu[::-1,:]
               u  = u[::-1,:]
               v  = v[::-1,:]
               w  = w[::-1,:]
               p  = p[::-1,:]
               po = po[::-1,:]
               To = To[::-1,:]
               pr = pr[::-1,:]   
               mut= mut[::-1,:]
               
               x  = x.T
               y  = y.T
               ro = ro.T
               mu = mu.T
               u  = u.T
               v  = v.T
               w  = w.T
               p  = p.T
               po = po.T
               To = To.T
               pr = pr.T   
               mut= mut.T
               
            if(jp_wall):
               x  = x[:,::-1]
               y  = y[:,::-1]
               ro = ro[:,::-1]
               mu = mu[:,::-1]
               u  = u[:,::-1]
               v  = v[:,::-1]
               w  = w[:,::-1]
               p  = p[:,::-1]
               po = po[:,::-1]
               To = To[:,::-1]
               pr = pr[:,::-1]   
               mut= mut[:,::-1]
               
            
            ni,nj=np.shape(x)
            vel = np.sqrt(u*u + v*v + w*w)
        
            psurf  = p[:,0]
            Tosurf = To[:,0]
            pomax  = np.max(po,axis=1)
            umax   = np.max(vel,axis=1)    
            
            psurf[psurf>pomax]=pomax[psurf>pomax]
        
            minf = np.sqrt((((psurf/pomax)**(-(gam-1.0)/gam))-1.0)*2.0/(gam-1.0))
            Tinf = Tosurf*((psurf/pomax)**((gam-1.0)/gam))
            uinf = minf*np.sqrt(gam*rgas*Tinf)
        
            ie = uinf > umax
            uinf[ie] = umax[ie]
            
            # put limit where uinf is very small
            uinf[uinf<=0.001]=0.001
            
            roinf = psurf/(rgas*Tinf)
            ruinf = roinf*uinf
            
            edgej = np.zeros(ni)
            
            
            for i in range(ni):
                
                ns = ns + 1
                
                if(i==0):
                   dxt = (x[i+1,0]-x[i,0])    
                   dyt = (y[i+1,0]-y[i,0])    
                elif(i==ni-1):
                   dxt = (x[i,0]-x[i-1,0])    
                   dyt = (y[i,0]-y[i-1,0])    
                else:
                   dxt = (x[i+1,0]-x[i-1,0])*0.5    
                   dyt = (y[i+1,0]-y[i-1,0])*0.5       
                
                # guess b.layer height
                if(ns>0):
                   if(d1[ns-1]>0.0 and d2[ns-1]>0.0):
                      hprev = d1[ns-1]/d2[ns-1]
                   else:
                      hprev = 2.5   
                   if(hprev<0):hprev = 2.50
                   Del = d2[ns-1]*(3.15 + 1.72/(hprev-1.0)) + d1[ns-1]   
                   #print(hprev,d0[ns-1],d1[ns-1],d2[ns-1])                   
                else:
                   Del = 1.0e-4
                
                ydist = y[i,3]-y[i,0]
                xdist = x[i,3]-x[i,0]
                 
                IPS=np.append(IPS,(ydist < 0))
                ISS=np.append(ISS,(ydist > 0))
               
             
                
                yprof = y[i,:]
                xprof = x[i,:]
                uprof = u[i,:]
                vprof = v[i,:]
                wprof = w[i,:]
                roprof = ro[i,:]
                turbprof = pr[i,:]/ro[i,:]
                muprof = mu[i,:]
                mutprof = mut[i,:]
                
                nprof = 10000
                
                xd = xprof-xprof[0]
                yd = yprof-yprof[0]
                #rprof = np.sqrt(xd*xd + yd*yd)
                #velprof = np.sqrt(uprof*uprof + vprof*vprof + wprof*wprof)
                
                rprof = np.abs(-xd*dyt + yd*dxt)/np.sqrt(dxt*dxt + dyt*dyt)
                velprof =(uprof*dxt + vprof*dyt)/np.sqrt(dxt*dxt + dyt*dyt)
                
                dvel = velprof[2]  
                dyy  = rprof[2]  
             
                muw = muprof[0]
                row = roprof[0]
                tw = muw*dvel/dyy #muw*velprof[2]/rprof[2]    
                ut = np.sqrt(np.abs(tw)/row)
                yp = ut*row*rprof[1]/muw
                xp = yp*np.sqrt(dxt*dxt + dyt*dyt)/rprof[1]
                zp = yp*delz/rprof[1]
                
                yplus=np.append(yplus,yp)
                xplus=np.append(xplus,xp)
                zplus=np.append(zplus,zp)
               
                tauw=np.append(tauw,tw)
                cf=np.append(cf,tw/(0.5*ruinf[i]*uinf[i]))
                XX=np.append(XX,xprof[0])
                YY=np.append(YY,yprof[0])
                
                ri = np.linspace(np.min(rprof),np.max(rprof),nprof)
                
                f = interp1d(rprof, yprof, kind='cubic')
                yi =  f(ri)
                
                f = interp1d(rprof, xprof, kind='cubic')
                xi =  f(ri)
                
                f = interp1d(rprof, uprof, kind='cubic')
                ui =  f(ri)
        
                f = interp1d(rprof, vprof, kind='cubic')
                vi =  f(ri)
        
                f = interp1d(rprof, wprof, kind='cubic')
                wi =  f(ri)
                
                f = interp1d(rprof, muprof, kind='cubic')
                mui =  f(ri)
        
                f = interp1d(rprof, mutprof, kind='cubic')
                muti = f(ri)
        
                f = interp1d(rprof, roprof, kind='cubic')
                roi =  f(ri)
        
                f = interp1d(rprof, turbprof, kind='cubic')
                turbi =f(ri)
        
                vmagi = np.sqrt(ui*ui + vi*vi + wi*wi)
                
                if(ns<100): # correction near LE
                   ii = vmagi < uinf[i]*0.99999
                   
                else:
                   ii = ri < Del
                  
                edgen = np.argmax(ri*ii)
                edgej[i] = np.argmin(abs(rprof-ri[edgen]))
                   
                   
                dx = xi[1:]-xi[:-1]
                dy = yi[1:]-yi[:-1]
                
                dely = np.abs(-dx*dyt + dy*dxt)/np.sqrt(dxt*dxt + dyt*dyt) 
                if(edgen > (len(dely)-1)): edgen = len(dely)-1                
                del_now = np.sum(dely[:edgen])

                # recompute bl edge estimate
                if(del_now > Del):
                   edgen = np.argmax(ri*ii)
                   edgej[i] = np.argmin(abs(rprof-ri[edgen]))
                
                
                delv = vmagi[1:]-vmagi[:-1]
                 
                uav = (ui[1:]+ui[:-1])*0.5
                vav = (vi[1:]+vi[:-1])*0.5
                wav = (wi[1:]+wi[:-1])*0.5
                                     
                turbav =(turbi[1:]+turbi[:-1])*0.5
                ronow  =  (roi[1:] + roi[:-1])*0.5
                munow  =  (mui[1:] + mui[:-1])*0.5
                mutnow = (muti[1:]+ muti[:-1])*0.5
                    
                
                vnow = (uav*dxt + vav*dyt)/np.sqrt(dxt*dxt + dyt*dyt) 
                diss_av = (munow/ronow)*(delv/dely)*(delv/dely)
                diss_model = (munow*mutnow/ronow)*(delv/dely)*(delv/dely)    
  
                vnow[vnow>uinf[i]]=uinf[i] 
  
                x0=np.append(x0,xi[edgen])
                y0=np.append(y0,yi[edgen])
                
                vn    = vnow/uinf[i]
                rvn   = ronow*vnow/ruinf[i]
                rvdef = 1.0 - rvn
                vdef  = 1.0 - vn 
                v2def = 1.0 - vn*vn
                                
                d0=np.append(d0,np.sum(dely[:edgen]))
                d1=np.append(d1,np.sum(rvdef[:edgen]*dely[:edgen]))
                d2=np.append(d2,np.sum(rvn[:edgen]*vdef[:edgen]*dely[:edgen]))
                d3=np.append(d3,np.sum(rvn[:edgen]*v2def[:edgen]*dely[:edgen]))
                
                d1i=np.append(d1i,np.sum(vdef[:edgen]*dely[:edgen]))
                d2i=np.append(d2i,np.sum(vn[:edgen]*vdef[:edgen]*dely[:edgen]))
                d3i=np.append(d3i,np.sum(vn[:edgen]*v2def[:edgen]*dely[:edgen]))
                
                d9=np.append(d9 ,  np.sum(diss_av[:edgen]*dely[:edgen]))
                d10=np.append(d10, np.sum((turbav[:edgen] + diss_model[:edgen])*dely[:edgen]))
                 
                   
                #ue(ns) = uinf(i);
                #retheta(ns) = d2(ns)*ruinf(i)/mu(i,nj);
                #cf(ns) = walls.tauw(ns)/(0.5*ruinf(i)*uinf(i));
                #cd(ns) = (d10(ns) + d9(ns))/(ue(ns)^3);
                #Us(ns) = (d9(ns)*2.0/cf(ns))/(ue(ns)^3);
                #prod(ns) = d10(ns)/(ue(ns)^3);
                #ctau(ns) = prod(ns)/(1-Us(ns));
                #
                #XX(ns) = x(i,1);
                #YY(ns) = y(i,1);
                
    #plt.figure(1)
    #plt.plot(XX,d0,'k.',XX,d1,'r.',XX,d2,'b.')  
    #plt.show()
    
    xLE = min(XX)
    xTE = max(XX)
    cax = xTE-xLE
    #
    xn = (XX-xLE)/cax
    #
    #ISS = ISS and (xn<0.95 and xn>0.05)
    #IPS = IPS and (xn<0.95 and xn>0.05)
    #
    
     
    
    bl = {}
    bl['ss'] = {}
    bl['ps'] = {}
    
    ISS = ISS > 0
    IPS = IPS > 0
   
    
    bl['ss']['x'   ] = XX[ISS]
    bl['ss']['y'   ] = YY[ISS]
    bl['ss']['d0'  ] = d0[ISS]
    bl['ss']['d1'  ] = d1[ISS]
    bl['ss']['d2'  ] = d2[ISS]
    bl['ss']['d3'  ] = d3[ISS]
    #bl['ss']['d10' ] =d10[ISS]
    bl['ss']['cf'  ] = cf[ISS]
    bl['ss']['yplus'  ] = yplus[ISS]
    bl['ss']['xplus'  ] = xplus[ISS]
    bl['ss']['zplus'  ] = zplus[ISS]
   
    #bl['ss']['cd'  ] = cd[ISS]
    #bl['ss']['Ret' ] =ret[ISS]
    #bl['ss']['ue'  ] = ue[ISS]
    #bl['ss']['us'  ] = Us[ISS]
    #bl['ss']['ctau']=ctau[ISS]
    #bl['ss']['prod']=prod[ISS]
   
    bl['ps']['x'   ] = XX[IPS]
    bl['ps']['y'   ] = YY[IPS]
    bl['ps']['d0'  ] = d0[IPS]
    bl['ps']['d1'  ] = d1[IPS]
    bl['ps']['d2'  ] = d2[IPS]
    bl['ps']['d3'  ] = d3[IPS]
    #bl['ps']['d10' ] =d10[IPS]
    bl['ps']['cf'  ] = cf[IPS]
    bl['ps']['yplus'  ] = yplus[IPS]
    bl['ps']['xplus'  ] = xplus[IPS]
    bl['ps']['zplus'  ] = zplus[IPS]
   
    #bl['ps']['cd'  ] = cd[IPS]
    #bl['ps']['Ret' ] =ret[IPS]
    #bl['ps']['ue'  ] = ue[IPS]
    #bl['ps']['us'  ] = Us[IPS]
    #bl['ps']['ctau']=ctau[IPS]
    #bl['ps']['prod']=prod[IPS]
    
    
    return bl


