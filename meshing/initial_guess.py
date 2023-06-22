# Copyright (c) 2023, University of Cambridge, all rights reserved. Written by Andrew Wheeler, University of Cambridge

import os
import numpy as np
from numpy import ndarray
from scipy.interpolate import interp1d,griddata

import matplotlib.pyplot as plt

from .read_profile import *
from .clip_curve import *
from .curve_length import *
from .spacing import *


def initial_guess(mesh,geom,bcs,gas,n,nj,Yp):
   # flow initial guess from blade profile

   alpha_in = bcs['alpha']
   vin      = bcs['vin']
   Toin     = bcs['Toin']
   Poin     = bcs['Poin']
   pexit    = bcs['pexit']
   
   gam  = gas['gamma']
   cp   = gas['cp']
   
   
   cv = cp/gam
   rgas = cp-cv

   Lup = (mesh['Lup'])*2.0
   Ldn = (mesh['Ldn'])*2.0
   xprof,yprof,pitch,_=read_profile(mesh['profile'],True)

   # get LE and TE points
   iLE = np.argmin(xprof)
   xLE = xprof[iLE]
   yLE = yprof[iLE]
   
   iTE = np.argmax(xprof)
   xTE = xprof[iTE]
   yTE = yprof[iTE]
   
    
   # split profile
   xi,yi,xi2,yi2=clip_curve(xprof,yprof,iLE,iTE,n)

  
   ii = np.array(range(1,n-1))
   dx = np.zeros([n])
   dy = np.zeros([n])   
   dx[ii] = xi[ii+1]-xi[ii-1]
   dy[ii] = yi[ii+1]-yi[ii-1]
   dx[0]  = dx[1]
   dx[-1] = dx[-2]
   dy[0]  = dy[1]
   dy[-1] = dy[-2]
   ds = np.sqrt(dx*dx + dy*dy)
   
   # create lower line
   xup = np.linspace(xLE - Lup,xLE,n)
   yup = np.linspace(yLE - Lup*np.tan(alpha_in*np.pi/180.0),xLE,n)

   xdn = np.linspace(xTE,xTE + Ldn,n)
   ydn = np.linspace(yTE,yTE + Ldn*dy[-2]/dx[-2],n)
   
   xs = np.concatenate( (xup,xi2[1:-2],xdn),axis=None )
   ys = np.concatenate( (yup,yi2[1:-2],ydn),axis=None )

   xp = np.concatenate( (xup,xi[1:-2],xdn),axis=None )
   yp = np.concatenate( (yup,yi[1:-2],ydn),axis=None ) 
   
   #ys = ys - pitch
   yp = yp + pitch


    

   
   ss = curve_length(xs,ys)
   ss = ss/ss[-1]
   ssi = np.linspace(-1.0,2.0,2000)
   f_x = interp1d(ss, xs, kind='cubic',fill_value="extrapolate")
   f_y = interp1d(ss, ys, kind='cubic',fill_value="extrapolate")
   xsi = f_x(ssi)
   ysi = f_y(ssi)

   sp = curve_length(xp,yp)
   sp = sp/sp[-1]
   spi = np.linspace(-1.0,2.0,2000)
   f_x = interp1d(sp, xp, kind='cubic',fill_value="extrapolate")
   f_y = interp1d(sp, yp, kind='cubic',fill_value="extrapolate")
   xpi = f_x(spi)
   ypi = f_y(spi)

   #plot(xs,ys,'-kx',xp,yp,'-rx'), axis equal

   # create h-mesh
   fex = 1.0
   fj = spacing(nj,fex,0)

   ni = len(xs)
   x     = np.zeros([ni,nj])
   y     = np.zeros([ni,nj])
   d_xi  = np.zeros([ni,nj])
   d_yi  = np.zeros([ni,nj])
   d_xj  = np.zeros([ni,nj])
   d_yj  = np.zeros([ni,nj])
   alpha = np.zeros([ni,nj])
   beta  = np.zeros([ni,nj])
   gamma = np.zeros([ni,nj])
   d2xij = np.zeros([ni,nj])
   d2yij = np.zeros([ni,nj])
   xiav  = np.zeros([ni,nj])
   xjav  = np.zeros([ni,nj])
   yiav  = np.zeros([ni,nj])
   yjav  = np.zeros([ni,nj])


   u  = np.ones([ni,nj])*vin*np.cos(alpha_in*np.pi/180.0)
   v  = np.ones([ni,nj])*vin*np.sin(alpha_in*np.pi/180.0)
   w  = np.zeros([ni,nj])
   ro = np.ones([ni,nj])*Poin/(rgas*Toin)

   for i in range(ni):
       x[i,:] = xs[i] + (xp[i] - xs[i])*fj    
       y[i,:] = ys[i] + (yp[i] - ys[i])*fj     

   floss = (x - xs[0])/(xs[-1] - xs[0])
   floss[ floss<0.0 ] = 0.0
   floss[ floss>1.0 ] = 1.0   
   
   for m in range(200):#range(200):
                     
       # smooth
       
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

       d_si = np.sqrt(d_xi*d_xi + d_yi*d_yi)
       d_sj = np.sqrt(d_xj*d_xj + d_yj*d_yj)

       gamma = d_xi*d_xi + d_yi*d_yi
       alpha = d_xj*d_xj + d_yj*d_yj
       beta  = d_xj*d_xi + d_yj*d_yi
                                  
       xiav[1:-1,:] = x[0:-2,:] + x[2:,:]
       xjav[:,1:-1] = x[:,0:-2] + x[:,2:]
       
       yiav[1:-1,:] = y[0:-2,:] + y[2:,:]
       yjav[:,1:-1] = y[:,0:-2] + y[:,2:]
                       
       # solve for x and y             
       x=0.5*(xiav*alpha - beta*d2xij*0.5 + xjav*gamma) / (alpha + gamma)
       y=0.5*(yiav*alpha - beta*d2yij*0.5 + yjav*gamma) / (alpha + gamma)    
                       
                   
       # wall conditions
       for i in range(ni):
           xnow = x[i,1]
           ynow = y[i,1]
           d = np.sqrt( (xnow-xsi)**2.0 + (ynow-ysi)**2.0 )
           inow=np.argmin(d)
           x[i,0] = xsi[inow]
           y[i,0] = ysi[inow]
           
           xnow = x[i,-2]
           ynow = y[i,-2]
           d = np.sqrt( (xnow-xpi)**2.0 + (ynow-ypi)**2.0 )
           inow=np.argmin(d)
           x[i,-1] = xpi[inow]
           y[i,-1] = ypi[inow]
           
       # inlet / exit conditions
       y[0,:]  = y[2,:]    - 2.0*(y[2,:]-y[1,:])
       x[0,:]  = x[2,:]    - 2.0*(x[2,:]-x[1,:])
       y[-1,:] = y[-2,:] + 2.0*(y[-2,:]-y[-3,:])
       x[-1,:] = x[-2,:] + 2.0*(x[-2,:]-x[-3,:])
       
       

      
   for iter in range(10):
      q  = (u*d_xi + v*d_yi)/d_si
      dh = (d_yj*d_xi - d_xj*d_yi)/np.sqrt(d_xi*d_xi + d_yi*d_yi)
          
      for i in range(1,ni):
           q[i,:] = ro[0,:]*q[0,:]*dh[0,:]/(ro[i,:]*dh[i,:])    
      
      # smooth q
      q[1:-1,:] = (q[0:-2,:] + 2.0*q[1:-1,:] + q[2:,:])/4.0
      q[:,1:-1] = (q[:,0:-2] + 2.0*q[:,1:-1] + q[:,2:])/4.0
      
      
      u = q*d_xi/d_si
      v = q*d_yi/d_si
          
      T = Toin - q*q*0.5/cp
      #dPo = 0.5*ro*q*q*Yp*(x - x[0,0])/(x[-1,-1] - x[0,0])
      dPo = 0.5*ro*vin*vin*Yp*floss
      p = (Poin-dPo)*((T/Toin)**(gam/(gam-1.0)))
      ro = p/(rgas*T)
          
  

   ## add boundary-layer and wake
   #fj = np.tile( np.linspace(0.001,0.999,nj),[ni,1])
   #fj2 = 1.0-fj
   #
   #fjj = 1.0/( (1.0/fj) + (1.0/fj2) )
   #fjj = fjj**(1.0/6.0)
   #fjj = fjj/np.max(fjj)
   #
   #ibl = fjj
   #ibl[x<xLE] = 1.0 
   #
   #print(np.max(fjj))
   #
   #u = u*ibl
   #v = v*ibl
   #w = w*ibl
  
  
   Et = p/(gam-1) + 0.5*ro*(u*u + v*v + w*w);
   ru = ro*u
   rv = ro*v
   rw = ro*w
   vel = np.sqrt(u*u + v*v + w*w)
   
  

   plt.figure(1)
   plt.axis('equal')
   plt.plot(xi2,yi2,'-r.')   
   plt.plot(xi,yi,'-g.')   
   plt.plot(xi2,yi2+pitch,'-r.')   
   plt.plot(xi,yi+pitch,'-g.')   
   plt.plot(x,y,'k')
   plt.plot(np.transpose(x),np.transpose(y),'k')
   
   #plt.pcolormesh(x,y,p,shading='gouraud')
   #plt.title('initial guess of pressure field')
   
   plt.pcolormesh(x,y,vel,shading='gouraud')
   plt.title('initial guess of velocity field')
   
   plt.show()   
          
   plt.figure(2)
   plt.plot(x[:,0],p[:,0],'k')
   plt.plot(x[:,-1],p[:,-1],'k')
   plt.xlabel('x')
   plt.ylabel('pressure')
   plt.title('estimated loading')
   plt.show()
     
   x = np.concatenate( (x,x,x,x,x),axis=1 )
   y = np.concatenate( (y-2.0*pitch,y-pitch,y,y+pitch,y+2.0*pitch),axis=1)
   ro = np.concatenate( (ro,ro,ro,ro,ro),axis=1 )
   ru = np.concatenate( (ru,ru,ru,ru,ru),axis=1 )
   rv = np.concatenate( (rv,rv,rv,rv,rv),axis=1 )
   rw = np.concatenate( (rw,rw,rw,rw,rw),axis=1 )
   Et = np.concatenate( (Et,Et,Et,Et,Et),axis=1 )
   
   ni,nj=np.shape(x)
   
   xv  = np.reshape(x ,(ni*nj))
   yv  = np.reshape(y ,(ni*nj))
   rov = np.reshape(ro,(ni*nj))
   ruv = np.reshape(ru,(ni*nj))
   rvv = np.reshape(rv,(ni*nj))
   rwv = np.reshape(rw,(ni*nj))
   Etv = np.reshape(Et,(ni*nj))
   
  
   #print(np.shape(xv))
   
   # now interpolate onto mesh
   plt.figure(3)
   #plt.pcolormesh(x,y,ro,shading='gouraud')   
   #plt.plot(x,y,'k')
   #plt.plot(np.transpose(x),np.transpose(y),'k')
   #plt.clim([1.1,1.15])
      
   Nb = len(geom)
   prop = Nb*[None]
   for ib in range(Nb):
   
       xb = geom[ib]['x']
       yb = geom[ib]['y']
   
       ni,nj = np.shape(xb) 
       
       xmin = np.amin(xb)
       ymin = np.amin(yb)

       xmax = np.amax(xb)
       ymax = np.amax(yb)
       
       xn = np.reshape((xb-xmin)/(xmax-xmin),[ni*nj])
       yn = np.reshape((yb-ymin)/(ymax-ymin),[ni*nj])
       
       xnv = (xv-xmin)/(xmax-xmin)
       ynv = (yv-ymin)/(ymax-ymin)
            
       points = (xnv,ynv)
       pointsi = (xn,yn)
       
       prop[ib] = {}    
 
       prop[ib]['ro'] = np.reshape(griddata(points, rov, pointsi, method='linear'),[ni,nj])
       prop[ib]['ru'] = np.reshape(griddata(points, ruv, pointsi, method='linear'),[ni,nj])
       prop[ib]['rv'] = np.reshape(griddata(points, rvv, pointsi, method='linear'),[ni,nj])
       prop[ib]['rw'] = np.reshape(griddata(points, rwv, pointsi, method='linear'),[ni,nj])
       prop[ib]['Et'] = np.reshape(griddata(points, Etv, pointsi, method='linear'),[ni,nj])   
       
       
       plt.pcolormesh(xb,yb, prop[ib]['Et'],shading='gouraud')
       plt.clim([np.min(Etv),np.max(Etv)])
       
       
   plt.title('interpolated flow (Et)')
   plt.axis('equal')
   plt.show()
       
   return prop



