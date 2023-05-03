import numpy as np
import matplotlib.pyplot as plt

from .read_flo import *
from meshing.curve_length import *

def plot_spectrum(casename,blocks,iexclude): 

    prop,geom = read_flo(casename)

    spec = {} 

    xf = [[]]
    yf = [[]]
    uf = [[[]]]
    vf = [[[]]]
    wf = [[[]]]

    # concatenate block data - assuming periodic in j-direction 
    for n,nb in enumerate(blocks):
    
        x = geom[nb-1]['x']
        y = geom[nb-1]['y']
        
        u = prop[nb-1]['u']
        v = prop[nb-1]['v']
        w = prop[nb-1]['w']
    
    
        ni,nj,nk=np.shape(u)
    
        #U = np.tile(np.mean(u,axis=2),[1,1,nk])
        #V = np.tile(np.mean(v,axis=2),[1,1,nk])
        #W = np.tile(np.mean(w,axis=2),[1,1,nk])
        
        up = u# - U
        vp = v# - V
        wp = w# - W
    
        if(n==0):
           xf = x
           yf = y
           uf = up
           vf = vp
           wf = wp
        else:
           # concatenate data for spectrum
           xf = np.concatenate((x,xf),axis=1)
           yf = np.concatenate((y,yf),axis=1)
           uf = np.concatenate((up,uf),axis=1)
           vf = np.concatenate((vp,vf),axis=1)
           wf = np.concatenate((wp,wf),axis=1)
        
    
    ni,nj,nk=np.shape(up)
    
    # get spanwise average ffts
    v1 = 0.0
    v2 = 0.0
    v3 = 0.0
    Pxx= 0.0
    m  = 0.0
    istart = iexclude
    iend   = ni-iexclude
    
    for k in range(nk):
    
        for i in range(istart,iend):
        
            # sort in ascending distance from 1st point
            xnow = xf[i,:]
            ynow = yf[i,:]
            unow = uf[i,:,k]
            vnow = vf[i,:,k]
            wnow = wf[i,:,k]
            
            d = np.sqrt((xnow-xnow[0])*(xnow-xnow[0]) + (ynow - ynow[0])*(ynow - ynow[0])) 
            ii = np.argsort(d)
            xnow = xnow[ii]
            ynow = ynow[ii]
            unow = unow[ii]
            vnow = vnow[ii]
            wnow = wnow[ii]
            
            # get distance
            s = curve_length(xnow,ynow)

            # sort in distance
            ii = np.argsort(s)
            s = s[ii]
            xnow = xnow[ii]
            ynow = ynow[ii]
            unow = unow[ii]
            vnow = vnow[ii]
            wnow = wnow[ii]
            
            # subtract mean              
            unow = unow - np.mean(unow)
            vnow = vnow - np.mean(vnow)
            wnow = wnow - np.mean(wnow)

            # interpolate onto uniform y    
            yi = np.linspace(np.min(s),np.max(s),nj)
            ui = np.interp(yi,s,unow)
            vi = np.interp(yi,s,vnow)
            wi = np.interp(yi,s,wnow)
        
            m=m+1    
            v1 = v1 + np.fft.fft(ui)
            v2 = v2 + np.fft.fft(vi)
            v3 = v3 + np.fft.fft(wi)
        
        
    dy = yi[1]-yi[0]
    
    Ek = (v1*np.conjugate(v1) + v2*np.conjugate(v2) + v3*np.conjugate(v3))/np.float(m*nj)
    
    Ek = np.real(Ek) # convert to real  
    
    f = np.linspace(1.0e-12,1.0/dy,nj)
      
    
    imid = np.int(nj/2)
    imax = np.argmax(Ek)
    iplot = np.array(range(1,imid))    
    
    # Kolmogorov for inertial range
    Ekm = np.mean(Ek)
    f_kol =  f
    Ek_kol = Ek[imax]*((f_kol/f[imax])**(-5.0/3.0))
    
    
    plt.figure(1)
    
    plt.loglog(f[iplot], Ek[iplot],'-k.',f_kol[iplot], Ek_kol[iplot],'--k')
    plt.xlabel('Wavenumber (m^-1)')
    plt.ylabel('Ek (m^2 s^-2)')
    plt.title('Average power spectrum in blocks '+str(blocks))
    plt.show()  
    
        
    return 
