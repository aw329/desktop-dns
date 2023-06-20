# Copyright (c) 2023, University of Cambridge, all rights reserved. Written by Andrew Wheeler, University of Cambridge


import numpy as np

def yp(prop,geom,inlet,exit):

    inlet_blocks = inlet['blocks']
    exit_blocks = exit['blocks']
    
    xinlet = inlet['x']
    xwake = exit['x']

    poin = 0
    pin = 0
    pitch = 0
    mass = 0

    for i in inlet_blocks:
        x = geom[i-1]['x'][:,0]
        if(i==inlet_blocks[0]): ii = np.argmin(np.abs(x-xinlet))
        maxy = np.max(geom[i-1]['y'][ii,:])
        miny = np.min(geom[i-1]['y'][ii,:])
        pitch = pitch + (maxy-miny)
    
        xx = np.squeeze(geom[i-1]['x'])
        yy = np.squeeze(geom[i-1]['y'])
        po = np.squeeze(prop[i-1]['po'])
        p =  np.squeeze(prop[i-1]['p'])
        ro = np.squeeze(prop[i-1]['ro'])
        u =  np.squeeze(prop[i-1]['u'])
        v =  np.squeeze(prop[i-1]['v'])
        ru = ro*u
        rv = ro*v
        
        dy   = -yy[ii,:-1]+yy[ii,1:]
        dx   = -xx[ii,:-1]+xx[ii,1:]
        ruav = (ru[ii,:-1]+ru[ii,1:])*0.5
        rvav = (rv[ii,:-1]+rv[ii,1:])*0.5
        poav = (po[ii,:-1]+po[ii,1:])*0.5
        pav = (p[ii,:-1]+p[ii,1:])*0.5
        dm = ruav*dy + rvav*dx
        mass = mass + np.sum(dm)
        poin = poin + np.sum(poav*dm)
        pin = pin + np.sum(pav*dy)
    poin = poin/mass
    pin = pin/pitch

    ywake = []    
    Yp = [] 
    poex = 0
    pex = 0
    pitch = 0
    mex = 0
    
    for i in exit_blocks:
        x = geom[i-1]['x'][:,1]
        if(i==exit_blocks[0]):
           ii = np.argmin(np.abs(x-xwake))
           y0 = geom[i-1]['y'][ii,0]

        maxy = np.max(geom[i-1]['y'][ii,:])
        miny = np.min(geom[i-1]['y'][ii,:])
        pitch = pitch + (maxy-miny)
        y = geom[i-1]['y'][ii,:]-y0

        xx = np.squeeze(geom[i-1]['x'])
        yy = np.squeeze(geom[i-1]['y'])
        po = np.squeeze(prop[i-1]['po'])
        p =  np.squeeze(prop[i-1]['p'])
        ro = np.squeeze(prop[i-1]['ro'])
        u =  np.squeeze(prop[i-1]['u'])
        v =  np.squeeze(prop[i-1]['v'])
        ru = ro*u
        rv = ro*v

        dy   = -yy[ii,:-1]+yy[ii,1:]
        dx   = -xx[ii,:-1]+xx[ii,1:]
        ruav = (ru[ii,:-1]+ru[ii,1:])*0.5
        rvav = (rv[ii,:-1]+rv[ii,1:])*0.5
        poav = (po[ii,:-1]+po[ii,1:])*0.5
        pav = (p[ii,:-1]+p[ii,1:])*0.5
        dm = ruav*dy + rvav*dx
        mex = mex + np.sum(dm)
        poex = poex + np.sum(poav*dm)
        pex = pex + np.sum(pav*dy)

        if(y[0]<0): 
           ywake = np.append(yy[ii,:],ywake)
           Yp = np.append(po[ii,:],Yp)
        else:
           ywake = np.append(ywake,yy[ii,:])
           Yp = np.append(Yp,po[ii,:])

    poex = poex/mex
    pex = pex/pitch
    
    # identify if compresssor or turbine
    if(pex > pin): # compressor
        dyn = (poin - pin)
    else: # turbine
        dyn = (poin - pex)

# wake dictionary
    wake = {}
    wake['y']  = ywake/pitch
    wake['yp'] = (poin-Yp)/dyn

# performance dictionary
    perf = {}
    perf['wake'] = wake
    perf['yp'] = (poin-poex)/dyn
    perf['mass in'] = mass
    perf['mass out'] = mass
    perf['poin'] = poin
    perf['pin'] = pin
    perf['poex'] = poex
    perf['pex'] = pex
    perf['dyn'] = dyn

    #print(poin,poex,pin,pex,dyn,mass,mex)

    return perf


