# Copyright (c) 2023, University of Cambridge, all rights reserved. Written by Andrew Wheeler, University of Cambridge

import numpy as np
from numpy import matlib
from scipy import interpolate
from .clip_curve import *
from .curve_length import *
from .mesh_smooth import *

def blade_topology(xprof,yprof,pitch,Lup,Ldn,Lo,stag,npp,ywall,msmooths,cor_fac):
# create 9-block topology with o-grid
# returns a coarse mesh defining an initial topology

# set up dictionaries
    blk = {}
    next_block = {}
    next_patch = {}
    corner = {}
    
    for i in range(9):
        blk[i] = {}
        next_block[i] = {}
        next_patch[i] = {}
        
    for i in range(4):     
        corner[i] = {}    
    
    
    pi = np.pi
    N=len(xprof)
# get LE and TE points    
    iLE = np.argmin(xprof)
    iTE = np.argmax(xprof)
    
    xLE = xprof[iLE]
    yLE = yprof[iLE]
    
    xTE = xprof[iTE]
    yTE = yprof[iTE]
    
    # O-grid boundary
    xo = np.zeros(N)
    yo = np.zeros(N)
    for i in range(N):
        im1=i - 1
        ip1=i + 1
        if (i == 0):
            im1=N - 2
        if (i == N-1):
            ip1=1
        dx=xprof[ip1] - xprof[im1]
        dy=yprof[ip1] - yprof[im1]
        ds=np.sqrt(dx*dx + dy*dy)
        xo[i]=xprof[i] - Lo*dy/ds
        yo[i]=yprof[i] + Lo*dx/ds
    
    xo[-1]=(xo[0] + xo[-1])*0.5
    xo[-1]=(yo[0] + yo[-1])*0.5
    xo[0]=xo[-1]
    yo[0]=yo[-1]

#plot bounding box
#rotate by stagger about LE to find bounding box;
    xdat=xLE
    ydat=yLE
    xr=xo - xdat
    yr=yo - ydat

    if (stag > 30*pi/ 180):
        stag= 30*pi/180
    stag = -stag
    xr_now=xr*np.cos(stag) - yr*np.sin(stag)
    yr_now=xr*np.sin(stag) + yr*np.cos(stag)
    xr=xr_now
    yr=yr_now

    minx=min(xr)
    maxx=max(xr)
    miny=min(yr)
    maxy=max(yr)

    # rotate back
    xb1=minx*np.cos(stag)    - miny*np.sin(-stag) + xdat
    xb2=minx*np.cos(stag)    - maxy*np.sin(-stag) + xdat
    xb3=maxx*np.cos(stag)    - maxy*np.sin(-stag) + xdat
    xb4=maxx*np.cos(stag)    - miny*np.sin(-stag) + xdat
    yb1=minx*np.sin(- stag)  + miny*np.cos(stag)  + ydat
    yb2=minx*np.sin(- stag)  + maxy*np.cos(stag)  + ydat
    yb3=maxx*np.sin(- stag)  + maxy*np.cos(stag)  + ydat
    yb4=maxx*np.sin(- stag)  + miny*np.cos(stag)  + ydat
    
    x1=xLE - Lup
    y1=yb1
    x2=x1
    y2=y1 + pitch*0.5
    x3=x1
    y3=y1 - pitch*0.5
    x4=xTE + Ldn
    y4=yb4
    x5=x4
    y5=y4 + pitch*0.5
    x6=x4
    y6=y4 - pitch*0.5

    #find point on o-bound nearest to xb2,yb2
    d=(xo - xb2) ** 2 + (yo - yb2) ** 2
    imin2=np.argmin(np.sqrt(d))
    x7=xo[imin2]
    y7=yo[imin2]
    x8=x7
    y8=y7 - pitch
    #find point on o-bound nearest to xb1,yb1
    d=(xo - xb1) ** 2 + (yo - yb1) ** 2
    imin3=np.argmin(np.sqrt(d))
    x9=xo[imin3]
    y9=yo[imin3]
    #find point on o-bound nearest to xb3,yb3
    d=(xo - xb3) ** 2 + (yo - yb3) ** 2
    imin5=np.argmin(np.sqrt(d))
    x10=xo[imin5]
    y10=yo[imin5]
    x11=x10
    y11=y10 - pitch
    #find point on o-bound nearest to xb4,yb4
    d=(xo - xb4) ** 2 + (yo - yb4) ** 2
    imin6=np.argmin(np.sqrt(d))
    x12=xo[imin6]
    y12=yo[imin6]
    x13=x12
    y13=y12 + pitch
        
    # get curve lengths to set mesh points based on smallest arc length (block
    # 3 LE)
    xp,yp,dum,dum=clip_curve(xprof,yprof,imin3,imin2,200)
    # get curve length
    s=curve_length(xp,yp)
    sle=s[-1]
    
    # estimate profile surface length
    xp,yp,dum,dum=clip_curve(xo,yo,imin2,imin5,200)
    sp=curve_length(xp,yp)
    ssurf=sp[-1]
    
    sprof=curve_length(xprof,yprof)
    #ssurf=sprof[-1]*0.5
    
    # Creat initial mesh to solve for block boundaries
    
    # Block 1
    ni=np.int(npp*np.ceil(Lup / sle))
    nj=np.int(npp*np.ceil((y2 - y1) / sle))
    fi=np.linspace(0,1,ni)
    fj=np.linspace(0,1,nj)
    
    # extract part of o-grid
    xp,yp,dum,dum=clip_curve(xo,yo,imin3,imin2,nj)
    nic=2
    njc=nj
    if (yp[-1] < yp[0]):
        yp=yp[::-1]
        xp=xp[::-1]
    
    xp=matlib.repmat(xp,nic,1)
    yp=matlib.repmat(yp,nic,1)
    xp[0,:]=np.linspace(x1,x2,njc)
    yp[0,:]=np.linspace(y1,y2,njc)
    
    # create arrays for interpolation
    fic=np.linspace(0,1,nic)
    fjc=np.linspace(0,1,njc)
    
    # use 2-d interpolation to generate base grid
    fx=interpolate.interp2d(fjc,fic,xp)
    fy=interpolate.interp2d(fjc,fic,yp)
    blk[0]['x'] = fx(fj,fi)
    blk[0]['y'] = fy(fj,fi)
    
    # Block 2
    ni=np.int(npp*np.ceil(Lup / sle))
    nj=np.int(npp*np.ceil((y9 - y8) / sle))
    fi=np.linspace(0,1,ni)
    fj=np.linspace(0,1,nj)  
    nic=2
    njc=nj
    xp = np.zeros([nic,njc])
    yp = np.zeros([nic,njc])
    xp[0,:]=np.linspace(x3,x1,njc)
    yp[0,:]=np.linspace(y3,y1,njc)
    xp[-1,:]=np.linspace(x8,x9,njc)
    yp[-1,:]=np.linspace(y8,y9,njc)
    fic=np.linspace(0,1,nic)
    fjc=np.linspace(0,1,njc)
    fx=interpolate.interp2d(fjc,fic,xp)
    fy=interpolate.interp2d(fjc,fic,yp)
    blk[1]['x'] = fx(fj,fi)
    blk[1]['y'] = fy(fj,fi) 
    
    # Block 3
    ni=np.int(npp*np.ceil((y2 - y1) / sle)) 
    nj=np.int(npp*np.ceil(Lo / sle))
    fi=np.linspace(0,1,ni)
    fj=np.linspace(0,1,nj)    
    xp,yp,dum,dum=clip_curve(xo,yo,imin3,imin2,ni)
    xp2,yp2,dum,dum=clip_curve(xprof,yprof,imin3,imin2,ni)
    nic=ni
    njc=2
    if (yp[-1] < yp[0]):
        yp=yp[::-1]
        xp=xp[::-1]
        yp2=yp2[::-1]
        xp2=xp2[::-1]
    xp=np.transpose(matlib.repmat(xp,njc,1))
    yp=np.transpose(matlib.repmat(yp,njc,1))
    xp[:,-1]=xp2
    yp[:,-1]=yp2
    fic=np.linspace(0,1,nic)
    fjc=np.linspace(0,1,njc)
    fx=interpolate.interp2d(fjc,fic,xp)
    fy=interpolate.interp2d(fjc,fic,yp)
    blk[2]['x'] = fx(fj,fi)
    blk[2]['y'] = fy(fj,fi) 
    
    
    # Block 4
    ni=np.int(npp*np.ceil(ssurf / sle))
    nj=np.int(npp*np.ceil(Lo / sle))
    fi=np.linspace(0,1,ni)
    fj=np.linspace(0,1,nj)    
    xp,yp,dum,dum=clip_curve(xo,yo,imin3,imin6,ni)
    xp2,yp2,dum,dum=clip_curve(xprof,yprof,imin3,imin6,ni)
    nic=ni
    njc=2
    if (xp[-1] < xp[0]):
        yp=yp[::-1]
        xp=xp[::-1]
        yp2=yp2[::-1]
        xp2=xp2[::-1]
    xp=np.transpose(matlib.repmat(xp,njc,1))
    yp=np.transpose(matlib.repmat(yp,njc,1))
    xp[:,-1]=xp2
    yp[:,-1]=yp2
    fic=np.linspace(0,1,nic)
    fjc=np.linspace(0,1,njc)
    fx=interpolate.interp2d(fjc,fic,xp)
    fy=interpolate.interp2d(fjc,fic,yp)
    blk[3]['x'] = fx(fj,fi)
    blk[3]['y'] = fy(fj,fi) 
    

    # Block 5
    ni=np.int(npp*np.ceil(ssurf / sle))
    nj=np.int(npp*np.ceil(Lo / sle))
    fi=np.linspace(0,1,ni)
    fj=np.linspace(0,1,nj)    
    xp,yp,dum,dum=clip_curve(xo,yo,imin2,imin5,ni)
    xp2,yp2,xp3,yp3=clip_curve(xprof,yprof,imin2,imin5,ni)
    if (np.mean(yp2) < np.mean(yp3)):
        xp2=xp3
        yp2=yp3    
    nic=ni
    njc=2
    if (xp[-1] < xp[0]):
        xp=xp[::-1]
        yp=yp[::-1]
        xp2=xp2[::-1]
        yp2=yp2[::-1]
    xp=np.transpose(matlib.repmat(xp,njc,1))
    yp=np.transpose(matlib.repmat(yp,njc,1))
    xp[:,-1]=xp2
    yp[:,-1]=yp2
    fic=np.linspace(0,1,nic)
    fjc=np.linspace(0,1,njc)
    fx=interpolate.interp2d(fjc,fic,xp)
    fy=interpolate.interp2d(fjc,fic,yp)
    blk[4]['x'] = fx(fj,fi)
    blk[4]['y'] = fy(fj,fi)   

    
    # Block 6
    ni=np.int(npp*np.ceil(ssurf / sle))
    nj=np.int(npp*np.ceil((y9 - y8) / sle))
    fi=np.linspace(0,1,ni)
    fj=np.linspace(0,1,nj)    
    xp2,yp2,dum,dum=clip_curve(xo,yo,imin3,imin6,ni)
    xp,yp,dum,dum=clip_curve(xo,yo,imin2,imin5,ni)
    nic=ni
    njc=2
    if (xp[-1] < xp[0]):
        xp=xp[::-1]
        yp=yp[::-1]
        
    if (xp2[-1] < xp2[0]):
        xp2=xp2[::-1]
        yp2=yp2[::-1]

    xp=np.transpose(matlib.repmat(xp,njc,1))
    yp=np.transpose(matlib.repmat(yp-pitch,njc,1))
    xp[:,-1]=xp2
    yp[:,-1]=yp2
    fic=np.linspace(0,1,nic)
    fjc=np.linspace(0,1,njc)
    fx=interpolate.interp2d(fjc,fic,xp)
    fy=interpolate.interp2d(fjc,fic,yp)
    blk[5]['x'] = fx(fj,fi)
    blk[5]['y'] = fy(fj,fi) 
    
    # Block 7
    ni=np.int(npp*np.ceil((y5 - y4) / sle))
    nj=np.int(npp*np.ceil(Lo / sle))
    fi=np.linspace(0,1,ni)
    fj=np.linspace(0,1,nj)    
    xp,yp,dum,dum=clip_curve(xo,yo,imin5,imin6,ni)
    xp2,yp2,dum,dum=clip_curve(xprof,yprof,imin5,imin6,ni)
    nic=ni
    njc=2
    if (yp[-1] < yp[0]):
        xp=xp[::-1]
        yp=yp[::-1]
        xp2=xp2[::-1]
        yp2=yp2[::-1]
        
    xp=np.transpose(matlib.repmat(xp,njc,1))
    yp=np.transpose(matlib.repmat(yp,njc,1))        
    xp[:,-1]=xp2
    yp[:,-1]=yp2   
    fic=np.linspace(0,1,nic)
    fjc=np.linspace(0,1,njc)
    fx=interpolate.interp2d(fjc,fic,xp)
    fy=interpolate.interp2d(fjc,fic,yp)
    blk[6]['x'] = fx(fj,fi)
    blk[6]['y'] = fy(fj,fi)  
    
    # Block 8
    ni=np.int(npp*np.ceil(Ldn / sle))
    nj=np.int(npp*np.ceil((y5 - y4) / sle))
    fi=np.linspace(0,1,ni)
    fj=np.linspace(0,1,nj)    
    xp,yp,dum,dum=clip_curve(xo,yo,imin5,imin6,nj)
    nic=2
    njc=nj
    if (yp[-1] < yp[0]):
        xp=xp[::-1]
        yp=yp[::-1]
        xp2=xp2[::-1]
        yp2=yp2[::-1]

    xp=matlib.repmat(xp,nic,1)
    yp=matlib.repmat(yp,nic,1)
    
    xp[-1,:]=np.linspace(x4,x5,njc)
    yp[-1,:]=np.linspace(y4,y5,njc)
    fic=np.linspace(0,1,nic)
    fjc=np.linspace(0,1,njc)
    fx=interpolate.interp2d(fjc,fic,xp)
    fy=interpolate.interp2d(fjc,fic,yp)
    blk[7]['x'] = fx(fj,fi)
    blk[7]['y'] = fy(fj,fi) 
    
    # Block 9
    ni=np.int(npp*np.ceil(Ldn / sle))
    nj=np.int(npp*np.ceil((y9 - y8) / sle))
    fi=np.linspace(0,1,ni)
    fj=np.linspace(0,1,nj) 
    nic=2
    njc=nj
    xp=np.zeros([nic,njc])
    yp=np.zeros([nic,njc])
    xp[0,:]=np.linspace(x11,x12,njc)
    yp[0,:]=np.linspace(y11,y12,njc)
    xp[-1,:]=np.linspace(x6,x4,njc)
    yp[-1,:]=np.linspace(y6,y4,njc)
    fic=np.linspace(0,1,nic)
    fjc=np.linspace(0,1,njc)
    fx=interpolate.interp2d(fjc,fic,xp)
    fy=interpolate.interp2d(fjc,fic,yp)
    blk[8]['x'] = fx(fj,fi)
    blk[8]['y'] = fy(fj,fi) 
    
    NB=len(blk)
    nib = {}
    njb = {}
    for i in range(NB):
         nib[i],njb[i] = np.shape(blk[i]['x'])

    
    # patch interfaces
    
    # Block 1
    next_block[0]['im'] = 0
    next_block[0]['ip'] = 3
    next_block[0]['jm'] = 2
    next_block[0]['jp'] = 2
    next_patch[0]['im'] = 1
    next_patch[0]['ip'] = 3
    next_patch[0]['jm'] = 4
    next_patch[0]['jp'] = 3
    
    # Block 2   
    next_block[1]['im'] = 0
    next_block[1]['ip'] = 6
    next_block[1]['jm'] = 1
    next_block[1]['jp'] = 1
    next_patch[1]['im'] = 1            
    next_patch[1]['ip'] = 1
    next_patch[1]['jm'] = 4
    next_patch[1]['jp'] = 3
    
    next_block[2]['im'] = 4
    next_block[2]['ip'] = 5
    next_block[2]['jm'] = 1
    next_block[2]['jp'] = 0
    next_patch[2]['im'] = 1
    next_patch[2]['ip'] = 1
    next_patch[2]['jm'] = 2
    next_patch[2]['jp'] = 3
    
    next_block[3]['im'] = 3
    next_block[3]['ip'] = 7
    next_block[3]['jm'] = 6
    next_block[3]['jp'] = 0
    next_patch[3]['im'] = 1
    next_patch[3]['ip'] = 1
    next_patch[3]['jm'] = 4
    next_patch[3]['jp'] = 3
    
    next_block[4]['im'] = 3
    next_block[4]['ip'] = 7
    next_block[4]['jm'] = 6
    next_block[4]['jp'] = 0
    next_patch[4]['im'] = 2
    next_patch[4]['ip'] = 2
    next_patch[4]['jm'] = 3
    next_patch[4]['jp'] = 3
    
    next_block[5]['im'] = 2
    next_block[5]['ip'] = 9
    next_block[5]['jm'] = 5
    next_block[5]['jp'] = 4
    next_patch[5]['im'] = 2
    next_patch[5]['ip'] = 1
    next_patch[5]['jm'] = 3
    next_patch[5]['jp'] = 3
    
    next_block[6]['im'] = 4
    next_block[6]['ip'] = 5
    next_block[6]['jm'] = 8
    next_block[6]['jp'] = 0
    next_patch[6]['im'] = 2
    next_patch[6]['ip'] = 2
    next_patch[6]['jm'] = 1
    next_patch[6]['jp'] = 3
    
    next_block[7]['im'] = 7
    next_block[7]['ip'] = 0
    next_block[7]['jm'] = 9
    next_block[7]['jp'] = 9
    next_patch[7]['im'] = 3
    next_patch[7]['ip'] = 2
    next_patch[7]['jm'] = 4
    next_patch[7]['jp'] = 3
    
    next_block[8]['im'] = 6
    next_block[8]['ip'] = 0
    next_block[8]['jm'] = 8
    next_block[8]['jp'] = 8
    next_patch[8]['im'] = 2
    next_patch[8]['ip'] = 2
    next_patch[8]['jm'] = 4
    next_patch[8]['jp'] = 3
    
    # corners
    corner[0]['block'] = [1,2,3,4,6]
    corner[0]['i'] = [nib[0],nib[1],1,1,1]
    corner[0]['j'] = [1,njb[1],1,1,njb[5]]
    corner[1]['block'] = [1,2,3,5,6]
    corner[1]['i'] = [nib[0],nib[1],nib[2],1,1]
    corner[1]['j'] = [njb[0],1,1,1,1]
    corner[2]['block'] = [8,9,7,4,6]
    corner[2]['i'] = [1,1,1,nib[3],nib[5]]
    corner[2]['j'] = [1,njb[8],1,1,njb[5]]
    corner[3]['block'] = [8,9,7,5,6]
    corner[3]['i'] = [1,1,nib[6],nib[4],nib[5]]
    corner[3]['j'] = [njb[7],1,1,1,1]
    
    
    for n in range(len(corner)):
        corner[n]['Nb'] = len(corner[n]['block'])

    # call mesh smooth to solve for block boundaries
    
    blk=mesh_smooth(blk,next_block,next_patch,corner,pitch,msmooths,xprof,yprof,ywall,cor_fac)


    return blk,next_block,next_patch,corner
