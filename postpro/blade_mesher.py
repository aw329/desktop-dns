# Copyright (c) 2023, University of Cambridge, all rights reserved. Written by Andrew Wheeler, University of Cambridge

import matplotlib.pyplot as plt
from .read_profile   import *
from .curve_length   import *
from .blade_topology import *
from .mesh_refinement import *
from .mesh_smooth import *

def blade_mesher(mesh):

    npp = mesh['npp']
    Lup = mesh['Lup']
    Ldn = mesh['Ldn']
    Lo  = mesh['Lo']
    ywall = mesh['ywall']
    msmooths = mesh['msmooths']
    scale_factor = mesh['scale_factor']
    refine_fac = mesh['refine_fac']
    flip = mesh['flip']
    cor_fac = mesh['cor_fac']
    
    # read in profile and normalize so that Cax=1.0
    xprof,yprof,pitch,stag=read_profile(mesh['profile'],True)
    
    # get surface distance
    sprof=curve_length(xprof,yprof)
    
    # create 9-block topology and initial coarse mesh
    blk,next_block,next_patch,corner=blade_topology(xprof,yprof,pitch,Lup,Ldn,Lo,stag,npp[0],ywall[0],msmooths[0],cor_fac)
    
    # create final refined mesh
    blk=mesh_refinement(blk,refine_fac,npp[1])
    blk=mesh_smooth(blk,next_block,next_patch,corner,pitch,msmooths[1],xprof,yprof,ywall[1],cor_fac)
 
    # apply a scaling factor
    NB = len(blk)
    for ib in range(NB):
        blk[ib]['x'] = blk[ib]['x']*scale_factor
        blk[ib]['y'] = blk[ib]['y']*scale_factor*flip
        
    pitch = pitch*scale_factor
    xprof = xprof*scale_factor
    yprof = yprof*scale_factor*flip
 
    
    # store 3DNS case files
    case = {}
    case['casename'] = mesh['case']
    case['blk'] = blk
    case['next_block'] = next_block
    case['next_patch'] = next_patch
    case['corner'] = corner
#
    return case
    

if __name__ == '__main__':

    mesh = {}
    bcs  = {}
    gas  = {}
    
    # mesh inputs 
    mesh['npp'] = [0,0]
    mesh['ywall'] = [0.,0.]
    mesh['msmooths'] = [0,0]
     
    # initial mesh inputs
    mesh['npp'][0]=4
    mesh['Lup']=0.75
    mesh['Ldn']=1.0
    mesh['Lo']=0.1
    mesh['ywall'][0]=0.01
    mesh['msmooths'][0]=500
    
    # final mesh inputs
    mesh['npp'][1]=28
    mesh['refine_fac']=4.0
    mesh['ywall'][1]=0.0005
    mesh['msmooths'][1]=10
        
    mesh['profile'] = 'geom/profile.txt'
    mesh['case'] = 'case' 
    main(mesh)    
