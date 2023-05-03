import numpy as np
import matplotlib.pyplot as plt


def plot_flo(prop,geom,plot_var,caxis):

    
    plt.figure(1)
    plt.axis('equal')
        
    for ib in range(len(geom)):
        x=geom[ib]['x']
        y=geom[ib]['y']
        f = prop[ib][plot_var]
        if(len(np.shape(f))==2):        
           plt.pcolormesh(x,y,f,shading='gouraud')
        else:
           plt.pcolormesh(x,y,f[:,:,0],shading='gouraud')

        plt.clim(caxis)
    plt.colorbar()  
    plt.title(plot_var)        

    return plt




