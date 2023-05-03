import numpy as np

def curve_length(x,y):
    N=len(x)
    #N = np.size(x)
    s = np.zeros(N) 
    s[0]=0.0
    for i in range(1,N):
        dx=x[i] - x[i-1]
        dy=y[i] - y[i-1]
        ds=np.sqrt(dx*dx + dy*dy)
        s[i]=s[i-1] + ds
    return s