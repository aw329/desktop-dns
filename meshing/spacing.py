# Copyright (c) 2023, University of Cambridge, all rights reserved. Written by Andrew Wheeler, University of Cambridge

import numpy as np

def spacing(n,fx_expan,flag):
    fx = np.zeros(n)
    if (flag == 1):
        fx[0]=0.0
        df=1.0
        for i in range(1,np.floor(n / 2)):
            fx[i]=fx[i-1] + df
            df=df*fx_expan
        for i in range(np.floor(n / 2),n):
            fx[i]=fx[i-1] + df
            df=df / fx_expan
    else:
        fx[0]=0.
        df=1.0
        for i in range(1,n):
            fx[i]=fx[i-1] + df
            df=df*fx_expan
    fx=fx / fx[-1]
    fx[0] =0.
    fx[-1]=1.
    return fx
