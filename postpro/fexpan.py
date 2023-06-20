# Copyright (c) 2023, University of Cambridge, all rights reserved. Written by Andrew Wheeler, University of Cambridge

def fexpan(L_over_a,N):

    # solve (L/a) - r(L/a) + r^N - 1 = 0
    
    if (abs(L_over_a - N) < 1e-12):
        r=1.0
        return r
    else:
        if (L_over_a > N):
            r=1.00001
        else:
            r=0.99999
    
    for iter in range(50):
        if (r < 1.0):
            r=1.0 - (1.0 - r ** N) / L_over_a
        else:
            if (r > 1.0):
                r=(1.0 - (L_over_a*(1.0 - r))) ** (1 / N)
    
    if(r < 0.1 or r > 1.9):
       r = 1.0
    
    return r
