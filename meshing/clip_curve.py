import numpy as np

def clip_curve(x,y,i1,i2,n):
    # clip out shortest curve between i1 and i2, assuming x,y form a continuous loop
    x = np.transpose(x)
    y = np.transpose(y)
    N=len(x)
    s = np.zeros(N)
    for i in range(1,N):
        dx=x[i] - x[i - 1]
        dy=y[i] - y[i - 1]
        s[i]=s[i - 1] + np.sqrt(dx*dx + dy*dy)
 
    s=s / s[-1]
    stot=1
    s1=s[i1]
    s2=s[i2]
    del1=abs(s1 - s2)
    del2=stot - del1
    if (del1 < del2):
        flip=1
        if (i2 < i1):
            flip=- 1
        i = np.array(range(i1,i2+flip,flip))
        xc=x[i]
        yc=y[i]
        sc=s[i]
    else:
        if (i1 < i2):
            i = np.array(range(i1,0,-1))
            ii = np.array(range(N-1,i2-1,-1))
            inow = np.concatenate((i,ii))
            inow = np.ndarray.astype(inow,int)
            xc=x[inow]
            yc=y[inow]
            sc=s[inow]
            nnow = len(sc)
            i = np.array(range(nnow-(N-i2),nnow))
            sc[i]=sc[i] - 1.0
        else:
            i = np.array(range(i2,0,-1))
            ii = np.array(range(N-1,i1-1,-1))
            inow = np.concatenate((i,ii))
            inow = np.ndarray.astype(inow,int)
            xc=x[inow]
            yc=y[inow]
            sc=s[inow]
            nnow = len(sc)
            i = np.array(range(nnow-(N-i1),nnow))    
            sc[i]=sc[i] - 1.0
    si=np.linspace(sc[0],sc[-1],n)
    si2=1.0 - si
    snow = np.concatenate((s-1,s[1:-1],s+1))
    xnow = np.concatenate((x,x[1:-1],x))
    ynow = np.concatenate((y,y[1:-1],y))

    xi = np.interp(si,snow,xnow)
    yi = np.interp(si,snow,ynow)

    xi2 = np.interp(si2,snow,xnow)
    yi2 = np.interp(si2,snow,ynow)
    

    return xi,yi,xi2,yi2