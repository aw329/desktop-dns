# Generated with SMOP  0.41
from libsmop import *
# blade_profile.m

    
@function
def blade_profile(chi=None,tLE=None,N=None,*args,**kwargs):
    varargin = blade_profile.varargin
    nargin = blade_profile.nargin

    # create circular arc blade with 'shape-space' thickness distribution
    
    chi_1=dot(chi(1),pi) / 180
# blade_profile.m:4
    chi_2=dot(chi(2),pi) / 180
# blade_profile.m:5
    cam=abs(chi_1 - chi_2)
# blade_profile.m:7
    stag=dot((chi_1 + chi_2),0.5)
# blade_profile.m:8
    c=linspace(0,1,N)
# blade_profile.m:10
    rad=0.5 / sin(dot(cam,0.5))
# blade_profile.m:11
    thet_ex=pi / 2 + chi_1
# blade_profile.m:13
    thet_in=pi / 2 + chi_2
# blade_profile.m:14
    thet=linspace(thet_ex,thet_in,N)
# blade_profile.m:16
    r=linspace(rad,rad,N)
# blade_profile.m:17
    # add thickness
    psi=linspace(0,1,N)
# blade_profile.m:20
    # shape space
    S=1
# blade_profile.m:23
    phi[1]=tLE
# blade_profile.m:24
    phi=multiply(multiply(sqrt(psi),(1 - psi)),S) + dot(psi,phi(1))
# blade_profile.m:25
    t=dot(dot(5,tLE),phi)
# blade_profile.m:27
    tTE=t(N)
# blade_profile.m:28
    # add TE
    rthet=multiply(r,thet)
# blade_profile.m:31
    for nn in arange(1,3).reshape(-1):
        iTE=rthet - rthet(N) < tTE
# blade_profile.m:34
        tTE=max(t(iTE))
# blade_profile.m:35
    
    a=tTE - abs(rthet(N) - rthet(iTE))
# blade_profile.m:38
    t[iTE]=sqrt(dot(tTE,tTE) - multiply(a,a))
# blade_profile.m:39
    rup=r + dot(t,0.5)
# blade_profile.m:41
    rdn=r - dot(t,0.5)
# blade_profile.m:42
    xc,yc=pol2cart(thet,r,nargout=2)
# blade_profile.m:44
    xu,yu=pol2cart(thet,rup,nargout=2)
# blade_profile.m:45
    xd,yd=pol2cart(thet,rdn,nargout=2)
# blade_profile.m:46
    xprof=concat([xu(arange(1,N - 1)),xd(arange(N,1,- 1))])
# blade_profile.m:48
    yprof=concat([yu(arange(1,N - 1)),yd(arange(N,1,- 1))])
# blade_profile.m:49
    # remove repeat points and smooth
    N=length(xprof)
# blade_profile.m:52
    for mm in arange(1,1).reshape(-1):
        ii=arange(2,N - 1)
# blade_profile.m:54
        xprof[ii]=dot((xprof(ii - 1) + xprof(ii + 1)),0.5)
# blade_profile.m:55
        yprof[ii]=dot((yprof(ii - 1) + yprof(ii + 1)),0.5)
# blade_profile.m:56
        # xprof(1) = (xprof(N-1) + xprof(2))*0.5;
# yprof(1) = (yprof(N-1) + yprof(2))*0.5;
# xprof(N) = xprof(1);
# yprof(N) = yprof(1);
    
    return xprof,yprof