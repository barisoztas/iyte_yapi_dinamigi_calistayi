import math
import numpy as np
import matplotlib.pyplot as plot
from matplotlib import rc as rc
pi=math.pi;
dt=0.001;
start_t=0;
end_t=5;
t=np.arange(start_t,end_t+dt,dt)
nag=t.size
m=1
w=2*pi
eps=0.05
k=250
wn=math.sqrt(k/m)
wD=wn*math.sqrt(1-eps**2)
p0=m*0.05*w**2
p_exc=p0*np.sin(w*t)
plot.plot(t,p_exc)

def sdof_harmonicsine(m,c,k,dt,w,p0,x0,xd0):


    eps=c/(2*wn*m)
    
    x_trans=np.zeros(len(t))
    x_steady=np.zeros(len(t))
    
    bn=w/wn
    xst=p0/k
    D=1/math.sqrt((1-bn**2)**2+(2*eps*bn)**2)
    ro=xst*D
    phi=math.atan((2*eps*bn)/(1-bn**2))
    B=x0-ro*math.sin(-phi)
    A=(xd0+eps*wn*B-ro*w*math.cos(-phi))
    for i in range(1,len(t)):
        
        x_trans[i]=math.exp(-eps*wn*t[i])*(A*math.sin(wD*t[i])+B*math.cos(wD*t[i]))
        x_steady[i]=ro*math.sin(w*t[i]-phi)
    return x_trans,x_steady

    plot.figure()
    plot(t,x_trans)
    
    plot.figure()
    plot(t,x_steady)
    