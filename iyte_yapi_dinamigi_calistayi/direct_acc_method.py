import numpy as np
import matplotlib.pyplot as plt
import scipy
import pandas as pd

# doğrusal ivme yöntemiyle hareket denklemi çözümü

m = 4 #lb-sec^2/ft
k= 4000 #lb/ft
F0 = 200 #lb
t0 = 0.2 #sec

# yapı durum
u0 = 0
ud0 = 0
h = 0.01 #sec, time step

# a when t = 0

udd0 = (F0 - k*u0) / m


t = np.linspace(0,0.2,21) #zaman vektorü
#t = np.array([0,0.01,0.02,0.03,0.04,0.05])
#Kuvvet vectoru

print(len(t))

def force(t):
    f = np.zeros(len(t))
    for i in range(len(t)):
        if t[i] <= t0/2:
            f[i] = (400*t[i] / (t0))
        else:
            f[i] = ( (200) -  (400 * (t[i] - t0/2) / t0))
    return(f)

F = force(t)

pd_table = {'Time':t,'Force':F}
pd_table = pd.DataFrame(pd_table)
print((pd_table))

plt.plot(t,F)
plt.title('Force-Time Graph')
plt.xlabel('time(sec)')
plt.ylabel('Force(lb)')

plt.show()


