import numpy as np

import math
import matplotlib.pyplot as plt
from matplotlib import rc as rc

pi = math.pi

dt = 0.001
start_t = 0.
end_t = 10.

t = np.arange(start_t,end_t+dt,dt)
nag = t.size

A_sin = 2.
w_sin = 0.5*pi
p_exc = A_sin * np.sin(w_sin*t)

# yapı özellikleri
wn = 12.25 #rad/sec
f = wn/(2*pi) # Hz
T = 1/f #sec

m = 1 #
k = m * wn**2

eps = 0.1
c = 2*eps*wn*m
wD = wn*math.sqrt(1-eps**2)

x0 = 0
v0 = 0
a0 = (p_exc[0] - k*x0 - c*v0)/m
beta = 1/4
gama = 1/2

k_tilde = k+gama/(beta*dt)*c + 1/(beta*dt**2)*m
c1 = 1/(beta*dt)*m + gama/beta*c
c2 = 1/(2*beta)*m + dt*(gama/(2*beta)-1)*c

x = np.zeros(nag)
v = np.zeros(nag)
a = np.zeros(nag)

x[0] = x0
v[0] = v0
a[0] = a0
dP_tilde = np.zeros(nag)
dx = np.zeros(nag)
dv = np.zeros(nag)
da = np.zeros(nag)
for i in range(1,nag):
    dP = p_exc[i] - p_exc[i-1]
    dP_tilde = dP + (c1*v[i-1]) + (c2*a[i-1])
    dx = dP_tilde / k_tilde
    dv = gama / (beta * dt) * dx - gama / beta * v[i - 1] + (1 - gama / (2 * beta)) * dt * a[i - 1]
    da = 1/(beta*dt**2) * dx - 1/(beta*dt)*v[i-1] - 1/(2*beta)*a[i-1]
    x[i] = x[i-1] + dx
    v[i] = v[i - 1] + dv
    a[i] = a[i - 1] + da

lw = 0.5
fig1, ax1 = plt.subplots()
ax1.plot(t,x, linewidth =lw)
ax1.set_title("Displacement")
ax1.set_xlabel('t (sec)')
ax1.set_ylabel('x(t)')
ax1.grid(True)
ax1.set_xlim([0,end_t])
plt.show()


fig0, ax0 = plt.subplots()
ax0.plot(t,p_exc, linewidth =lw)
ax0.set_title("External Force")
ax0.set_xlabel('t (sec)')
ax0.set_ylabel('p_exc(t)')
ax0.grid(True)
ax0.set_xlim([0,end_t])
plt.show()

fig3, ax3 = plt.subplots()
ax3.plot(t,a, linewidth =lw)
ax3.set_title("Acceleration")
ax3.set_xlabel('t (sec)')
ax3.set_ylabel('a(t)')
ax3.grid(True)
ax3.set_xlim([0,end_t])
plt.show()

fig4, ax4 = plt.subplots()
ax4.plot(t,v, linewidth =lw)
ax4.set_title("Velocity")
ax4.set_xlabel('t (sec)')
ax4.set_ylabel('v(t)')
ax4.grid(True)
ax4.set_xlim([0,end_t])
plt.show()

