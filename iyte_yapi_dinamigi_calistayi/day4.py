import numpy as np
import math
import matplotlib.pyplot as plt
from scipy.interpolate import make_interp_spline

ags = np.loadtxt('xg.txt')  # the units are in g
nag = len(ags)

start_t = 0
dt = 0.005 #sec
end_t = (nag-1) * dt
t = np.arange(0 , (nag*dt) , dt)

pi = math.pi

m1 = 235.53507
m2 = 231.86409

M = np.diag([m1,m1,m1,m1,m2])
ndof = M.shape[0]
k = 230144.240741
K = np.array([
    [2*k  ,-k  ,0   ,0  ,  0],
    [-k   ,2*k ,-k  ,0  ,  0],
    [0    ,-k  ,2*k ,-k  ,  0],
    [0    ,0   ,-k  ,2*k,-k ],
    [0    ,0   ,0   ,-k ,k  ]
])
invMK = np.dot(np.linalg.inv(M) , K)

V,D = np.linalg.eig(invMK) #Eigenvectors and eigenvalues

idx = V.argsort()[::1]
V = V[idx]
D = D[:,idx]

w = [np.sqrt(item) for item in V]
T = [2*np.pi/item for item in w]

DT = np.transpose(D)
Mb = DT.dot(M).dot(D)
Kb = DT.dot(K).dot(D)

Mb_diag = Mb.diagonal()
wn = np.sqrt(V)
eps = 0.05

Cb = np.diag(2*eps*np.multiply(wn,Mb_diag))

invDt = np.linalg.inv(DT)
invD = np.linalg.inv(D)

C = invDt.dot(Cb).dot(invD)

Cb_test = DT.dot(C).dot(D)

Mb[Mb < 1e-10] = 0
Kb[Kb < 1e-10] = 0
Cb[Cb < 1e-10] = 0

r = np.ones(ndof)

mMr = -M.dot(r)

p_exc = np.outer(mMr, ags) * 9.806
p_exc0  = p_exc[:,0]

invM = np.linalg.inv(M)

x0 = np.zeros(ndof)
v0 = np.zeros(ndof)
a0 = invM.dot(p_exc0-C.dot(v0)- K.dot(x0))

beta = 1/4
gamma = 1/2

K_tilde = K + gamma/(beta*dt)*C + 1/(beta*dt**2)*M
C1 = 1/(beta*dt)*M + gamma/beta*C
C2 = 1/(2*beta)*M + dt*(gamma/(2*beta)-1)*C

x = np.zeros((ndof,nag))
v = np.zeros((ndof,nag))
a = np.zeros((ndof,nag))

x[:,0]=x0
v[:,0]=v0
a[:,0]=a0
invK_tilde = np.linalg.inv(K_tilde)

for i in range(1, len(t)):
    dP = p_exc[:,i] - p_exc[:,i - 1]
    dP_tilde = dP + (C1.dot(v[:,i - 1]) + (C2.dot(a[:,i - 1])))

    dx = dP_tilde.dot(invK_tilde)
    dv = gamma / (beta * dt) * dx - gamma / beta * v[:,i - 1] + (1 - gamma / (2 * beta)) * dt * a[:,i - 1]
    da = 1 / (beta * dt ** 2) * dx - 1 / (beta * dt) * v[:,i - 1] - 1 / (2 * beta) * a[:,i - 1]
    x[:, i] = x[:, i - 1] + dx
    v[:,i] = v[:,i-1] + dv
    a[:,i] = a[:,i-1] + da

plt.figure(4)
plt.plot(t,a[1])
plt.title("a vs t Graph")
plt.xlabel("t (sec)")
plt.ylabel("a (m/s^2")
plt.grid()


plt.figure(0)
plt.plot(t,x[1], label='Pyhton')

#Add control data to the plot
# x5 = np.loadtxt('x5.txt')  # the units are in g
# nx5 = len(x5)

# stx5 = 0
# dtx5 = 0.01 #sec
# etx5 = (nx5-1)*dt
# tx5=np.arange(0,(nx5*dtx5),dtx5)
# plt.plot(tx5,x5,'--', linewidth=0.5, label='SAP2000')

# plt.xlim([t.min(), t.max()])
# plt.title("x_5")
# plt.xlabel("t (sn)")
# plt.ylabel("x_5 (m)")
# plt.legend()
# plt.grid()

# plt.savefig('x5.pdf', bbox_inches='tight')


#Base Shear
fbase = -k*x[0]
plt.figure(2)
plt.plot(t,fbase, label='Pyhton')

fb = np.loadtxt('fb.txt')  # the units are in g
nfb = len(fb)

stfb = 0
dtfb = 0.01#sec
etfb = (nfb-1)*dt
tfb=np.arange(0,(nfb*dtfb),dtfb)

# plt.plot(tfb,fb,'--', linewidth=0.5, label='SAP2000')
plt.xlim([t.min(), t.max()])
plt.title("Base Shear")
plt.xlabel("t (sn)")
plt.ylabel("f_b (kN)")
plt.legend()
plt.grid()

plt.savefig('fb.pdf', bbox_inches='tight')


#Plot Normalized Mode Shapes
norm_j = ndof-1
for i in range(0,ndof):
    D[:,i] = D[:,i]/D[norm_j,i]

y = np.arange(0,ndof+1)
mcolor = ['r', 'g', 'b', 'k', 'm']

plt.figure(1)
for i in range(0,ndof):
    Dbase = np.append(0, D[:,i])
    ynew = np.linspace(y.min(), y.max(), 300)
    spl = make_interp_spline(y, Dbase, k=3)
    xnew = spl(ynew)
    plt.plot(xnew,ynew,mcolor[i])
plt.show()
plt.savefig('modes.pdf', bbox_inches='tight')
