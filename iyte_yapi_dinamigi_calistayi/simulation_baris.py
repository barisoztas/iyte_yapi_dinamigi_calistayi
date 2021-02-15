import numpy as np
import math
import matplotlib.pyplot as plt
from scipy.interpolate import make_interp_spline
import pandas as pd

read_data = pd.read_csv(r"./Bolu_1999_Lab/YDC_Lab_Bolu_1999_05kat_filtreli.dat", delimiter=r"\s+",header= None, names=['Time', '3. Kat', '2. Kat', '1. Kat', 'Zemin'])


t = np.array(read_data['Time'])

dt = 0.001
a_zemin = np.array(read_data['Zemin'])
a_1 = np.array(read_data['1. Kat'])
a_2 = np.array(read_data['2. Kat'])
a_3 = np.array(read_data['3. Kat'])
a = np.array([a_1,a_2,a_3])

m1 = 3.6390
m2 = 3.5650
m3 = 3.4850

M = np.diag([m1, m2, m3])

ndof = M.shape[0]
k1=408.86 #N/m
k2=649.73 #N/m
k3=744.04 #N/m
K = np.array([
    [k1+k2,-k2,0],
    [-k2,k2+k3,-k3],
    [0,-k3,k3]
    ])

C = np.diag([0.028,0.012,0.007])



print(read_data)

def newmark_ckb (M,C,K,a,dt):

    r = np.ones(ndof)
    mMr = -M.dot(r)

    p_exc = np.outer(mMr, a) * 9.806
    p_exc0 = p_exc[:, 0]
    pi = math.pi


    invMK = np.dot(np.linalg.inv(M), K)

    V, D = np.linalg.eig(invMK)  # Eigenvectors and eigenvalues

    idx = V.argsort()[::1]
    V = V[idx]
    D = D[:, idx]

    w = [np.sqrt(item) for item in V]
    T = [2 * np.pi / item for item in w]

    DT = np.transpose(D)
    Mb = DT.dot(M).dot(D)
    Kb = DT.dot(K).dot(D)

    Mb_diag = Mb.diagonal()
    wn = np.sqrt(V)
    eps = 0.05

    Cb = np.diag(2 * eps * np.multiply(wn, Mb_diag))


    Cb_test = DT.dot(C).dot(D)

    Mb[Mb < 1e-10] = 0
    Kb[Kb < 1e-10] = 0
    Cb[Cb < 1e-10] = 0

    r = np.ones(ndof)

    mMr = -M.dot(r)



    nag = len(t)
    invM = np.linalg.inv(M)
    a0 = np.array([0,0,0])

    x0 = 0
    v0 = 0
    a0 = invM.dot(p_exc0 - C.dot(v0) - K.dot(x0))

    beta = 1 / 4
    gamma = 1 / 2

    K_tilde = K + gamma / (beta * dt) * C + 1 / (beta * dt ** 2) * M
    C1 = 1 / (beta * dt) * M + gamma / beta * C
    C2 = 1 / (2 * beta) * M + dt * (gamma / (2 * beta) - 1) * C

    x = np.zeros((ndof, nag))
    v = np.zeros((ndof, nag))


    x[:, 0] = x0
    v[:, 0] = v0

    invK_tilde = np.linalg.inv(K_tilde)

    for i in range(1, len(t)):
        dP = p_exc[:, i] - p_exc[:, i - 1]
        dP_tilde = dP + (C1.dot(v[:, i - 1]) + (C2.dot(a[:, i - 1])))

        dx = dP_tilde.dot(invK_tilde)
        dv = gamma / (beta * dt) * dx - gamma / beta * v[:, i - 1] + (1 - gamma / (2 * beta)) * dt * a[:, i - 1]
        da = 1 / (beta * dt ** 2) * dx - 1 / (beta * dt) * v[:, i - 1] - 1 / (2 * beta) * a[:, i - 1]
        x[:, i] = x[:, i - 1] + dx
        v[:, i] = v[:, i - 1] + dv
        a[:, i] = a[:, i - 1] + da
    return (x,v,a)

(x,v,a_newmark) = newmark_ckb (M,C,K,a[0],dt)

plt.figure(0)
plt.plot(t,a[0,:])
plt.title('1. Kat ivmesi')
plt.grid()


plt.figure(1)
plt.plot(t,a[1,:])
plt.title('1. Kat ivmesi')
plt.grid()

plt.figure(2)
plt.plot(t,a[2,:])
plt.title('3. Kat ivmesi')
plt.grid()

plt.figure(3)
plt.plot(t,x[0,:])
plt.title('1. Kat Yer Değiştirme')
plt.grid()

plt.figure(4)
plt.plot(t,x[1,:])
plt.title('2. Kat Yer Değiştirme')
plt.grid()

plt.figure(5)
plt.plot(t,x[2,:])
plt.title('3. Kat Yer Değiştirme')
plt.grid()

plt.figure(6)
plt.plot(t,a)
plt.title('3. Kat Yer Değiştirme')
plt.grid()



plt.show()



