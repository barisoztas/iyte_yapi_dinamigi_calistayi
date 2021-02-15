import math

import numpy as np

import matplotlib.pyplot as plt


def sdof_harmonicsine(m, c, k, dt, t, Nt, w, p0, x0, xd0):
    pi = math.pi;
    wn = math.sqrt(k / m)  # rad/sec
    f = wn / (2 * pi);  # Hz
    T = 1 / f;  # sec
    ksi = c / (2 * wn * m)  # damping ratio
    wD = wn * math.sqrt(1 - ksi ** 2)

    nag = len(t)

    x_trans = np.zeros((nag))
    x_steady = np.zeros((nag))
    x_d = np.zeros((nag))
    x_dd = np.zeros((nag))
    F_tr = np.zeros((nag))

    bn = (w / wn)
    xst = p0 / k
    D = 1 / math.sqrt((1 - bn ** 2) ** 2 + (2 * ksi * bn) ** 2)
    ro = xst * D
    phi = math.atan((2 * ksi * bn) / (1 - bn ** 2))

    B = x0 - ro * math.sin(-phi)
    A = (xd0 + ksi * wn * B - ro * w * math.cos(-phi)) / wD

    jj = np.arange(1, nag)

    for j in jj:
        x_trans[j] = math.exp(-ksi * wn * t[j]) * (A * math.sin(wD * t[j]) + B * math.cos(wD * t[j]))
        x_steady[j] = ro * math.sin(w * t[j] - phi)
        x_d[j] = math.exp(-ksi * wn * t[j]) * (-ksi * wn) * (
                    A * math.sin(wD * t[j]) + B * math.cos(wD * t[j])) + math.exp(-ksi * wn * t[j]) * (wD) * (
                             A * math.cos(wD * t[j]) - B * math.sin(wD * t[j])) + ro * w ** math.cos(w * t[j] - phi)
        x_dd[j] = math.exp(-ksi * wn * t[j]) * (ksi ** 2 * wn ** 2 - wD ** 2) * (
                    A * math.sin(wD * t[j]) + B * math.cos(wD * t[j])) + math.exp(-ksi * wn * t[j]) * (
                              -2 * ksi * wn) * (
                              A * math.cos(wD * t[j]) - B * math.sin(wD * t[j])) - ro * w ** 2 * math.sin(
            w * t[j] - phi)
        F_tr[j] = ((x_trans[j] + x_steady[j]) * k) + (x_d[j] * c)
    return x_trans, x_steady, F_tr, x_dd


pi = np.pi

m = 1

# wn=2*pi # rad/sec

k = 250

ksi = 0.05

wn = math.sqrt(k / m)

c = 2 * ksi * wn * m

dt = 0.01

startt = 0
endt = 5
Nt = int((endt - startt) / dt + 1)

t = np.linspace(startt, endt, Nt)

w = 2 * pi;

p0 = m * 0.05 * w ** 2;

p_exc = p0 * np.sin(w * t)

plt.plot(t, p_exc);

plt.title("Dış Kuvvet")
plt.xlabel("zaman (sn)")
plt.ylabel("genlik (kN)")

x0 = 0
xd0 = 0

# ii=np.arange(1,Nt)

x_trans, x_steady, F_tr, x_dd = sdof_harmonicsine(m, c, k, dt, t, Nt, w, p0, x0, xd0)

plt.figure()
plt.plot(t, (x_dd + p_exc / (m * w * 2)) / 9.81)

plt.title("Toplam İvme")
plt.xlabel("zaman (sn)")
plt.ylabel("genlik (g)")

plt.figure()
plt.plot(t, F_tr)

plt.title("Zemine İletilen Kuvvet")
plt.xlabel("zaman (sn)")
plt.ylabel("İletilen Kuvvet (kN)")
plt.show()

plt.plot(t,x_trans)
plt.plot(t,x_steady)
plt.show()
