import numpy as np
import math
import matplotlib.pyplot as plt
from scipy.interpolate import make_interp_spline
import pandas as pd

read_file = pd.read_csv(r'180_FN.acc',delimiter=r"\s+", header=3)

a = np.zeros(len(read_file["a"]))
a = np.array(read_file["a"])


def newmark_method(m,dt,a,beta,gamma,x0,v0):
    p_exc = - m* a
    pi = math.pi
    start_t = 0.
    end_t = 10.

    t = np.arange(start_t, end_t + dt, dt)
    nag = t.size




    # yapı özellikleri
    wn = 12.25  # rad/sec
    f = wn / (2 * pi)  # Hz
    T = 1 / f  # sec


    k = m * wn ** 2

    eps = 0.1
    c = 2 * eps * wn * m
    wD = wn * math.sqrt(1 - eps ** 2)

    a0 = (p_exc[0] - k * x0 - c * v0) / m


    k_tilde = k + gamma / (beta * dt) * c + 1 / (beta * dt ** 2) * m
    c1 = 1 / (beta * dt) * m + gamma / beta * c
    c2 = 1 / (2 * beta) * m + dt * (gamma / (2 * beta) - 1) * c

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
    for i in range(1, nag):
        dP = p_exc[i] - p_exc[i - 1]
        dP_tilde = dP + (c1 * v[i - 1]) + (c2 * a[i - 1])
        dx = dP_tilde / k_tilde
        dv = gamma / (beta * dt) * dx - gamma / beta * v[i - 1] + (1 - gamma / (2 * beta)) * dt * a[i - 1]
        da = 1 / (beta * dt ** 2) * dx - 1 / (beta * dt) * v[i - 1] - 1 / (2 * beta) * a[i - 1]
        x[i] = x[i - 1] + dx
        v[i] = v[i - 1] + dv
        a[i] = a[i - 1] + da

    return (a,v,x,t)
def plot_a_v_x(a,v,x,t):
    plt.figure(0)
    plt.plot(t,a)
    plt.title("a vs t Graph")
    plt.xlabel("t (sec)")
    plt.ylabel("a (m/s^2")
    plt.grid()


    plt.figure(1)
    plt.plot(t,v)
    plt.title("v vs t Graph")
    plt.xlabel("t (sec)")
    plt.ylabel("v (m/s")
    plt.grid()

    plt.figure(2)
    plt.plot(t,x)
    plt.title("x vs t Graph")
    plt.xlabel("t (sec)")
    plt.ylabel("x (m)")
    plt.grid()
    plt.show()
    return 1



(a,v,x,t) = newmark_method(100,0.01,a,1/4,1/2,0,0)
plot_a_v_x(a,v,x,t)









