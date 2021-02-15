import numpy as np
import matplotlib.pyplot as plt
import math

start_time = 0
end_time = 30
dt = 0.1
nag = (end_time-start_time)/dt +1
t = np.arange(start_time,end_time+dt,dt)
q1 = np.zeros(len(t))
q2 = np.zeros(len(t))
for i in range(len(t)):
    q1[i] = (10/4) * (1 - math.cos(0.76*t[i]))
    q2[i] = (-4.14/2.83) * (1 - math.cos(1.55*t[i]))

lw = 0.5
fig1, ax1 = plt.subplots()
ax1.plot(t,q1, linewidth =lw)
ax1.set_title("q1 vs time")
ax1.set_xlabel("time (sec)")
ax1.set_ylabel("q_1(t)")
ax1.grid(True)
plt.show()

fig2, ax2 = plt.subplots()
ax2.plot(t,q2, linewidth =lw)
ax2.set_title("q2 vs time")
ax2.set_xlabel("time (sec)")
ax2.set_ylabel("q_2(t)")
ax2.grid(True)
plt.show()
