import numpy as np
import math

K = np.array([
    [300,-100],
    [-100,100]])
M = np.diag([1000,1000])
phi1 = np.array([2.414, 1])
phi2 = np.array([1,-0.414])
M_modal = np.array([
    [6.874],
    [1.171]])
K_modal = np.array([
    [399.9396],
    [399.9396]])

v1 = np.ones(2).dot(K).dot(phi1)
v2 = np.ones(2).dot(K).dot(phi2)
print("v1 : ",v1,"v2: ", v2)

start_t = 0.
end_t = 10.
dt = 0.05
t = np.arange(start_t,end_t+dt,dt)
q1 = np.zeros(len(t))
q2= np.zeros(len(t))


for i in range(len(t)):
    q1[i] = np.array([2.5 * (1 - math.cos(0.59*t[i]))])
    q2[i] = np.array([-4.14/2.83 * (1 - math.cos((2.83/1.171)*t[i]))])

q1max = q1.max()
q2max = q1.max()

F1 = v1 * q1max
F2 = v2 * q2max

print( F1,F2, 'Total Force= ', F1+F2)


