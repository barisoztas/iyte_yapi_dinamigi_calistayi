import math
import numpy as np
import matplotlib.pyplot as plt
from matplotlib import rc as rc

pi=math.pi;

dt=0.001;
start_t=0;
end_t=10;
t=np.arange(start_t,end_t+dt,dt)
nag=t.size
g=9.806
k=10000
m=1
K=np.array([[3*k,-k],[-k,k]])
M=np.array([[m,0],[0,m]])




fi_1=np.array([[2.414],[1]])
fi_2=np.array([[1],[-0.414]])

M1=(fi_1.transpose().dot(M)).dot(fi_1)
M2=(fi_2.transpose().dot(M)).dot(fi_2)
K1=(fi_1.transpose().dot(K)).dot(fi_1)
K2=(fi_2.transpose().dot(K)).dot(fi_2)

w1=math.sqrt(K1/(M1))
w2=math.sqrt(K2/(M2))
one=np.array([[1,1]])

Kfi_1=K.dot(fi_1)
Kfi_2=K.dot(fi_2)

onekfi_1=one.dot(Kfi_1)
onekfi_2=one.dot(Kfi_2)
q1=np.zeros(len(t))
q2=np.zeros(len(t))

T1=2*pi/w1
T2=2*pi/w2

sa1=0.6664
sa2=0.604

alfa1=(fi_1.transpose().dot(M)).dot(one.transpose())/M1
alfa2=(fi_2.transpose().dot(M)).dot(one.transpose())/M2

q1maks=alfa1*sa1
q2maks=alfa2*sa2

V1=onekfi_1*q1maks
V2=onekfi_2*q2maks
a = 1
