# -*- coding: utf-8 -*-
"""
Created on Sat Feb 13 15:58:09 2021

@author: ufuky
"""

import numpy as np
import math
import matplotlib.pyplot as plt
from scipy.interpolate import make_interp_spline
pi = math.pi

# TSDS Özellikleri
Tn=1.0 #[s] - TSDS periyodu (AKS'siz)
wn=2*pi/Tn #[rad/s] - TSDS açısal hızı
M=30e3 #[kg] - TSDS kütlesi
K=M*(wn**2) #[N/m] - TSDS yay rijitligi

# Dış Yük Özellikleri
p0=1e3 #[N] - Dış yük genliği
#wp_wn_orani=0.5 #[-] - Dış yük frekans oranı

wp_wn_oranlari=np.arange(0.1,1.5,0.01)

max_tsds=np.zeros(wp_wn_oranlari.shape)    
# - Her bir omege_p için döngü
for j in range(1,len(wp_wn_oranlari)):
    wp_wn_orani=wp_wn_oranlari[j-1]
    wp=wp_wn_orani*wn #[rad/s] - Dış yük açısal hızı
        
    
    # Zaman vektörü, t
    t_basla=0 #[s]
    t_bitis=30*Tn #[s]
    dt=Tn/30 #[s] - Zaman integrasyonu adımı
    t=np.arange(t_basla,t_bitis+dt,dt)
    nag=len(t)
    
    # Yüklevel vektörü, p(t)
    pts=p0*np.sin(wp*t)
    
    # AKS özellikleri
    mu=0.05 #[-] AKS kütlesi TSDS kütlesi oranı
    m=mu*M #[kg] - AKS kütlesi
    f=1 #[-] - AKS frekansının TSDS frekansına oranı
    w=f*wn #[rad/s] - AKS açısal hızı
    k=m*(w**2) #[N/m] - AKS yay rijitliği
    
    Ms=np.diag([M, m])#[kg] - Sistem kütle matrisi
    # Ks: [N/m] - Sistem rijitlik matrisi
    Ks=np.array([[K+k, -k],
                 [-k,   k]])
    # M^-1*K matrisi
    invMK=np.dot(np.linalg.inv(Ms),Ks)
    # Özdeğer, özvektör çözümü
    V,D=np.linalg.eig(invMK)
    # Özdeğer sıralaması
    idx=V.argsort()[::1]
    V=V[idx]
    D=D[:,idx]
    # ws: [rad/s] - ÇSDS için açısal hızlar
    ws=[np.sqrt(item) for item in V]
    # Ts: [s] - ÇSDS için periyodlar
    Ts=[2*np.pi/item for item in ws]
    
    # r: Dış etki vektörü
    ndof=Ms.shape[0]
    r = np.ones(ndof)
    r[1]=0
    P_exc=np.outer(r,pts)
    
    Cs = np.zeros(Ms.shape) # - Sistem sönüm matrisi
    
    P_exc0=P_exc[:,0]
    invM=np.linalg.inv(Ms)
    
    x0=np.zeros(ndof)
    v0=np.zeros(ndof)
    a0=invM.dot(P_exc0 - Cs.dot(v0) - Ks.dot(x0))
    
    beta=1/4;
    gamma=1/2;
    
    K_tilde = Ks + gamma/(beta*dt)*Cs + 1/(beta*dt**2)*Ms
    C1 = 1/(beta*dt)*Ms + gamma/beta*Cs
    C2 = 1/(2*beta)*Ms + dt*(gamma/(2*beta)-1)*Cs
    
    x = np.zeros((ndof,nag))
    v = np.zeros((ndof,nag))
    a = np.zeros((ndof,nag))
    
    # Newmark zaman integrasyonu döngüsü
    x[:,0]=x0
    v[:,0]=v0
    a[:,0]=a0
    for i in range(1,len(t)):
        dP = P_exc[:,i] - P_exc[:,i-1]
        dP_tilde = dP + C1.dot(v[:,i-1]) + C2.dot(a[:,i-1])
        
        dx = np.linalg.inv(K_tilde).dot(dP_tilde)
        dv = gamma/(beta*dt)*dx - gamma/beta*v[:,i-1] 
        + (1-gamma/(2*beta))*dt*a[:,i-1]
        da = 1/(beta*dt**2)*dx - 1/(beta*dt)*v[:,i-1] - 1/(2*beta)*a[:,i-1]
        
        x[:,i] = x[:,i-1] + dx
        v[:,i] = v[:,i-1] + dv
        a[:,i] = a[:,i-1] + da
        
    # - Omega_p için max_tsds yer değiştirmesini sakla
    max_tsds[j-1]=max(abs(x[0]))

plt.figure(0)
plt.plot(wp_wn_oranlari,max_tsds,label='max_tsds')
plt.legend()
plt.grid()
plt.show()



