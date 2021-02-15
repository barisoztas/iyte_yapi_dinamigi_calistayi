import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from scipy.fft import fft, ifft
import scipy.fftpack

df = pd.read_csv('acc_data.txt', sep = ";")
time = df["index"]/6
time_ = np.array(time)
df_ = np.array(df["x"])


fig1, ax1 = plt.subplots()
lw = 0.5
ax1.set_title("Acceleration in Y Direction")
ax1.plot(time_, df_, linewidth =lw)
ax1.set_xlim(-10,360)
ax1.grid(True)
ax1.set_xlabel("time (sec)")
ax1.set_ylabel("Acceleration (m/s^2)")



# Fast Fourier Transform


yf = fft(df_)
x = scipy.fftpack.fftfreq(yf.size, 1 / 50)
plt.figure(figsize=(38,10))
plt.xticks(np.arange(min(x), max(x)+1, 0.2),fontsize=5)
plt.plot(x[:x.size//2], abs(yf)[:yf.size//2])
plt.grid(True)
plt.savefig('Fast Fourier Transform.png', dpi=500)

plt.show()