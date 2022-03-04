# Dependencies
import matplotlib.pyplot as plt
import numpy as np
from scipy.fft import fft,ifft

# length of the signal
t_fin = 1
# sampling frequency
fs = 12      
# sampling points
sp = fs*t_fin
# time partition 
t = np.linspace(0,t_fin,sp) #[s]
# frequency
f = np.arange(0,sp/t_fin,1/t_fin)


# function
freq = 1 #[Hz]
omega = 2*np.pi*freq
func = np.sin(omega*t)

# fourier transform of func
ft = np.empty(sp, dtype = np.complex_)
ft = fft(func) # by default is operating with backward normalisation (i.e, not multiplying by any prefactor)

# inverse fourier transform of ft
ift = np.empty(sp, dtype = np.complex_)   # *1/sp
ift = ifft(ft)


fig, ax = plt.subplots(3)
ax[1].plot(f,ft.real)
ax[1].set_xlabel("Frequency [Hz]")
ax[1].set_ylabel(r"$Re(F[\nu])$")
ax[2].plot(f,ft.imag)
ax[2].set_xlabel("Frequency [Hz]")
ax[2].set_ylabel(r"$Im(F[\nu])$")
ax[0].plot(t,func,"-",color = "red",label=r"$sin(\omega t)$")
ax[0].plot(t,ift.real,".",color = "blue",label =r"$F^{-1}[F[\nu]]$")
ax[0].legend()
ax[0].set_xlabel("Time (s)")
ax[0].set_ylabel(r"$sin(\omega t)$")

plt.show()
