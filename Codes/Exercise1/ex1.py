# Dependencies
import matplotlib.pyplot as plt
import numpy as np
from scipy.fft import fft,ifft
from module_1 import *

def rad_to_degree(rad_angle):
    """
    Returns the rad_angle, 
    """
    return 180./np.pi*rad_angle

def degree_to_rad(degree_angle):
    return np.pi/180*degree_angle

# Sampling
# length of the signal
L = 1
# sampling frequency
sf = 12      
# sampling points
sp = sf*L
# time partition 
t_part = np.linspace(0,L,sp,endpoint=False) #[s]
# frequency partition
f_part = np.arange(0,sp/L,1/L)


# Sinewave signal parameters
freq = 1 # frequency of the signal in Hz (1/period)
A = 2. # Amplitude of the signal [V]
phi = 10 # phase shift of the signal [degree]
# Sinewave signal function
omega = 2*np.pi*freq 
func = A*np.sin(omega*t_part + degree_to_rad(phi))

# fourier transform of func
ft = np.empty(sp, dtype = np.complex_)  
ft = fft(func) # backward normalization (i.e, no normalisation for fft, 1/sp for ifft)

# inverse fourier transform of ft
ift = np.empty(sp, dtype = np.complex_)   # *1/sp
ift = ifft(ft)


# modulus and phase spectrum
modulus = np.abs(ft[0:int(sp/2)])
phase = np.arctan2(ft.imag[0:int(sp/2)],ft.real[0:int(sp/2)])

# Index associated with the maximum Fourier's coefficient 
max_mod = np.max(modulus)
index = np.where(modulus==max_mod)

# Amplitute of the signal through maximum fourier coefficient
A_max = max_mod/(sp/2)
phase_max = phase[index]

# PRINTING AMPLIUDE AND PHASE OF THE SIN FUNCTION
print("PRINTING THE VALUES OF THE AMPLITUDE AND THE PHASE OF THE SINEWAVE CALCULATED FROM THE FOURIER'S COEFFICIENTS")
print("----------------------------------------------------------------------------------------------------------")
print("AMPLITUTE:")
print("Expected value = %1.8f"%A)
print("Calculated from the maximum Fourier coefficient = %1.8f"%A_max)
print("-----------------------------------------------")
print("PHASE")
print("Expected value = %1.8f"%phi)
print("Calculated from the maximum Fourier coefficients = %1.8f"%(90+rad_to_degree(phase_max)))

# Creating visualization
fig, ax = plt.subplots(3)

# plotting the sin function and the inverse fourier transform
ax[0].plot(t_part,func,"-",color = "red",label=r"$sin(\omega t)$")
ax[0].plot(t_part,ift.real,".",color = "blue",label =r"$F^{-1}[F[\nu]]$")
ax[0].legend()
ax[0].set_xlabel("Time (s)")
ax[0].set_ylabel(r"$sin(\omega t)$")

# plotting the modulus of the fourier transform
ax[1].plot(f_part[0:int(sp/2)],modulus[0:int(sp/2)])
ax[1].set_xlabel("Frequency [Hz]")
ax[1].set_ylabel("Amplitude")

# plotting the fourier transform (imaginary part)
ax[2].plot(f_part[0:int(sp/2)],phase[0:int(sp/2)])
ax[2].set_xlabel("Frequency [Hz]")
ax[2].set_ylabel("Phase")

plt.show()
