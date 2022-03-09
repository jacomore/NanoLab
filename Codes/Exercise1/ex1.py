# Dependencies
import matplotlib.pyplot as plt
import numpy as np
from scipy.fft import fft,ifft

# Sampling
# length of the signal
L = 1
# sampling frequency
sf = 12      
# sampling points
sp = sf*L
# time partition 
t_part = np.linspace(0,L,sp) #[s]
# frequency partition
f_part = np.arange(0,sp/L,1/L)


# Sinewave signal parameters
freq = 1 # frequency of the signal in Hz (1/period)
A = 5. # Amplitude of the signal [V]
phi = 0# phase shift of the signal [rad]
# Sinewave signal function
omega = 2*np.pi*freq 
func = A*np.sin(omega*t_part + phi)

def power_spectrum(Vin,N,T):
    """
    calculate the power spectrum of a signal Vin
    -----------------------------------------------
    input: 
        -Vin: N-dimensional, REAL or COMPLEX. Input signal
        -N: INTEGER. number of sampling points
        -T: REAL or INTEGER. Length of the signal
    """
    tmp = fft(Vin)  
    Vft  = np.empty(int(N/2)+1, dtype = np.complex_)
    Vft = tmp[0:int(N/2)]   # selecting only half of the fourier transform spectrum
    return np.abs(Vft)**2*T/(2*(N/2)**2)

def tot_power(Power,N,T):
    """
    Calculate the area of the power spectrum, i.e, the total power in V^2 of a signal
    --------------------------------------------------------------------------------
    input: 
        - Power: COMPLEX, N-dimensional. Power spectrum of the signal
        - N: INTEGER. Number of sampling points
        - T: REAL or INTEGER: number of sampling points
    output: 
        - REAL. Total power 
    """
    tmp = 0 
    for i in range(1,int(N/2)):
        tmp+=Power[i]
    return tmp/T

# fourier transform of func
ft = np.empty(sp, dtype = np.complex_)  
ft = fft(func) # backward normalization (i.e, no normalisation for fft, 1/sp for ifft)

# inverse fourier transform of ft
ift = np.empty(sp, dtype = np.complex_)   # *1/sp
ift = ifft(ft)

# Power spectrum
P = power_spectrum(func,sp,L)

# Amplitude of the signal through power spectrum
V_rms_2 = tot_power(P,sp,L) # square value of the V_rms 
A_ft = np.sqrt(2*V_rms_2) 

# modulus and phase spectrum
modulus = np.abs(ft[0:int(sp/2)])
phase = np.arctan2(ft.imag[0:int(sp/2)],ft.real[0:int(sp/2)])

# Index associated with the maximum Fourier's coefficient 
max_mod = np.max(modulus)
index = np.where(modulus==max_mod)

# Amplitute of the signal through maximum fourier coefficient
A_max = max_mod/(sp/2) # divided by half the sampling point (we are considering only half the interval)
phase_max = phase[index[0]]

print("AMPLITUTE:")
print("Expected value:%1.5f"%A)
print("From the power spectrum:%1.5f"%A_ft)
print("From the maximum Fourier coefficient:%1.5f"%A_max)

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
