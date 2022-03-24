# Dependencies
import matplotlib.pyplot as plt
import numpy as np
from scipy.fft import fft,ifft
from module_1 import *

# Sampling
# length of the signal
L = 2
# sampling frequency
sf = 500      
# sampling points
sp = sf*L

# time partition 
t_part = np.linspace(0,L,sp,endpoint=False) #[s]
# frequency partition
f_part = np.arange(0,sp/L,1./L)

#---------------------------------------------------------
# Signal 1: white noise
from numpy.random import normal

def white_noise(N,A = 1):
    return A*normal(0,1,N)

# noise parameters
T = 300   #Temperature [K]
R = 1e6   #Resistance [Ohm]
df = 1e4 #Bandwidth [Hz]
kb = 1.38064852e-23 #[m2 kg s-2 K-1] Boltzmann constant
Vrms = np.sqrt(4*kb*T*R*df) #Johnson-Nyquist noise

V_1 = white_noise(sp,Vrms)  
#---------------------------------------------------------
# Signal 2: sinewave

A2 = 12e-6 # Signal amplitude [V]
freq2 = 80  # Frequency of the sinewave [Hz]
V_2 = A2*np.sin(2*np.pi*freq2*t_part)
#---------------------------------------------------------
# Signal 3: sinewave
A3 = 6e-6   
freq3 = 170 
V_3 = A3*np.sin(2*np.pi*freq3*t_part)
#---------------------------------------------------------

# Combining all the three signals
V_comb = V_1 + V_2 + V_3

# Evaluating the power spectrum of the combined signal in V^2/Hz and in dB
P_V_comb = power_spectrum(V_comb,sp,L)
P_V_comb_dB = dB(P_V_comb)

# Evaluating the V_rms from statistical argument
V_stat = np.std(V_comb)

# Integrating over the power spectrum by means of the trapezoidal formula
f_min = 0.
f_max = f_part[int(sp/2)]
area_P_comb = trapezoidal(P_V_comb[0:int(sp/2)],f_min,f_max,int(sp/2))

print("COMBINED SIGNAL: V_1 + V_2 + V_3")
print("Statistical V_rms^2",V_stat**2)
print("V_rms^2 from the trapezoidal formula",area_P_comb)
print("--------------------------------------------------------------------------------------------")


# Evaluating the V_rms from the Power spectrum
totP_V_comb = tot_power(P_V_comb,sp,L)

# Evaluating the theoretical total power of the combined signal
totP_th = Vrms**2+A2**2/2+A3**2/2  # theoretical value of the power for the three (uncorrelated) signals

# Calculating the power spectrum in V^2/Hz of the single components
P_V1 = power_spectrum(V_1,sp,L)
P_V2 = power_spectrum(V_2,sp,L)
P_V3 = power_spectrum(V_3,sp,L)

# Evaluating the total power of the three signal
totP_V1 = tot_power(P_V1,sp,L)
totP_V2 = tot_power(P_V2,sp,L)
totP_V3 = tot_power(P_V3,sp,L)

# Printing results
print("SIGNAL 1")
print("theoretical V_rms:",Vrms**2)
print("V_rms calculated from the sum over the power spectrum component:",totP_V1)
print("--------------------------------------------------------------------------------------------")
print("SIGNAL 2")
print("theoretical V_rms:",A2**2/2)
print("V_rms calculated from the sum over the power spectrum component:",totP_V2)
print("--------------------------------------------------------------------------------------------")
print("SIGNAL 3")
print("theoretical V_rms:",A3**2/2)
print("V_rms calculated from the sum over the power spectrum component:",totP_V3)
print("--------------------------------------------------------------------------------------------")
print("COMBINED SIGNAL: V_1 + V_2 + V_3")
print("Theoretical power:",totP_th)
print("Power from the power spectrum of the combined signal",totP_V_comb)
print("Power from the sum of the power of the single components",totP_V1+totP_V2+totP_V3)# Plotting the three signals as a function of time

fig1, ax1 = plt.subplots(3, sharex = True)
fig1.suptitle('Voltage signals (V)', fontsize=13)
ax1[0].plot(t_part, V_1)
ax1[1].plot(t_part, V_2)
ax1[2].plot(t_part,V_3)
ax1[2].set_xlabel("Time (s)")

#Plotting the power spectrum of the combined signal
fig4, ax4 = plt.subplots()
fig4.suptitle('Combined signal power spectrum' ,fontsize = 13)
ax4.plot(f_part[0:int(sp/2)], P_V_comb[0:int(sp/2)])
ax4.set_ylabel(r"Power $(\frac{V^2}{Hz})$")
ax4.set_xlabel("Frequency (Hz)")


#---------------------------------------------------------------------------------------
# Dependence on the sampling frequency and on the length of the signal

# sampling frequency
sf = [100,300,500,750,1000] # Hz
# length of the signal
L = [0.5,2.5,5,7.5,10]

# Creating visualization to observe the dependence on the sampling frequency
fig5, ax5 = plt.subplots(len(sf),sharex = True)
fig5.suptitle('Dependence on the sampling frequency', fontsize=13)
ax5[len(sf)-1].set_xlabel("time (s)")

# the length is fixed and the sampling frequeny varies
length = 2 #s
# iterating over the various sampling frequency
for i,freq in enumerate(sf):
    sp = int(freq*length)
    t = np.linspace(0,length,sp,endpoint=False)
    V_3 = A3*np.sin(2*np.pi*freq3*t)
    ax5[i].plot(t,V_3, label= str(freq)+" Hz")
    ax5[i].axhline(-A3, color = "red")
    ax5[i].axhline(A3,color = "red")
    ax5[i].set_ylim(-7e-6,7e-6)
    ax5[i].set_xlim(0,0.3)
    ax5[i].legend(loc = "right")

# Creating visualization to observe the dependence on the sampling frequency
fig6, ax6 = plt.subplots(len(L),sharex = True)
fig6.suptitle('Dependence on the signal length', fontsize=13)
ax6[len(L)-1].set_xlabel("time (s)")

# the sampling frequency is fixed and the length of the signal varies
freq = 500 # Hz
# iterating over the various length of the signal
for i,length in enumerate(L):
    sp = int(freq*length)
    t = np.linspace(0,length,sp,endpoint=False)
    V_3 = A3*np.sin(2*np.pi*freq3*t)
    ax6[i].plot(t,V_3, label= str(length)+" sec")
    ax6[i].axhline(-A3, color = "red")
    ax6[i].axhline(A3,color = "red")
    ax6[i].set_ylim(-7e-6,7e-6)
    ax6[i].set_xlim(0,0.3)
    ax6[i].legend(loc = "right")


plt.show()
