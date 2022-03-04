# Dependencies
import matplotlib.pyplot as plt
import numpy as np
from scipy.fft import fft,ifft
#%matplotlib notebook    #uncomment if you work on jupyter notebook

def power_spectrum(V_in,N):
    """
    Calculates the power spectrum in unit of V^2/Hz
    """
    V_out = np.empty(N,dtype = np.complex_)
    V_out = fft(V_in, axis=0)    
    return (abs(V_out)**2)*T/(2*(N/2)**2)
        
def dB(P_in, P_ref = 1):
    """
    Calculates the power spectrum in unit of dB
    """
    return 20*np.log10(abs(P_in)/P_ref)

def trapezoidal(f,xmin,xmax,N):
    delta_k = (xmax-xmin)/N
    Trap = 0.
    for k in range(1,N):
        Trap += delta_k*(f[k]+f[k-1])/2.0
    return Trap

def power_area(V_in):
    """
    Calculates the power spectrum in unit of V^2/Hz and in dB. 
    In addition to that, it calculates the area underneath the power spectrum in V^2/Hz.
    The returned area should be equal to the square value of V_rms of the signal.
    """
    P_V2s = power_spectrum(V_in,sp)
    area = trapezoidal(P_V2s,0,f[int(sp/2)],int(sp/2))
    P_db = dB(P_V2s,sp)
    return P_V2s,area, P_db

def tot_pow(P_in):
    acc = 0
    for P_k in P_in[0:int(len(P_in)/2)]:
        acc += P_k
    return acc/T

# sampling frequency
fs = 500 # Hz
# Period of time
T = 2 #s
# sampling point
sp = fs*T
# time array
t = np.linspace(0,T,sp)
# frequency array
f = np.arange(0,(sp)/T,1./T)

#----------------------------------------------------------------------------
# Signal 1: white noise
from numpy.random import normal

def white_noise(time,amp = 1):
    return amp*normal(0,1,len(time))

# noise parameters
Temp = 300   #K
R = 1e6   #ohm
df = 1e4 #Hz
kb = 1.38064852e-23 # m2 kg s-2 K-1 Boltzmann constant
Vnoise = np.sqrt(4*kb*Temp*R*df) #Johnson-Nyquist noise

V_1 = white_noise(t,Vnoise)
#---------------------------------------------------------------------------
# Signal 2: sin function
A2 = 12e-6 # V amplitude
freq2 = 80  # Hz
V_2 = A2*np.sin(2*np.pi*freq2*t)

#---------------------------------------------------------------------------
# Signal 3: sin function
A3 = 6e-6   # V
freq3 = 170 # Hz
V_3 = A3*np.sin(2*np.pi*freq3*t)
#---------------------------------------------------------------------------
# Calculating the power spectrum (V^2/Hz, dB) and the area underneath the signal

P_V2s_1, area_1, P_db_1 = power_area(V_1)
P_V2s_2, area_2, P_db_2 = power_area(V_2)
P_V2s_3, area_3, P_db_3 = power_area(V_3)

pow1 = tot_pow(P_V2s_1)
pow2 = tot_pow(P_V2s_2)
pow3 = tot_pow(P_V2s_3)
pow_th_comb = Vnoise**2+A2**2/2+A3**2/2  # theoretical value of the power for the three (uncorrelated) signals
 
#---------------------------------------------------------------------------
# Combining signals:
V_comb = V_1 + V_2 + V_3
P_V2s_comb, _ , _ = power_area(V_comb)
pow_comb = tot_pow(P_V2s_comb)
#---------------------------------------------------------------------------

print("Signal 1: V_rms^2:",Vnoise**2," Area:",area_1)
print("Signal 2: V_rms^2:", A2**2/2," Area:", area_2)
print("Signal 3: V_rms^2:", A3**2/2, "Area:", area_3)

print("Power of the first signal:", pow1)
print("Power of the second signal:", pow2)
print("Power of the third signal:", pow3)
print("Theorical power of the combined signal:", pow_th_comb, "Calculated power:", pow_comb)

fig1, ax1 = plt.subplots(3, sharex = True)
fig1.suptitle('Voltage signals (V)', fontsize=13)
ax1[0].plot(t, V_1)
ax1[1].plot(t, V_2)
ax1[2].plot(t,V_3)
ax1[2].set_xlabel("Time (s)")

fig2, ax2 = plt.subplots(3, sharex = True)
fig2.suptitle(r'Power spectra $(\frac{V^2}{Hz})$', fontsize=13)
ax2[0].plot(f[0:int(sp/2)], P_V2s_1[0:int(sp/2)])
ax2[1].plot(f[0:int(sp/2)], P_V2s_2[0:int(sp/2)])
ax2[2].plot(f[0:int(sp/2)], P_V2s_3[0:int(sp/2)])
ax2[2].set_xlabel("Frequency (Hz)")

fig3, ax3 = plt.subplots(3, sharex = True)
fig3.suptitle(r'Power spectra (dB)', fontsize=13)
ax3[0].plot(f[0:int(sp/2)], P_db_1[0:int(sp/2)])
ax3[1].plot(f[0:int(sp/2)], P_db_2[0:int(sp/2)])
ax3[2].plot(f[0:int(sp/2)], P_db_3[0:int(sp/2)])
ax3[2].set_xlabel("Frequency (Hz)")

fig4, ax4 = plt.subplots()
fig4.suptitle('Combined signal power spectrum' ,fontsize = 13)
ax4.plot(f[0:int(sp/2)], P_V2s_comb[0:int(sp/2)])
ax4.set_ylabel(r"Power $(\frac{V^2}{Hz})$")
ax4.set_xlabel("Frequency (Hz)")


#---------------------------------------------------------------------------------------
# Dependence on the sampling frequency and on the length of the signal

# sampling frequency
fs = [100,300,500,750,1000] # Hz
# length of the signal
T = [0.5,2.5,5,7.5,10]

fig5, ax5 = plt.subplots(len(fs),sharex = True)
fig5.suptitle('Dependence on the sampling frequency', fontsize=13)
ax5[len(fs)-1].set_xlabel("time (s)")
length = 2 #s
for i,freq in enumerate(fs):
    sp = int(freq*length)
    t = np.linspace(0,length,sp)
    V_3 = A3*np.sin(2*np.pi*freq3*t)
    ax5[i].plot(t,V_3, label= str(freq)+" Hz")
    ax5[i].axhline(-A3, color = "red")
    ax5[i].axhline(A3,color = "red")
    ax5[i].set_ylim(-7e-6,7e-6)
    ax5[i].set_xlim(0,0.3)
    ax5[i].legend(loc = "right")

fig6, ax6 = plt.subplots(len(T),sharex = True)
fig6.suptitle('Dependence on the signal length', fontsize=13)
ax6[len(T)-1].set_xlabel("time (s)")
freq = 500 # Hz
for i,length in enumerate(T):
    sp = int(freq*length)
    t = np.linspace(0,length,sp)
    V_3 = A3*np.sin(2*np.pi*freq3*t)
    ax6[i].plot(t,V_3, label= str(length)+" sec")
    ax6[i].axhline(-A3, color = "red")
    ax6[i].axhline(A3,color = "red")
    ax6[i].set_ylim(-7e-6,7e-6)
    ax6[i].set_xlim(0,0.3)
    ax6[i].legend(loc = "right")
    
plt.show()
