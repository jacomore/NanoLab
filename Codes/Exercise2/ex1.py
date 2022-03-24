import matplotlib.pyplot as plt
import numpy as np

def Re_Z(f,R1,R2,C2):
    w = 2*np.pi*f
    return R1 + R2/(1+(w*C2*R2)**2)

def Im_Z(f,R1,C1,R2,C2):
    w = 2*np.pi*f
    return -1.0/(w*C1)-w*C2*R2**2/(1+(w*C2*R2)**2)

R1 = 90 # Ohm
C1 = 20e-6 # F
R2 = 7e3 # Ohm
C2 = 0.3e-6 # F

# Frequency extremes
f_min = 1 #Hz
f_max = 1e6 #Hz

# sampling frequency
sf = 1 #Hzu
# sampling points 
sp = (f_max-f_min)*sf

# Frequency partition
f_part = np.geomspace(f_min,f_max,int(sp))

# Calculatin real and imaginary part of the impedance
Z_real = Re_Z(f_part,R1,R2,C2)
Z_imag = Im_Z(f_part,R1,C1,R2,C2)


# Calculating modulus and phase of the impedance
Z_modulus = np.sqrt(Z_real**2 + Z_imag**2)
Z_phase = np.arctan2(Z_imag,Z_real)

# Creating visualisation
fig, ax = plt.subplots(2, sharex = True)
ax[0].plot(f_part,20*np.log10(Z_modulus))
ax[0].set_ylabel("|Z| [dB]")
ax[1].plot(f_part,180/(np.pi)*Z_phase)
ax[1].set_ylabel("Phase [°]")
ax[1].set_xscale("log")
ax[1].set_xlabel("Frequency [Hz]")
#---------------------------------------------------------

#Johnson-Nyquist Noise formula
k_b = 1.38064852e-23 #m2 kg s-2 K-1
T = 300
S = 4*T*k_b*Z_real

# Creating visualization
fig1,ax1 = plt.subplots()
ax1.semilogx(f_part,10*np.log10(S))
ax1.set_ylabel(r"$S_{\nu}$ [dB]")
ax1.set_xlabel("Frequency [Hz]")
plt.show()

#----------------------------------------------------------

Z = Z_real + 1j*Z_imag
