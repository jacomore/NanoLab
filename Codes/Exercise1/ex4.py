import matplotlib.pyplot as plt
import numpy as np
from scipy.fft import fft,ifft

def gaussian(x,y0,A,x0,width): 
    return y0 + A*np.exp(-(x-x0)**2/(2*width**2))


# Length of the signal
L = 1.25   #[s]
# sampl. freq
fs = 1000   #[Hz]
# sampling period
T = 1/fs   
# sam. points
sp = int(fs*L)

# time partition 
t_part = np.linspace(0,L,sp,endpoint=False) 
# frequency partitio
f_part = np.arange(0,sp/L,1/L)

# sinewave definition
freq = 10    #[Hz]
sine = np.sin(2*np.pi*freq*t_part)

# Fourier transform of sinewave
sine_ft = fft(sine)

# Creating the windowing function.
# A Gaussian function with standard deviation (s) equals to L/6 and centred on the L/2, so that the 3*s falls at the edges of the interval [0, L]
sigma = L/6.
y0 = 0.
x0 = L/2.
A = 1     # to avoid shrinking/enhancing the amplitude of the sinewave function at the centre of [0,L]
window = gaussian(t_part,y0,A,x0,sigma)   

# product of the two functions
sine_wind = sine*window

# Fourier transform of sine_wind
sine_wind_ft = fft(sine_wind)

# Creating visualization
fig, ax = plt.subplots()

# Plotting sine and sine_wind in real space
ax.set_title("Real Space")
ax.plot(t_part,sine,label = "sine")
ax.plot(t_part,sine_wind,label = "windowed sine")
ax.legend()

fig1,ax1 = plt.subplots(2,sharex=True)
# Plotting sine_ft and sine_wind_ft in reciprocal space
ax1[1].set_xlabel("Frequency [Hz]")
ax1[1].set_xscale("log")
ax1[0].set_yscale("log")
ax1[0].plot(f_part[0:int(sp/2)],abs(sine_ft[0:int(sp/2)]))
ax1[1].set_yscale("log")
ax1[1].plot(f_part[0:int(sp/2)],abs(sine_wind_ft[0:int(sp/2)]))


plt.show()
