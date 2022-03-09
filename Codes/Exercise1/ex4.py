import matplotlib.pyplot as plt
import numpy as np
from scipy.fft import fft,ifft

def gaussian(x,y0,A,x0,width): 
    return y0 + A*np.exp(-(x-x0)**2/(2*width**2))


L = 1.25
# sampl. freq
fs = 100
# sampling period
T = 1/fs
# sam. points
sp = fs*int(L)
# time partition 
t = np.linspace(0,L,sp) #[s]
# frequency
freq = np.arange(0,sp/L,1/L)

# sine definition
f = 10
y = np.sin(2*np.pi*f*t)
y_nowind = fft(y)


# gaussian
window = gaussian(t,0,1.,L/2,L/6.)  #  3 sigma 

y_wind = fft(y*window)

fig, ax = plt.subplots(2)
ax[0].plot(f[0:int(sp/2)],y_nowind[0:int(sp/2)])
ax[1].plot(f[0:int(sp/2)],y_wind[0:int(sp/2)])

plt.show()
