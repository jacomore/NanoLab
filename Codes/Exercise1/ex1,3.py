# Dependencies
import matplotlib.pyplot as plt
import numpy as np
from scipy.fft import fft,ifft
from scipy.optimize import curve_fit 
import pandas as pd


def PeaksFinder(arr,std):
    n = len(arr)
    peaks = []  # contains the values of arr in correspondence of the peaks
    pos = []   # contains the indeces of arr in correspondence of the peaks
    
    # First element 
    if (arr[0]-arr[1] >= 3*std):
        peaks.append(arr[0])
        pos.append(0)
    
    # from the second to the one before the last
    for i in range(1, n-1):
        if (arr[i] - arr[i-1] >= 3*std and arr[i] - arr[i+1]>= 3*std):
            peaks.append(arr[i])
            pos.append(i)
    # last element
    if (arr[n-1] - arr[n-2] >= 3*std):
        peaks.append(arr[n-1])
        pos.append(n-1)
    
    return pos, peaks


def modulus(freq,R,C):
    return R / np.sqrt(1+(2*np.pi*C*R*freq)**2)

def phase(freq,R,C):
    return np.arctan(-2*np.pi*freq*C*R)

#selecting the x-range to fit
def select_range(df,header,x_min,x_max):
    selected_df = df.loc[(df[header] >= x_min) & (df[header] <= x_max)]
    return selected_df

#Importing data
data = pd.read_excel("data_DAQ.xls")

# Interpreting and converting data into numpy arrays
time = np.array(data["time(s)"])
volt = np.array(data["input(V)"])
amp = np.array(data["output(A)"])

# sampling point
sp = len(time)
#Defining length of the signal (L) and sampling frequency (sf)
L = time[-1]-time[0]
sf = 1/(time[1]-time[0])

# Defining frequency array
f = np.arange(0,(sp)/L,1./L)

#Calculting DFT of volt and amp 
v_dft = fft(volt)
a_dft = fft(amp)

# Calculating stardard deviation of voltage and current in a flat interval
df_freq = pd.DataFrame(columns =["volt", "amp", "freq"])
df_freq["volt"] = v_dft
df_freq["amp"] = a_dft
df_freq["freq"] = f

x_min = 1e4
x_max = 1.8e4
df_flat = select_range(df_freq,"freq",x_min,x_max)

std_values = df_flat.std()  
v_std = std_values[0]
a_std = std_values[1]

# Finding the peaks position ror v_dft 
pos, _ = PeaksFinder(v_dft[0:int(sp/2)],v_std)


# calculating the impedence
Z = v_dft/a_dft
Z = Z[pos]

# Calculating phase and modulus of the impedance
Z_phase = np.arctan(Z.imag/Z.real)
Z_modulus = abs(Z)

#  Creating a Bode plot of the impedance
fig7, ax7 = plt.subplots(2,sharex = True)
fig7.suptitle(r'Bode plot', fontsize=12)
ax7[0].plot(f[pos],Z_modulus, label = "Modulus")
ax7[0].set_yscale('log')
ax7[0].set_ylabel(r'$log_{10}(|Z|)$')
ax7[0].set_xscale('log')
ax7[1].plot(f[pos],abs(Z_phase),label ="Phase")
ax7[1].set_yscale('log')
ax7[1].set_ylabel(r'$log_{10}(|atan(\frac{Im(Z)}{Re(Z)})|)$')
ax7[1].set_xscale('log')

# Defyining a frequency array for plotting
freq_arr = np.linspace(pos[0],20000,1000)

print(Z_phase)


# Fitting the phase of the impedance
par_phase, covs_phase = curve_fit(phase,f[pos], Z_phase,
                                               p0=[1e5,1e-9])


# Fitting the modulus of the impedance 
par_mod, covs_mod = curve_fit(modulus,f[pos], Z_modulus,
                                               p0=(1e5,1e-9))

print("Fit result of the phase of the impedance")
print("Resistance: ", par_phase[0],"Ohm","Capacitance: ",par_phase[1],"F")

print("Fit results of the modulus of the impedance")
print("Resistance: ",par_mod[0],"Ohm","Capacitance: ",par_mod[1],"F")

fit_phase = phase(par_phase[0],par_phase[1],freq_arr)
fit_modulus = modulus(freq_arr,par_mod[0],par_mod[1])

fig8,ax8 = plt.subplots(2)
ax8[0].plot(f[pos],Z_phase,".",markersize = 10 ,color = "red", label ="data_DAQ")
ax8[0].plot(freq_arr, fit_phase, color = "blue", label ="Fitting")
ax8[0].set_title("Impedance phase")
ax8[0].legend()
ax8[1].plot(f[pos],Z_modulus,".", markersize = 10,color = "red", label ="data_DAQ")
ax8[1].plot(freq_arr, fit_modulus, color = "blue", label ="Fitting")
ax8[1].set_title("Impedance modulus")
ax8[1].legend()
fig8.tight_layout()

plt.show()
