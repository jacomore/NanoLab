# Dependencies
import matplotlib.pyplot as plt
import numpy as np
from scipy.fft import fft,ifft
from scipy.optimize import curve_fit 
import pandas as pd

def PeaksFinder(arr,std):
    """
    Calculates the multiple peaks (if any) of arr, along with the indeces at which they occur.
    ------------------------------------------------------------------------------------------
    input: 
        arr: REAL, N-dimensional. Array whose maxima have to be computed. 
        std: REAL. Standard deviation of the array calculated in the region without peaks. 
    output: 
        pos: list, INTEGER.  
    """
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


def Z_RC_par_mod(freq,R,C):
    """
    Calculates the modulus of the complex impedance of an RC parallel circuit
    -------------------------------------------------------------------------
    input: 
        freq: REAL, N-dimensional. An array with the frequencies in Hz
        R: REAL. Resistance of the circuit
        C: REAL. Capacitance of the circuit
    output:
        modulus of the impedance 
    """
    return R / np.sqrt(1+(2*np.pi*C*R*freq)**2)

def Z_RC_par_phase(freq,tau):
    """
    Calculates the phase of the complex impedance of an RC parallel circuit
    ------------------------------------------------------------------------
    input: 
        freq: REAL, N-dimensional. An array with the frequencies in Hz
        tau: REAL. tau = RC, is the characteristic time.
    output:
        phase of the impedance
    """  
    return np.arctan(-2*np.pi*freq*tau)

#selecting the x-range to fit
def select_range(df,header,x_min,x_max):
    """
    Returns a logic array whose components are TRUE when the elements of df[header] 
    are within [xmin,xmax], FALSE elsewhere.
    ---------------------------------------------------------------------------------
    input: 
        df: Pandas Dataframe object. (NXM), with M number of columns and N number of rows
        header: String object. Contains the name of the column of df to which xmin,xmax are referred
        xmin,xmax: REAL. Extremes of a subinterval of df[header]
    output:
        selected_df: array, N-dimensional. Its elements are FALSE where df[header] is not within [xmin,xmax]
    """
    selected_df = df.loc[(df[header] >= x_min) & (df[header] <= x_max)]
    return selected_df

#Importing the excel data through the pandas function read_excel
data = pd.read_excel("../../Data/data_DAQ.xls")

# Interpreting and converting data into numpy arrays 
time = np.array(data["time(s)"])
volt = np.array(data["input(V)"])
amp = np.array(data["output(A)"])

# sampling point
sp = len(time)

#Defining length of the signal (L) and sampling frequency (sf)
L = time[-1]-time[0]  
sf = 1/(time[1]-time[0])

# Defining frequency partition
f_part = np.arange(0,(sp)/L,1./L)

#Calculting DFT of volt and amp 
v_dft = fft(volt)
a_dft = fft(amp)

# Calculating stardard deviation of voltage and current in a FLAT interval
# 1. Creating a new Pandas DataFrame that contains the FFT of volt, amp
df_freq = pd.DataFrame(columns =["volt", "amp", "freq"])
df_freq["volt"] = v_dft
df_freq["amp"] = a_dft
df_freq["freq"] = f_part
# 2. Selecting the "flat" range where the standard deviation is computed
x_min = 1e4
x_max = 1.8e4
df_flat = select_range(df_freq,"freq",x_min,x_max)
# 3. Using  std() method to calculate the standard deviation of v_dft and a_dft
std_values = df_flat.std()  
v_std = std_values[0]
a_std = std_values[1]

# Finding the peaks position ror v_dft 
pos, _ = PeaksFinder(v_dft[0:int(sp/2)],v_std)

# calculating the impedence
Z = v_dft/a_dft
Z_peak = Z[pos]

# Calculating phase and modulus of the impedance
Z_phase = np.arctan2(Z_peak.imag,Z_peak.real)
Z_modulus = np.abs(Z_peak)

#  Creating a Bode plot of the impedance
fig7, ax7 = plt.subplots(2,sharex = True)
fig7.suptitle(r'Bode plot', fontsize=12)
ax7[0].plot(f_part[pos],Z_modulus,".-",markersize=10, label = "Modulus")
ax7[0].set_yscale('log')
ax7[0].set_ylabel(r'$log_{10}(|Z|)$')
ax7[1].plot(f_part[pos],Z_phase*180/np.pi,".-",markersize=10, label ="Phase")
ax7[1].set_ylabel(r'$atan2(\frac{Im(Z)}{Re(Z)}) \ [\theta^°]$')

# Guessing the value of resistance R and capacitance C
R = 1e5   # R = 100 KOhm
C = 1e-9  # C = 1nF
tau = R*C

# Defining a frequency array for plotting
freq_arr = np.linspace(pos[0],20000,1000)

# Fitting the phase of the impedance
par_phase, covs_phase = curve_fit(Z_RC_par_phase,f_part[pos],Z_phase,
                                               p0=[tau])

covs_phase = np.sqrt(np.diag(covs_phase))
# Fitting the modulus of the impedace 
par_mod, covs_mod = curve_fit(Z_RC_par_mod,f_part[pos], Z_modulus,
                                               p0=(R,C))
covs_mod = np.sqrt(np.diag(covs_mod))

print("Printing results of the curve fittings on the phase and modulus of the impedance")
print("")

print("CURVE FITTING OF THE PHASE OF THE IMPEDANCE")
print("Parameter: tau")
print("Tau: ",par_phase[0],"+-",covs_phase[0],"s")
print("-------------------------------------------")

tau_mod = par_mod[0]*par_mod[1]
err_tau_mod = np.sqrt((par_mod[1]*covs_mod[0])**2+(par_mod[0]*covs_mod[1])**2)

print("CURVE FITTING OF THE PHASE OF THE IMPEDANCE")
print("Parameters: Resistance, Capacitance")
print("Resistance: ",par_mod[0],"+-",covs_mod[0],"Ohm")
print("Capacitance: ",par_mod[1],"+-",covs_mod[1],"F")
print("Tau:",tau_mod,"+-",err_tau_mod,"s")
print("---------------------------------------------")


tau_phase_mod = abs(par_phase[0]-tau_mod)
err_tau_phase_mod = np.sqrt(err_tau_mod**2 + covs_phase[0]**2)

print("COMPATIBILITY OF THE TWO TAU")
print("|Tau(phase)- Tau(modulus)|=", tau_phase_mod ,"+-",err_tau_phase_mod,"s")
fit_phase = Z_RC_par_phase(freq_arr,par_phase)
fit_modulus = Z_RC_par_mod(freq_arr,par_mod[0],par_mod[1])

fig8,ax8 = plt.subplots(2)
ax8[0].plot(f_part[pos],Z_phase,".",markersize = 10 ,color = "red", label ="data_DAQ")
ax8[0].plot(freq_arr, fit_phase, color = "blue", label ="Fitting")
ax8[0].set_title("Impedance phase")
ax8[0].legend()
ax8[1].plot(f_part[pos],Z_modulus,".", markersize = 10,color = "red", label ="data_DAQ")
ax8[1].plot(freq_arr, fit_modulus, color = "blue", label ="Fitting")
ax8[1].set_title("Impedance modulus")
ax8[1].legend()
fig8.tight_layout()

plt.show()
