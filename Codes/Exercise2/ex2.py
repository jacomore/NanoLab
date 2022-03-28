import pandas as pd 
import matplotlib.pyplot as plt
import numpy as np 
from scipy.optimize import minimize

def rad_to_degree(rad_angle):
    """
    Returns the angle in degree if provided in radiant.
    INPUT: 
        - rad_angle: REAL. Angle [rad]
    OUTPUT: angle [degree]
    """
    return 180./np.pi*rad_angle

def Re_Z(f,R1,R2,C2):
    """
    Return the real part of the complex impedance of a circuit formed by a series of a resistance (R1), a capacitor (C1) and a parallel of a resistance (R2) and a capacitor (C2).
    ---------------------------------------------------------------------
    INPUT: 
        f: array, N dimensional. frequency array
        R1, C1, R2, C2: real, scalar. Resistnces and capacitors.
    OUTPUT:
        Real part of the impedance
    """
    w = 2*np.pi*f
    return R1 + R2/(1+(w*C2*R2)**2)

def Im_Z(f,R1,C1,R2,C2):
    """
    Returns the imaginary part of the complex impedance of a circuit formed by a series of a resistance (R1), a capacitor (C1) and a parallel of a resistance (R2) and a capacitor (C2).
    ---------------------------------------------------------------------
    INPUT: 
        f: array, N dimensional. frequency array
        R1, C1, R2, C2: real, scalar. Resistnces and capacitors.
    OUTPUT:
        Real part of the impedance
    """
    w = 2*np.pi*f
    return -1.0/(w*C1)-w*C2*R2**2/(1+(w*C2*R2)**2)


def tot_Z(f,R1,C1,R2,C2):
    """
    Returns the complex impedance of a circuit formed by a series of a resistance (R1), a capacitor (C1) and a parallel of a resistance (R2) and a capacitor (C2).
    ---------------------------------------------------------------------
    INPUT: 
        f: array, N dimensional. frequency array
        R1, C1, R2, C2: real, scalar. Resistnces and capacitors.
    OUTPUT:
        Complex impedance
    """
    return Re_Z(f,R1,R2,C2) + Im_Z(f,R1,C1,R2,C2)*1j

def single_error(Z_mea, Z_calc):
    """
    Returns the sum of the difference of the relative errors between the measured impedance (Z_mea)
    and the calculated impedance (Z_calc)
    -----------------------------------------------------------------------------------------------
    INPUT: 
        Z_mea: complex, scalar. Measure impedance associated to a single frequency.
        Z_calc: complex, scalar. Complex impedance associated to a single frequency.
    OUTPUT: 
        Relative error given by the sum of the two relative errors.
    """
    Re_term = np.abs(np.real(Z_mea-Z_calc)/np.real(Z_mea))  
    Im_term = np.abs(np.imag(Z_mea - Z_calc)/np.imag(Z_mea))
    return Re_term + Im_term

def total_error(pars,freq,Z_mea):
    """
    Returns the total error associated with the difference between the calculated and measured
    function
    ----------------------------------------------------------------------------------------
    INPUT: 
       Z_mea: complex, scalar. Measure impedance associated to a single frequency.
       freq: array, N dimensional. frequency array
       R1, C1, R2, C2: real, scalar. Resistnces and capacitors.

    """
    R1, C1, R2, C2 = pars[0], pars[1], pars[2], pars[3]
    Z_calc = tot_Z(freq,R1,C1,R2,C2)
    acc = 0.
    for i in range(len(Z_calc)):
        acc += single_error(Z_mea[i],Z_calc[i])
    return acc

# Importing the excel data through the pandas function read_excel
# The converters argument takes as input the column name of interest and
# apply the function lambda over that pd.Series
file_in = "../../Data/complex impedance data.xlsx"

data = pd.read_excel(file_in, names= ["freq","Z_meas"],
                     converters={"Z_meas" : lambda s: s.replace('i','')}) 

# Transforming data into np.array
freq = np.array(data["freq"])

# Using map method & split methods to separate the real and the complex part of Z_meas
Z_meas_Re = np.array(data["Z_meas"].map( lambda s : s.split("+")[0]), dtype = 'float64')
Z_meas_Im = -np.array(data["Z_meas"].map( lambda s : s.split("+")[1]), dtype = 'float64')



# Initial guess of the parameters
R1 = 90 # Ohm
C1 = 20e-6 # F
R2 = 7e3 # Ohm
C2 = 0.3e-6 # F
# Recasting into a list
guess = [R1,C1,R2,C2]

# Recasting real and imaginary part of the measured impedance into a complex numpy array
Z_meas = Z_meas_Re + Z_meas_Im*1j
Z_meas_mod = np.abs(Z_meas)
Z_meas_phase = np.arctan2(np.imag(Z_meas),np.real(Z_meas))

# Minimizing the total error as a function of the parameters 
res = minimize(total_error,x0 = guess,args =(freq,Z_meas))

# Calculating Z through formula
Z_calc = tot_Z(freq,res.x[0],res.x[1],res.x[2],res.x[3])
Z_calc_mod = np.abs(Z_calc)
Z_calc_phase = np.arctan2(np.imag(Z_calc),np.real(Z_calc))


# Creating visualisation
fig, ax = plt.subplots(2, sharex = True)
# Plotting the bode plot of the impedance
ax[0].semilogx(freq,20*np.log10(Z_calc_mod),"-",linewidth = 3,color = "blue", label = "Calculated")
ax[0].semilogx(freq,20*np.log10(np.abs(Z_meas_mod)),".", markersize=4.4,color = "red" ,label = "Measured")
ax[0].set_ylabel("|Z| [dB]")
ax[0].legend()
ax[1].set_xlabel("Frequency [Hz]")
ax[1].semilogx(freq,rad_to_degree(Z_calc_phase), "-", linewidth = 3, color = "blue",label = "Calculated")
ax[1].semilogx(freq, rad_to_degree(Z_meas_phase),".",markersize=4.4,color = "red", label = "Measured")
ax[1].set_ylabel("Phase [degree]")
ax[1].legend()
plt.show()
