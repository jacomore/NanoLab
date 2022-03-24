import numpy as np
from scipy.fft import fft,ifft


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
    Vft  = np.empty(int(N/2), dtype = np.complex_)
    Vft = tmp[0:int(N/2)]   # selecting only half of the fourier transform spectrum
    return np.abs(Vft)**2*T/(2*(N/2)**2)

       
def dB(P_in, P_ref = 1):
    """
    Calculates the power spectrum in unit of dB
    ------------------------------------------
    input:
        -P_in:COMPLEX, N-dimensional. Power spectrum of the signal under test
        -P_ref: REAL/COMPLEX, optinal. Reference power that normalizes P_in
    """
    return 10*np.log10(abs(P_in)/P_ref)

def trapezoidal(f,xmin,xmax,N):
    """
    Evaluate the integral of the function f within the interval [xmin,xmax], 
    over a number of points N
    ----------------------------------------------------------------------
    input: 
        -f: function object. 
        -xmin,max. Real. Extremes of the interal [xmin,xmax]
        -N: number of points
    """
    delta_k = (xmax-xmin)/(N-1)
    Trap = 0.
    for k in range(1,N):
        Trap += delta_k*(f[k]+f[k-1])/2.0
    return Trap

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


def rad_to_degree(rad_angle):
    """
    Returns the angle in degree if provided in radiant.
    INPUT: 
    	- rad_angle: REAL. Angle [rad]
    OUTPUT: angle [degree]
    """
    return 180./np.pi*rad_angle

def degree_to_rad(degree_angle):
    """
    Returns the angle in radiants if provided in degree.
    INPUT: 
    	- degree_angle: REAL. Angle [degree]
    OUTPUT: angle [rad]
    """
    return np.pi/180*degree_angle


