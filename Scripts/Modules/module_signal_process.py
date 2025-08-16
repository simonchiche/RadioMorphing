# -*- coding: utf-8 -*-
"""
Created on Thu May 21 22:44:06 2020

@author: Simon
"""
import numpy as np
from scipy.signal import butter, lfilter, hilbert
from scipy.interpolate import UnivariateSpline
from scipy.fftpack import fft
import matplotlib.pyplot as plt
import sys

def add_noise(traces, vrms):
    """Add normal random noise on traces
    Parameters:
    -----------
        efield: numpy array
            efield trace
        vrms: float
            noise rms, default loaded from config file
    Returns:
    ----------
        numpy array
        noisy voltages (time in ns)
    """

    noisy_traces = np.copy(traces)
    noisy_traces[:, 1:] = traces[:, 1:] + \
        np.random.normal(0, vrms, size=np.shape(traces[:, 1:]))

    return noisy_traces

def add_noise_traces(Traces, n, time_sample):
    
    traces_with_noise = np.zeros([time_sample, 4*n])
    
    # for each of the "n" antennas we add a gaussian noise 
    for i in range(n):
        traces = np.array([Traces[:,i], Traces[:,i + n], Traces[:, i + 2*n], Traces[:, i +3*n]]).T
        traces_with_noise[:,i] = traces[:,0] 
        traces_with_noise[:,i + n] = add_noise(traces, 20)[:,1]
        traces_with_noise[:, i + 2*n] = add_noise(traces, 20)[:,2]
        traces_with_noise[:, i +3*n] = add_noise(traces, 20)[:,3]

    return traces_with_noise

def Digitization(traces, TSAMPLING=2):
    """Digitize the voltages at an specific sampling -- v2
    Parameters:
    -----------
        voltages: numpy array
            voltage trace
        samplingrate: float
            sampling rate in ns, default loaded from config file
    Returns:
    ----------
        numpy array:
        newly sampled trace
    """
    traces = traces.T
    tstep = np.mean(np.diff(traces[0]))  
    ratio = int(round(TSAMPLING/tstep))
    SAMPLESIZE = int(len(traces[0])/ratio)
    ex = np.zeros(SAMPLESIZE)
    ey = np.zeros(SAMPLESIZE)
    ez = np.zeros(SAMPLESIZE)
    tf = np.zeros(SAMPLESIZE)
    ind = np.arange(0, SAMPLESIZE)*ratio

    if len(ind) > SAMPLESIZE:
        ind = ind[0:TSAMPLING]
    ex[0:len(ind)] = traces[1, ind]
    ey[0:len(ind)] = traces[2, ind]
    ez[0:len(ind)] = traces[3, ind]
    tf[0:len(ind)] = traces[0, ind]
    for k in range(len(ind), SAMPLESIZE):
        tf[k] = tf[k-1]+TSAMPLING

    return np.array([tf, ex, ey, ez]).T

def digitalize_traces(Traces, n, time_sample, TSAMPLING):

    # function that samples the traces 
    
    tstep = np.mean(np.diff(Traces[:,0])) # temporal step of the input traces  
    ratio = int(round(TSAMPLING/tstep)) # ratio between the initial and the expected temporal step
    traces_sampled = np.zeros([int(time_sample/ratio), 4*n])
    
    # for each of the "n" antennas we apply the sampling
    for i in range(n):
        traces = np.array([Traces[:,i], Traces[:,i + n], Traces[:, i + 2*n], Traces[:, i +3*n]]).T
        traces_sampled[:,i] = Digitization(traces, TSAMPLING)[:,0]
        traces_sampled[:,i + n] = Digitization(traces, TSAMPLING)[:,1]
        traces_sampled[:, i + 2*n] = Digitization(traces, TSAMPLING)[:,2]
        traces_sampled[:, i +3*n] = Digitization(traces, TSAMPLING)[:,3]
    time_sample = len(traces_sampled[:,0])
        
    return (traces_sampled, time_sample)

def _butter_bandpass_filter(data, lowcut, highcut, fs):
    """subfunction of filters
    """

    b, a = butter(5, [lowcut / (0.5 * fs), highcut / (0.5 * fs)],
                  btype='band')  # (order, [low, high], btype)

    return lfilter(b, a, data)


def filters(traces, FREQMIN=5.e6, FREQMAX=50.e6):
    """ Filter signal e(t) in given bandwidth
    Parameters
    ----------
    traces: numpy array
        The array of time (s) + efield traces (muV/m) vectors to be filtered
    FREQMIN: float
        The minimal frequency of the bandpass filter (Hz)
    FREQMAX: float
        The maximal frequency of the bandpass filter (Hz)
    Returns
    -------
        numpy array
            time in ns, Voltages (x,y,z)
    Raises
    ------
    Notes
    -----
    At present Butterworth filter only is implemented
    Examples
    ATTENTION: output traces inversed now
    --------
    ```
    >>> from signal_treatment import _butter_bandpass_filter
    ```
    """
    #print(np.shape(traces))
    t = traces.T[0]
    t *= 1e-9  # from ns to s
    e = np.array(traces.T[1:, :])  # Raw signal
    #sys.exit()
    # fs = 1 / np.mean(np.diff(t))  # Compute frequency step
    fs = round(1 / np.mean(np.diff(t)))  # Compute frequency step
    #print("Trace sampling frequency: ",fs/1e6,"MHz")
    nCh = np.shape(e.T)[1]
    #vout = np.zeros(shape=(len(t), nCh))
    res = t
    for i in range(nCh):
        ei = e[i, :]
        #vout[:, i] = _butter_bandpass_filter(ei, FREQMIN, FREQMAX, fs)
        res = np.append(res, _butter_bandpass_filter(ei, FREQMIN, FREQMAX, fs))

    res = np.reshape(res, (nCh+1, len(t)))  # Put it back inright format
    res[0] *= 1e9  # s to ns

    return res.T

def filter_traces(Traces, n, time_sample):
   
    Traces_filtered = np.zeros([time_sample, 4*n])
    for i in range(n):
        traces = np.array([Traces[:,i], Traces[:,i + n], Traces[:, i + 2*n], Traces[:, i +3*n]]).T
        res = filters(traces, FREQMIN=50.e6, FREQMAX=200.e6)
        Traces_filtered[:,i] = res[:,0]
        Traces_filtered[:,i + n] = res[:,1]
        Traces_filtered[:,i + 2*n] = res[:,2]
        Traces_filtered[:,i + 3*n] = res[:,3]
    return Traces_filtered


def filter_single_trace(Time, Efield, n, time_sample):
    
    traces = np.array([Time, Efield]).T
    res = filters(traces, FREQMIN=50.e6, FREQMAX=200.e6)
        
    return res[:,1]