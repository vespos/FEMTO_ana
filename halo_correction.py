# -*- coding: utf-8 -*-
"""
Created on Fri Feb  3 19:37:50 2017

@author: esposito_v
"""

""" test halo correction """


import numpy as np
import matplotlib.pyplot as plt

def gaussian(x,A,x0,sigma):
    return  A*np.exp(-(x-x0)**2/2/sigma**2)





def halo_correction(int0, int, dl, halo_signal):
    """
    Halo correction for slicing data
    inputs:
        int0: intensity of the signal before t0
        int: intensity of the signal of the data point considered
        dl: dl at which the data point is taken. This is important to know the pumped fraction of the halo
        halo_signal: halo measurement ot the data point
    
    The code works in two steps:
        i) the ratio of the pumped and unpumped halo is calculated
        ii) the correction is composed of two parts; (1-ratio) of the halo (unpumped) and ratio*int/int0 of the halo
            pumped, assuming the same drop than in the signal).
    Ideally one would want to iterate between these steps, as the drop in signal is biased by the halo, but this is not done,
    and probably does not change much to the result.
    """
    
    FWHM = 40
    sigma = FWHM /2/np.sqrt(2*np.log(2))
    
    t0 = dl
    t = np.arange(-200,300,0.2) # [ps] t=0 at t[1000]
    halo = gaussian(t,1,t0,sigma)
    int0_halo = np.trapz(halo,x=t)
    int_halo = np.trapz(halo[1000:], x=t[1000:])
    ratio = int_halo/int0_halo
    
    correction = (1-ratio)*halo_signal + ratio*int/int0*halo_signal
    
    return correction