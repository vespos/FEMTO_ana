# -*- coding: utf-8 -*-
"""
Created on Tue Jan 17 15:57:11 2017

@author: esposito_v
"""

"""
fit function for double peak depth dependence
"""

import numpy as np

def doublepeak_depth(x,fluence,A,sigma):
    
    """ sample data """
    T = 0.9 # transmission coefficient
    z0 = np.sin(np.deg2rad(60))*60. # penetration depth multiplied by the sin of the incident angle
    d = 40. # sample thickness [nm]
    nc = 500. # critical energy density
    layers = 40 # number of layers
    dlayer = d/layers
    
    A = A/layers
    
    int = np.zeros(x.shape[0])
    
    for ii in range(layers):
        f_layer_top = fluence*T*np.exp(-ii*dlayer/z0)
        f_layer_bottom = fluence*T*np.exp(-(ii+1)*dlayer/z0)
        n_layer = (f_layer_top-f_layer_bottom) / (dlayer*1e-7)/1000 # excitation density J/cm^3
        
        if n_layer < nc:
            peak_pos = 266
        else:
            peak_pos = 266.27
        
        int_layer = A*np.exp(-(x-peak_pos)**2/2/sigma**2)
        int = int+int_layer
    
#    int = int + 150*np.exp(-(x-266.27)**2/2/0.4**2)
    return int
    
    
    
    
def double_peak(fluence, x, g1, g2):
    """ sample data """
    T = 0.9 # transmission coefficient
    z0 = np.sin(np.deg2rad(60))*60. # penetration depth multiplied by the sin of the incident angle
    d = 40. # sample thickness [nm]
    nc = 350. # critical energy density
    layers = 40 # number of layers
    dlayer = d/layers
    
    for ii in range(layers):
        f_layer_top = fluence*T*np.exp(-ii*dlayer/z0)
        f_layer_bottom = fluence*T*np.exp(-(ii+1)*dlayer/z0)
        n_layer = (f_layer_top-f_layer_bottom) / (dlayer*1e-7)/1000 # excitation density J/cm^3
        
        if n_layer > nc:
            break
    
    layers_abovenc = ii
    ratio = ii/layers
    
    g1_int = np.trapz(g1,x=x)
    g2_int = np.trapz(g2,x=x)
    
    
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        