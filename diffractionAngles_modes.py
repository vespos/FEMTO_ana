# -*- coding: utf-8 -*-
"""
Created on Fri Jul 08 11:54:32 2016

@author: VEsp
"""

import numpy as np

def diffAngles_mode1(H,K,alpha):
    """
    mode 1: horizontal geometry, fixed incident angle
    inputs:
        H: orthonormalized hkl
        K: wavevector
        alpha: incident angle
        
    outputs:
        delta, gamma, omega: diffractometer angles
        angOut: outgoing angle
        
        
    based on Simon's code
    """
    
    alpha=np.deg2rad(alpha)

#    h = H[0,:]/K
#    k = H[1,:]/K
#    l = H[2,:]/K
    h = H[0]/K
    k = H[1]/K
    l = H[2]/K
    
    Y = -.5*(h**2+k**2+l**2)
    Z = (l+np.sin(alpha)*Y)/np.cos(alpha)
    X = np.sqrt( h**2+k**2 - (np.cos(alpha)*Y+np.sin(alpha)*Z)**2 ) # Change sign here to do gamma positive (and also in line 53)
    W = np.cos(alpha)*Y+np.sin(alpha)*Z
    
    gamma = -np.arctan2(-X,Y+1)
    delta = np.arctan2(Z*np.sin(gamma),X)
    omega = np.arctan2(h*W-k*X,h*X+k*W)    
    angOut = np.arcsin(l-np.sin(alpha))
    
    return delta, gamma, omega, angOut