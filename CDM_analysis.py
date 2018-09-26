# -*- coding: utf-8 -*-
"""
Created on Fri Apr 15 15:26:14 2016

@author: esposito_v
"""


import numpy as np
import os
import sys
import matplotlib.pyplot as plt
import glob
import functions_FEMTO as fem
import tools_imgana as tlimg
import CDM_fct as cdm

procList = glob.glob('Z:\Data1\*\proc\proc*.dat')
procFile = []
dfList = []

rNo = np.array([[2932,2933,2934],
               [2935,2936,2937],
               [2938,2939,2940],
               [2941,2942,2943],
               [2944,2945,2946],
               [2947,2948,2949],
               [2950,2951,2952],
               [2953,2954,2955],
               [2956,2957,2958],
               [2959,2960,2961]])

CDMhkl = np.zeros([rNo.shape[0],3])
CDM_omega = np.zeros([rNo.shape[0]])
omega_max = np.zeros([rNo.shape[0]])
CDM_delta = np.zeros([rNo.shape[0]])
CDM_gamma = np.zeros([rNo.shape[0]])
param = np.zeros([5,rNo.shape[0]])
images = np.zeros([rNo.shape[0],2*10,2*12])

fig, axs = plt.subplots(2,5, figsize=(16, 8), facecolor='w', edgecolor='k')
#fig.subplots_adjust(hspace = 1, wspace=.01)

for ii in range(rNo.shape[0]):
    images[ii,:,:], CDMhkl[ii,:], CDM_omega[ii], CDM_delta[ii], CDM_gamma[ii], procFile, omega_max[ii] = cdm.analyzeCDM_motion(rNo[ii,:], procList)
    plt.figure(1)
    
    param[:,ii] = tlimg.fitgaussian(images[ii,:,:])

    fit0 = tlimg.gaussian(*param[:,ii])
    fit = fit0(*np.indices(images[ii,:,:].shape))
    
    axs = axs.ravel()
    axs[ii].imshow(images[ii,:,:])
    axs[ii].contour(fit)
    axs[ii].set_title(str(rNo[ii,1]))
  
  
t = np.array([-1,3,6,9,12,15,20,35,50,75])
plt.figure(2)
plt.subplot(2,2,1)
plt.plot(t,CDM_omega-266)
plt.ylabel('omega')

plt.subplot(2,2,2)
plt.plot(t,omega_max-266)
plt.ylabel('omega_max')

plt.subplot(2,2,3)
plt.plot(t,CDM_gamma)
plt.ylabel('gamma')

plt.subplot(2,2,4)
plt.plot(t,CDM_delta)
plt.ylabel('delta')





























    