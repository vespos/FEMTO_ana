# -*- coding: utf-8 -*-
"""
Created on Tue Nov 15 11:25:12 2016

@author: esposito_v
"""

import numpy as np
import sys
import matplotlib.pyplot as plt
import pandas as pd

plt.close('all')

plot = 1

datapath = 'C:/Users/esposito_v/Documents/PCMO/FEMTO 201506/data_corr/'

l_area = 0.0520*0.0620 #[cm]
flu_factor = 1./(1000*l_area/np.sin(np.deg2rad(10)))


""" PCMO 0.5 """
PCMO05 = dict()
files = np.array([1701,1616,1735,1769])
fluence = flu_factor * np.array([20,40,60,80,100])

for ii in range(files.size):
    file_name = datapath + 'run' + str(files[ii]) + '.txt'    
    temp = pd.read_table(file_name, usecols=[0,1,2,3,4])
    PCMO05['flu'+str( np.round(fluence[ii]) )] = temp
    
    if plot:
        fig = plt.figure(1000, figsize=(18,7))
        plt.subplot(1,3,1)
        leg = '%0.2f mJ/cm^2' % fluence[ii]
        plt.plot(temp.dl, temp.lOn/np.mean(temp.lOn[0:4]), label = leg)
        plt.xlabel('Time [ps]')
        plt.ylabel('Normalized diffracted intensity')
        plt.title('PCMO x=0.5')   
        plt.ylim([-0.05,1.3])
        plt.legend()
    



""" PCMO 0.4 """
PCMO04 = dict()
files = np.array([2666,2688,2709,2723])
fluence = flu_factor * np.array([25,40,60,100])

for ii in range(files.size):
    file_name = datapath + 'run' + str(files[ii]) + '.txt'    
    temp = pd.read_table(file_name, usecols=[0,1,2,3,4])
    PCMO04['flu'+str( np.round(fluence[ii]) )] = temp
    
    if plot:
        plt.figure(1000)
        plt.subplot(1,3,2)
        leg = '%0.2f mJ/cm^2' % fluence[ii]
        plt.plot(temp.dl, temp.lOn/np.mean(temp.lOn[0:4]), label = leg)
        plt.xlabel('Time [ps]')
        plt.ylabel('Normalized diffracted intensity')
        plt.title('PCMO x=0.4') 
        plt.ylim([-0.05,1.3])
        plt.legend()







""" PCMO 0.35 """
PCMO35 = dict()
files = np.array([2555,2571,2511])
fluence = flu_factor * np.array([50,75,120])

for ii in range(files.size):
    file_name = datapath + 'run' + str(files[ii]) + '.txt'    
    temp = pd.read_table(file_name, usecols=[0,1,2,3,4])
    PCMO35['flu'+str( np.round(fluence[ii]) )] = temp
    
    if plot:
        plt.figure(1000)
        plt.subplot(1,3,3)
        leg = '%0.2f mJ/cm^2' % fluence[ii]
        plt.plot(temp.dl, temp.lOn/np.mean(temp.lOn[0:4]), label = leg)
        plt.xlabel('Time [ps]')
        plt.ylabel('Normalized diffracted intensity')
        plt.title('PCMO x=0.35') 
        plt.ylim([-0.05,1.3])
        plt.legend()
        
plt.subplots_adjust(left=0.05, bottom=None, right=0.95, top=None, wspace=0.25, hspace=None)









































