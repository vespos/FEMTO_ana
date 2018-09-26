# -*- coding: utf-8 -*-
"""
Created on Mon Apr 10 11:51:54 2017

@author: esposito_v
"""

"""
analysis of the time scans. Loads timescans that were corrected with Matlab
"""

import numpy as np
import sys
import matplotlib.pyplot as plt
import pandas as pd
from lmfit import Parameters, Model, models, minimize

sys.path.append("C:/Users/esposito_v/Documents/Python Scripts/FEMTO analysis/")
import fit_function as fitfct

plt.close('all')

datapath = 'C:/Users/esposito_v/Documents/PCMO/FEMTO 201506/data_corr/'

l_area = 0.0520*0.0620 #[cm]
flu_factor = 1./(1000*l_area/np.sin(np.deg2rad(10)))


""" dl scans for PCMO x=0.5 """
wp_PCMO5_dl = dict()
files = np.array([1701,1616,1735,1755,1769])
file_name = dict()
fluence = flu_factor * np.array([20,40,60,80,100])

freq_5 = []
freq_err_5 = []

for ii in range(files.size):
    file_name[ii] = datapath + 'run' + str(files[ii]) + '.txt'
    dataii = pd.read_table(file_name[ii])
    dataii['lOn_norm'] = 1+ (dataii['lOn'] - np.mean(dataii['lOn'][0:6]))/np.mean(dataii['lOn'][0:6])
    
    plt.figure(99)
    plt.plot(dataii['dl'],dataii['lOn_norm'], label=('flu = %0.1f mJ/cm$^2$' % fluence[ii]))
#    plt.errorbar(dataii['dl'],dataii['lOn'],dataii['lOn_err'], label=('flu = %0.1f mJ/cm$^2$' % fluence[ii]))
    plt.legend()
    plt.title('PCMO x=0.5')

    model = fitfct.incoh_and_coh
#    model = fitfct.displacive

    params = Parameters()
    # add with tuples: (NAME VALUE VARY MIN  MAX  EXPR  BRUTE_STEP)
    if model is fitfct.incoh:
        params.add_many(('A_fast', -0.02, True, None, 0), ('tau_fast', 0.1), ('A_slow', -0.05, True, None, 0), ('tau_slow', 5))
    elif model is fitfct.displacive:
        params.add_many(('A', -0.1), ('tau', 4), ('freq', 2.5), ('A_fast',0.2))
    elif model is fitfct.incoh_and_coh:
        params.add_many(('A_fast', -0.02, True, None, 0), ('tau_fast', 0.1), ('A_slow', -0.05, True, None, 0), ('tau_slow', 5), \
                        ('A_ph',0.005), ('freq',2.4), ('tau_ph',1))
    else:
        print('Model not found')
    
    fit_out = minimize(model, params, args=(dataii['dl'],dataii['lOn_norm']-1))
    yfit = dataii['lOn_norm']+fit_out.residual
    plt.figure(20)
    plt.plot(dataii['dl'],dataii['lOn_norm'])
    plt.plot(dataii['dl'],yfit)
    dataii['yfit'] = yfit
    freq_5 = np.append(freq_5,fit_out.params['freq'])
    freq_err_5 = np.append(freq_err_5,fit_out.params['freq'].stderr)
    
    wp_PCMO5_dl['flu'+str( fluence[ii] )] = dataii
















