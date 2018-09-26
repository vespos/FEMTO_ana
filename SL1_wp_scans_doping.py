# -*- coding: utf-8 -*-
"""
Created on Mon Apr 10 12:16:30 2017

@author: esposito_v
"""

"""
analysis of the waveplate scans.
"""

import numpy as np
import sys
import matplotlib.pyplot as plt
import pandas as pd
from lmfit import Parameters, Model, models, minimize

sys.path.append("C:/Users/esposito_v/Documents/Python Scripts/FEMTO analysis/")
import fit_function as fitfct
import halo_correction as halocorr

plt.close('all')

plot = 1

datapath = 'C:/Users/esposito_v/Documents/PCMO/FEMTO 201506/data/averages2/'

l_area = 0.0520*0.0620*np.pi/4 #[cm]
flu_factor = 1./(1000*l_area/np.sin(np.deg2rad(10)))

""" doping dependence of WP scans """
wp_doping = dict()

#delay = 3
#files = np.array([1535,2650,2398])

#delay = 10
#files = np.array([1559,940,2412])

delay = 20
files = np.array([1573,950,2427])

file_name = dict()
doping = np.array([0.5,0.4,0.35])

bkg = 10.4 # background from rotation scans

popt_dl_u = np.zeros([doping.shape[0],4])
perr_dl_u = np.zeros([doping.shape[0],4])
popt_dl_p = np.zeros([doping.shape[0],4])
perr_dl_p = np.zeros([doping.shape[0],4])

fc = []
fc_err = []
gamma = []
gamma_err = []

for ii in range(files.size):
    file_name[ii] = datapath + 'wp_' + str(files[ii]) + '.dat'
    if files[ii] == 950:
        dataii = pd.read_table(file_name[ii], skiprows=[1,2,3,4,5,6,7], usecols=[1,2,3,5,6,20,21,23,24], sep=' ')
        dataii.columns = wp_doping['doping0.5'].columns[0:9]
        dataii['fluence'] = ( 125.1+124.1*np.cos(dataii['Waveplate (rbk)']*0.07026-1.28) ) * flu_factor
    else:
        dataii = pd.read_table(file_name[ii], skiprows=[1,2,3,4,5,6,7,8,9,10,11], usecols=[1,2,3,5,6,20,21,23,24])
        dataii['fluence'] = ( 119.9+118.05*np.cos(dataii['Waveplate (rbk)']*0.070255-1.29) ) * flu_factor
    
    
    """ halo and bkg correction """
    dataii['d1 wl'] = dataii['d1 wl'] - bkg - halocorr.halo_correction(dataii['d1 wol'],dataii['d1 wl'], \
        delay,dataii['d2 wl'])
    dataii['d1 wol'] = dataii['d1 wol'] - bkg - halocorr.halo_correction(dataii['d1 wol'],dataii['d1 wol'], \
        delay,dataii['d2 wl'])
    err_p = np.sqrt(dataii['d1 wl']/10.)
    err_u = np.sqrt(dataii['d1 wol']/10.)
    
#    plt.figure(101,figsize=(18,6))
#    plt.subplot(1,3,ii+1)
#    plt.errorbar(dataii['fluence'],dataii['d1 wl'], err_p, fmt='o')
#    plt.errorbar(dataii['fluence'],dataii['d1 wol'], err_u, fmt='o')
#    plt.title('doping = %0.2f' % doping[ii])
    
    plt.figure(102,figsize=(18,6))
    plt.subplot(1,3,ii+1)
    plt.errorbar(dataii['fluence'],dataii['d1 wl']/np.mean(dataii['d1 wol'][-3:]), err_p/np.mean(dataii['d1 wol'][-3:]), fmt='o')
    plt.errorbar(dataii['fluence'],dataii['d1 wol']/np.mean(dataii['d1 wol'][-3:]), err_u/np.mean(dataii['d1 wol'][-3:]), fmt='o')
    plt.title('doping = %0.2f' % doping[ii])
    
    dataii['lOn_norm'] = dataii['d1 wl']/np.mean(dataii['d1 wl'][-2:])

    """ fit order param model """
    model = fitfct.order_param
    params = Parameters()
    params.add_many(('fc', 2), ('gamma', 0.5), ('scale',1))
    
    fit_out = minimize(model, params, args=(dataii['fluence'],dataii['lOn_norm']))
    yfit = dataii['lOn_norm']+fit_out.residual
    plt.figure(21)
    plt.plot(dataii['fluence'],dataii['lOn_norm'],'o')
    plt.plot(dataii['fluence'],yfit)
    dataii['yfit'] = yfit
    fc = np.append(fc,fit_out.params['fc'])
    fc_err = np.append(fc_err,fit_out.params['fc'].stderr)
    gamma = np.append(gamma,fit_out.params['gamma'])
    gamma_err = np.append(gamma_err,fit_out.params['gamma'].stderr)

    wp_doping['doping'+str( doping[ii] )] = dataii

plt.figure(103)
plt.errorbar(wp_doping['doping0.5']['fluence'],wp_doping['doping0.5']['d1 wl']/np.mean(wp_doping['doping0.5']['d1 wl'][-2:]), \
             np.sqrt(wp_doping['doping0.5']['d1 wl']/10.)/np.mean(wp_doping['doping0.5']['d1 wl'][-3:]), fmt='o', label='x=0.5')
plt.errorbar(wp_doping['doping0.4']['fluence'],wp_doping['doping0.4']['d1 wl']/np.mean(wp_doping['doping0.4']['d1 wl'][-2:]), \
             np.sqrt(wp_doping['doping0.4']['d1 wl']/10.)/np.mean(wp_doping['doping0.4']['d1 wl'][-3:]), fmt='o', label='x=0.4')
plt.errorbar(wp_doping['doping0.35']['fluence'],wp_doping['doping0.35']['d1 wl']/np.mean(wp_doping['doping0.35']['d1 wl'][-2:]), \
             np.sqrt(wp_doping['doping0.35']['d1 wl']/10.)/np.mean(wp_doping['doping0.35']['d1 wl'][-3:]), fmt='o', label='x=0.35')
plt.legend()




#%%
#plt.close(200)
#plt.close(201)
#plt.close(202)
#
#fluence = wp_doping['doping0.35']['fluence']
#pumped = wp_doping['doping0.35']['d1 wl']
#unpumped = wp_doping['doping0.35']['d1 wol']
#
#model = models.LinearModel()
#params = model.make_params()
#
#output1 = model.fit(pumped[-6:], params, x=fluence[-6:])
#yfit1 = model.eval(output1.params, x=fluence)
#
#output2 = model.fit(pumped[15:22], params, x=fluence[15:22])
#yfit2 = model.eval(output2.params, x=fluence)
#
#output3 = model.fit(pumped[:8], params, x=fluence[:8])
#yfit3 = model.eval(output3.params, x=fluence)
#
#plt.figure(200)
#plt.errorbar(fluence,pumped,yerr=np.sqrt(unpumped/10), fmt='o')
#plt.plot(fluence,yfit1)
#plt.plot(fluence,yfit2)
#plt.plot(fluence,yfit3)
#plt.ylim([-10,150])
#plt.title('x=0.35')
#
#
#
#
#
#
#
#fluence = wp_doping['doping0.4']['fluence']
#pumped = wp_doping['doping0.4']['d1 wl']
#unpumped = wp_doping['doping0.4']['d1 wol']
#
#model = models.LinearModel()
#params = model.make_params()
#
#output1 = model.fit(pumped[-4:], params, x=fluence[-4:])
#yfit1 = model.eval(output1.params, x=fluence)
#
#output2 = model.fit(pumped[14:17], params, x=fluence[14:17])
#yfit2 = model.eval(output2.params, x=fluence)
#
#output3 = model.fit(pumped[:8], params, x=fluence[:8])
#yfit3 = model.eval(output3.params, x=fluence)
#
#plt.figure(201)
#plt.errorbar(fluence,pumped,yerr=np.sqrt(unpumped/10), fmt='o')
#plt.plot(fluence,yfit1)
#plt.plot(fluence,yfit2)
#plt.plot(fluence,yfit3)
#plt.ylim([-10,100])
#plt.title('x=0.4')
#
#
#
#
#
#
#fluence = wp_doping['doping0.5']['fluence']
#pumped = wp_doping['doping0.5']['d1 wl']
#unpumped = wp_doping['doping0.5']['d1 wol']
#
#model = models.LinearModel()
#params = model.make_params()
#
#output1 = model.fit(pumped[-5:], params, x=fluence[-5:])
#yfit1 = model.eval(output1.params, x=fluence)
#
#output2 = model.fit(pumped[14:17], params, x=fluence[14:17])
#yfit2 = model.eval(output2.params, x=fluence)
#
#output3 = model.fit(pumped[:8], params, x=fluence[:8])
#yfit3 = model.eval(output3.params, x=fluence)
#
#plt.figure(202)
#plt.errorbar(fluence,pumped,yerr=np.sqrt(unpumped/10), fmt='o')
#plt.plot(fluence,yfit1)
#plt.plot(fluence,yfit2)
#plt.plot(fluence,yfit3)
#plt.ylim([-10,200])
#plt.title('x=0.5')
#


















