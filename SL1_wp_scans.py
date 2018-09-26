# -*- coding: utf-8 -*-
"""
Created on Mon Dec 12 17:47:38 2016

@author: esposito_v
"""

"""
analysis of the waveplate scans.
"""

import numpy as np
import sys
import matplotlib.pyplot as plt
import pandas as pd
from lmfit import Parameters, models, minimize

sys.path.append("C:/Users/esposito_v/Documents/Python Scripts/FEMTO analysis/")
import fit_function as fitfct
import halo_correction as halocorr

plt.close('all')

plot = 1

datapath = 'C:/Users/esposito_v/Documents/PCMO/FEMTO 201506/data/averages2/'

l_area = 0.0520*0.0620*np.pi/4 #[cm]
flu_factor = 1./(1000*l_area/np.sin(np.deg2rad(10)))

""" WP scans for PCMO x=0.5 """
wp_PCMO5_dl = dict()
files = np.array([1547,1535,1559,1573,1587,1601])
file_name = dict()
delay = np.array([-100,3,10,20,30,40])

bkg = 10.4 # background from rotation scans

popt_dl_u = np.zeros([delay.shape[0],4])
perr_dl_u = np.zeros([delay.shape[0],4])
popt_dl_p = np.zeros([delay.shape[0],4])
perr_dl_p = np.zeros([delay.shape[0],4])

if plot:
    fig = plt.figure(101,figsize=(18,6))
    
fc_fit = []
fc_err = []
gamma_fit = []
gamma_err = []

for ii in range(files.size):
    file_name[ii] = datapath + 'wp_' + str(files[ii]) + '.dat'
    dataii = pd.read_table(file_name[ii], skiprows=[1,2,3,4,5,6,7,8,9,10,11], usecols=[1,2,3,5,6,20,21,23,24])
    dataii['fluence'] = ( 119.9+118.05*np.cos(dataii['Waveplate (rbk)']*0.070255-1.29) ) * flu_factor
    
    """ halo and bkg correction """
    dataii['d1 wl'] = dataii['d1 wl'] - bkg - halocorr.halo_correction(dataii['d1 wol'],dataii['d1 wl'], \
        delay[ii],dataii['d2 wl'])
    dataii['d1 wol'] = dataii['d1 wol'] - bkg - halocorr.halo_correction(dataii['d1 wol'],dataii['d1 wol'], \
        delay[ii],dataii['d2 wl'])
    err_p = np.sqrt(dataii['d1 wl']/10.)
    err_u = np.sqrt(dataii['d1 wol']/10.)

    plt.figure(99)
    plt.subplot(2,3,ii+1)
    plt.plot(dataii['fluence'],halocorr.halo_correction(dataii['d1 wol'],dataii['d1 wl'],delay[ii],dataii['d2 wl']))
    plt.title('dl = %01d ps' % delay[ii])
    
    dataii['lOn_norm'] = dataii['d1 wl']/np.mean(dataii['d1 wl'][-2:])
    dataii['lOff_norm'] = dataii['d1 wol']/np.mean(dataii['d1 wol'][-2:])

    wp_PCMO5_dl['dl'+str( delay[ii] )] = dataii
    
    
#    """ Fit step function to data """
#    """ (i) unpumped """
#    A = 100
#    f0 = 3
#    FWHM = 4
#    offset = 200
#    p0 = np.array([A,f0,FWHM,offset])
#    
#    popt_dl_u[ii,:], perr_dl_u[ii,:]  = fitfct.fit_curvefit(p0, dataii['fluence'], dataii['d1 wol'], fitfct.step_fct, \
#                yerr=err_u, absolute_sigma=True)
#                
#    """ (ii) pumped """
#    A = 180
#    f0 = 3
#    FWHM = 4
#    offset = 200
#    p0 = np.array([A,f0,FWHM,offset])
#    
#    popt_dl_p[ii,:], perr_dl_p[ii,:]  = fitfct.fit_curvefit(p0, dataii['fluence'], dataii['d1 wl'], fitfct.step_fct, \
#                yerr=err_p, absolute_sigma=True)
#    
#    yfit_p = fitfct.step_fct(dataii['fluence'],*popt_dl_p[ii,:])
#    yfit_u = fitfct.step_fct(dataii['fluence'],*popt_dl_u[ii,:])
#    if plot:
#        axe = fig.add_subplot(1,6,ii+1)
#        axe.errorbar(dataii['fluence'],dataii['d1 wl'],err_p, fmt='o')
#        axe.plot( dataii['fluence'], yfit_p)
#        axe.errorbar(dataii['fluence'],dataii['d1 wol'],err_u, fmt='o')
#        axe.plot( dataii['fluence'], yfit_u)
#        axe.plot(dataii['fluence'],dataii['d2 wl'])
#        axe.set_title('dl = %01d ps' % delay[ii])
#        axe.set_ylim(-10,250)
#        plt.xlabel('Fluence (mJ/cm^2)')
#        fig.tight_layout()
#    
#    wp_PCMO5_dl['dl'+str( delay[ii] )]['yfit_p'] = yfit_p
#    wp_PCMO5_dl['dl'+str( delay[ii] )]['yfit_u'] = yfit_u
        
    
    """ fit order param model """
    model = fitfct.order_param
    params = Parameters()
    params.add_many(('fc', 2), ('gamma', 0.5), ('scale',1))
    
    fit_out = minimize(model, params, args=(dataii['fluence'],dataii['lOn_norm']))
    yfit = dataii['lOn_norm']+fit_out.residual
    dataii['yfit'] = yfit
    fc_fit = np.append(fc_fit,fit_out.params['fc'])
    fc_err = np.append(fc_err,fit_out.params['fc'].stderr)
    gamma_fit = np.append(gamma_fit,fit_out.params['gamma'])
    gamma_err = np.append(gamma_err,fit_out.params['gamma'].stderr)
    
    if plot:
        axe = fig.add_subplot(1,6,ii+1)
        axe.errorbar(dataii['fluence'],dataii['lOn_norm'],err_p/100, fmt='o')
        axe.plot( dataii['fluence'], yfit)
        axe.errorbar(dataii['fluence'],dataii['lOff_norm'],err_u/100, fmt='o')
        axe.set_title('dl = %01d ps' % delay[ii])
        axe.set_ylim(-0.1,1.2)
        plt.xlabel('Fluence (mJ/cm^2)')
        fig.tight_layout()
        
    wp_PCMO5_dl['dl'+str( delay[ii] )]['yfit'] = yfit

#%%                
fig = plt.figure()
axe = fig.add_subplot(1,2,1)
axe.errorbar(delay, fc_fit, fc_err, fmt='o')
axe.set_title('f_c')
axe.set_xlim(0,50)
axe = fig.add_subplot(1,2,2)
axe.errorbar(delay, gamma_fit, gamma_err, fmt='o')
axe.set_title('gamma')
axe.set_xlim(0,50)



#%%
fluence = wp_PCMO5_dl['dl30']['fluence']
pumped = wp_PCMO5_dl['dl30']['d1 wl']
unpumped = wp_PCMO5_dl['dl30']['d1 wol']

int_exc = unpumped[0:3].mean()
int0 = unpumped[-3:].mean()

T = 0.94 # transmission coefficient
z0 = 48. #* np.sin(np.deg2rad(60)) # penetration depth multiplied by the sin of the incident angle (taking into account refraction)
d = 40. # sample thickness [nm]
nc = 465. # critical energy density
fc = 2.4 # critical fluence
layers = 40 # number of layers
dlayer = d/layers
layers_abovenc = []
layers_abovefc = []

for flu in fluence:
    f_layer_top = []
    f_layer_bottom = []
    n_layer = []
    for ii in range(layers):
        f_layer_top.append( flu*T*np.exp(-ii*dlayer/z0) )
        f_layer_bottom.append( flu*T*np.exp(-(ii+1)*dlayer/z0) )
        n_layer.append( (f_layer_top[ii]-f_layer_bottom[ii]) / (dlayer*1e-7)/1000 ) # excitation density J/cm^3
    
#    plt.figure(11)
#    plt.plot(n_layer)
#    plt.plot(f_layer_top)
     
    for ii in range(layers):   
        if f_layer_top[ii] < fc:
            threshold_f = ii
            break
        elif (f_layer_top[ii] > fc) & ((ii == layers-1)):
            threshold_f = layers
        
    for ii in range(layers):
        if n_layer[ii] < nc:
            threshold_n = ii
            break
        elif (n_layer[ii] > nc) & ((ii == layers-1)):
            threshold_n = layers
            
    layers_abovefc.append(threshold_f)
    layers_abovenc.append(threshold_n)
    ratio = np.array(layers_abovenc)/layers

unpumped_simu = (1-ratio)*int0 + ratio*int_exc
plt.figure()
plt.errorbar(fluence,unpumped,yerr=np.sqrt(unpumped/10), fmt='o')
plt.errorbar(fluence,pumped,yerr=np.sqrt(pumped/10), fmt='o')
plt.plot(fluence,unpumped_simu)


#%%
#""" Line extraction """
##fluence = np.array(fluence)[::-1]
##pumped = np.array(pumped)[::-1]
#
#threshold = 5.
#line_nb = 1
#split_idx = np.array([0])
#plt.figure()
#plt.plot(fluence, pumped,'o')
#maxdiff = threshold + 1
#maxdiff2 = threshold + 1
#
#model = models.LinearModel()
#params = model.make_params()
#
#fit1 = model.fit(pumped, params, x=fluence)
#y = fit1.best_fit
#
#while maxdiff2 > threshold:
#    splitdata = dict()
#    splitx = dict()
#    fit_results = dict()
#
#    for ii in range(len(split_idx)):
#        if ii == len(split_idx)-1:
#            splitdata[ii] = pumped[split_idx[ii]:]
#            splitx[ii] = fluence[split_idx[ii]:]
#        else:
#            splitdata[ii] = pumped[split_idx[ii]:split_idx[ii+1]]
#            splitx[ii] = fluence[split_idx[ii]:split_idx[ii+1]]
#        
#        fit_results[ii] = model.fit(splitdata[ii], params, x=splitx[ii])
#        plt.plot(splitx[ii],fit_results[ii].best_fit)
#        plt.plot(splitx[ii],fit_results[ii].best_fit)
#        
#        diff = np.abs(fit_results[ii].best_fit - splitdata[ii])
#        maxdiff = np.max(diff)
#        
#        if maxdiff > threshold:
#            line_nb +=1
#            idx = np.where(diff==maxdiff)
#            split_idx = np.append(split_idx,idx)
#            
#        maxdiff2 = 2

model = models.LinearModel()
params = model.make_params()

output1 = model.fit(pumped[-4:], params, x=fluence[-4:])
yfit1 = model.eval(output1.params, x=fluence)

output2 = model.fit(pumped[14:17], params, x=fluence[14:17])
yfit2 = model.eval(output2.params, x=fluence)

output3 = model.fit(pumped[:8], params, x=fluence[:8])
yfit3 = model.eval(output3.params, x=fluence)




plt.figure()
plt.errorbar(fluence,pumped,yerr=np.sqrt(unpumped/10), fmt='o')
plt.plot(fluence,yfit1)
plt.plot(fluence,yfit2)
plt.plot(fluence,yfit3)
plt.ylim([0,180])





























        
        