# -*- coding: utf-8 -*-
"""
Created on Tue Nov 15 14:57:25 2016

@author: esposito_v
"""

import numpy as np
import sys
import matplotlib.pyplot as plt
import pandas as pd
from lmfit import minimize, Parameters
from lmfit.models import GaussianModel, VoigtModel, LorentzianModel

sys.path.append("C:/Users/esposito_v/Documents/Python Scripts/FEMTO analysis/")
import functions_FEMTO as fem

plt.close('all')

plot = 1

datapath = 'C:/Users/esposito_v/Documents/PCMO/FEMTO 201506/data/averages2/'

l_area = 0.0520*0.0620*np.pi/4 #[cm]
flu_factor = 1./(1000*l_area/np.sin(np.deg2rad(10)))


""" rot scans at 50mW for different delays, x=0.5 """
rot_50mW = dict()
files = np.array([1779,1790,1797,1803,1809])
delay = np.array([3,10,20,30,40])
fluence = flu_factor*50
file_name = dict()
int = np.zeros(6)

bkg = np.zeros(files.size)

if plot:
    fig = plt.figure(101,figsize=(18,4))

for ii in range(files.size):
    file_name[ii] = datapath + 'rot_' + str(files[ii]) + '.dat'
    if ii==0:
        dataii = pd.read_table(file_name[ii], skiprows=[1,2,3,4,5,6,7,8], usecols=[1,2,3,5,6,20,21,23,24])
    else:
        dataii = pd.read_table(file_name[ii], skiprows=[1,2,3,4,5,6], usecols=[1,2,3,5,6,20,21,23,24])
                
    rot_50mW['dl'+str( delay[ii] )] = dataii
    
    """ bkg and halo correction """ 
    bkg = np.mean( np.array(dataii['d1 wl'][0:3]) + np.array(dataii['d1 wl'][-3:]) )/2
    dataii['d1 wl'] = dataii['d1 wl'] - bkg - fem.halo_correction(dataii['d1 wol'],dataii['d1 wl'], \
        delay[ii],dataii['d2 wl'])
    bkg = np.mean( np.array(dataii['d1 wol'][0:3]) + np.array(dataii['d1 wol'][-3:]) )/2
    dataii['d1 wol'] = dataii['d1 wol'] - bkg - fem.halo_correction(dataii['d1 wol'],dataii['d1 wol'], \
        delay[ii],dataii['d2 wl'])
    
    """ integration """
    int[ii] = np.trapz(dataii['d1 wl']/np.max(dataii['d1 wol']), x=dataii['top rotation (rbk)'])
    
    plt.figure(10, figsize=(14,7))
    leg = 'delay = %01d ps' % delay[ii]
    plt.subplot(1,2,1)
    plt.plot(dataii['top rotation (rbk)'], dataii['d1 wl']/np.max(dataii['d1 wol']), lw=2, label=leg )
#    plt.subplot(1,2,2)
#    plt.plot(dataii['top rotation (rbk)'], dataii['d1 wl']-dataii['d1 wol'], lw=2, label=leg )
    plt.legend()
    plt.xlabel('Rotation [deg]')
    plt.ylabel('Intensity')
    plt.title('Rotation scans at different delays (f = 2.7 mJ/cm^2)')
    
    if plot:
        axe = fig.add_subplot(1,6,ii+1)
        axe.plot(dataii['top rotation (rbk)'],dataii['d1 wl'])
        axe.plot(dataii['top rotation (rbk)'],dataii['d1 wol'])
        axe.set_title('f = %01d ps' % delay[ii])
        axe.set_ylim(-10,175)
        fig.tight_layout()


rot_50mW['delay'] = np.append(delay,0)
int[5] = np.trapz(rot_50mW['dl10']['d1 wol']/np.max(rot_50mW['dl10']['d1 wol']), x=rot_50mW['dl3']['top rotation (rbk)'])
rot_50mW['total_int'] = int

plt.figure(10)
plt.subplot(1,2,2)
plt.plot(rot_50mW['delay'],rot_50mW['total_int'], 'o')
plt.xlim([-2,42])
plt.xlabel('Time [ps]')
plt.ylabel('Intgrated intensity')
plt.title('Integrated intensity')

plt.subplots_adjust(left=0.05, bottom=None, right=0.95, top=None, wspace=0.25, hspace=None)












#%%



""" rot scans at 3ps for different fluence, x=0.5 """
rot_3ps = dict()
files = np.array([1963,1969,1975,1982,1988,1995])
fluence = flu_factor * np.array([10,30,40,50,100,120])
delay = 3
file_name = dict()
int = np.zeros(7)
int2 = np.zeros(7)
intu = np.zeros(7)
intu2 = np.zeros(7)

fit_out = dict()
ampl = []
center = []
sigma = []

if plot:
    fig = plt.figure(100,figsize=(18,4))

for ii in range(files.size):
    file_name[ii] = datapath + 'rot_' + str(files[ii]) + '.dat'
    dataii = pd.read_table(file_name[ii], skiprows=[1,2,3,4,5,6], usecols=[1,2,3,5,6,20,21,23,24])
                
    rot_3ps['flu'+str( np.round(fluence[ii]) )] = dataii
    
    
    """ bkg and halo correction """ 
    bkg = np.mean( np.array(dataii['d1 wl'][0:3]) + np.array(dataii['d1 wl'][-3:]) )/2
    dataii['d1 wl'] = dataii['d1 wl'] - bkg - fem.halo_correction(dataii['d1 wol'],dataii['d1 wl'], \
        delay,dataii['d2 wl'])
    bkg = np.mean( np.array(dataii['d1 wol'][0:3]) + np.array(dataii['d1 wol'][-3:]) )/2
    dataii['d1 wol'] = dataii['d1 wol'] - bkg - fem.halo_correction(dataii['d1 wol'],dataii['d1 wol'], \
        delay,dataii['d2 wl'])
    
    """ Fit """
    model = GaussianModel()
    params = model.make_params()
    params['amplitude'].value = 150
    params['center'].value = -95
    params['sigma'].value = 1
    fit_out[ii] = model.fit(dataii['d1 wol'], params, x=dataii['top rotation (rbk)'], weights=np.sqrt(np.abs(dataii['d1 wol']/5)))
    dataii['fit unpumped'] = model.eval(fit_out[ii].params, x=dataii['top rotation (rbk)'])
    
    ampl.append([fit_out[ii].params['amplitude'].value, fit_out[ii].params['amplitude'].stderr])
    center.append([fit_out[ii].params['center'].value, fit_out[ii].params['center'].stderr])
    sigma.append([fit_out[ii].params['sigma'].value, fit_out[ii].params['sigma'].stderr])
    
    
    """ integration """
    norm = np.trapz(rot_3ps['flu1.0']['d1 wol'],x=dataii['top rotation (rbk)'])
    int[ii] = np.trapz(dataii['d1 wl']/norm, x=dataii['top rotation (rbk)'])
    int2[ii] = np.trapz(dataii['d1 wl'], x=dataii['top rotation (rbk)'])
    intu[ii] = np.trapz(dataii['d1 wol']/norm, x=dataii['top rotation (rbk)'])
    intu2[ii] = np.trapz(dataii['d1 wol'], x=dataii['top rotation (rbk)'])
    
    plt.figure(13, figsize=(14,7))
    leg = 'f = %0.2f mJ/cm^2' % fluence[ii]
    plt.subplot(1,2,1)
    plt.plot(dataii['top rotation (rbk)'], dataii['d1 wl']/np.max(rot_3ps['flu1.0']['d1 wol']), lw=2, label=leg )
#    plt.subplot(1,2,2)
#    plt.plot(dataii['top rotation (rbk)'], dataii['d1 wl']-dataii['d1 wol'], lw=2, label=leg )
    plt.legend()
    plt.xlabel('Rotation [deg]')
    plt.ylabel('Intensity')
    plt.title('Rotation scans at different fluence (dl = 3ps) \n normalized to the unpumped of the low fluence')
    plt.ylim([-0.1,1.1])
    
    plt.figure(12, figsize=(14,7))
    plt.subplot(1,2,1)
    plt.plot(dataii['top rotation (rbk)'], dataii['d1 wl']/np.max(dataii['d1 wol']), lw=2, label=leg )
    plt.title('Rotation scans at different fluence (dl = 3ps)')
    plt.ylim([-0.1,1.1])
    
    if plot:
        axe = fig.add_subplot(1,7,ii+1)
        axe.plot(dataii['top rotation (rbk)'],dataii['d1 wl'])
        axe.plot(dataii['top rotation (rbk)'],dataii['d1 wol'], 'o')
        axe.plot(dataii['top rotation (rbk)'],fit_out[ii].best_fit)
        axe.set_title('f = %0.1f mJ/cm^2' % fluence[ii])
        axe.set_ylim(-10,240)
        fig.tight_layout()
        
ampl = np.array(ampl)
center = np.array(center)
sigma = np.array(sigma)

rot_3ps['fluence'] = np.append(fluence,0)
int[6] = np.trapz(rot_3ps['flu1.0']['d1 wol']/norm, x=rot_3ps['flu1.0']['top rotation (rbk)'])
int2[6] = np.trapz(rot_3ps['flu1.0']['d1 wol'], x=rot_3ps['flu1.0']['top rotation (rbk)'])
intu[6] = np.trapz(rot_3ps['flu1.0']['d1 wol']/norm, x=rot_3ps['flu1.0']['top rotation (rbk)'])
intu2[6] = np.trapz(rot_3ps['flu1.0']['d1 wol'], x=rot_3ps['flu1.0']['top rotation (rbk)'])
rot_3ps['total_int'] = int
rot_3ps['total_int2'] = int2
rot_3ps['total_intu'] = intu
rot_3ps['total_intu2'] = intu2

plt.figure(13)
plt.subplot(1,2,2)
plt.plot(rot_3ps['fluence'],rot_3ps['total_int'], 'o')
plt.plot(rot_3ps['fluence'],rot_3ps['total_intu'], 'o')
plt.xlim([-2,8.5])
plt.xlabel('Fluence [mJ/cm^2]')
plt.ylabel('Integrated intensity')
plt.title('Integrated intensity normalized \n to the unpumped of the low fluence')

plt.figure(12)
plt.subplot(1,2,2)
plt.plot(rot_3ps['fluence'],rot_3ps['total_int2'], 'o')
plt.plot(rot_3ps['fluence'],rot_3ps['total_intu2'], 'o')
plt.xlim([-2,8.5])
plt.xlabel('Fluence [mJ/cm^2]')
plt.ylabel('Intgrated intensity')
plt.title('Integrated intensity')

#plt.subplots_adjust(left=0.05, bottom=None, right=0.95, top=None, wspace=0.25, hspace=None)

#%%
plt.figure()
plt.plot(rot_3ps['flu1.0']['top rotation (rbk)'],rot_3ps['flu1.0']['d1 wol'],label='unpumped low flu')
plt.plot(rot_3ps['flu1.0']['top rotation (rbk)'],rot_3ps['flu1.0']['d1 wl'],label='pumped low flu')

plt.plot(rot_3ps['flu7.0']['top rotation (rbk)'],rot_3ps['flu7.0']['d1 wol'],label='unpumped high flu')
plt.plot(rot_3ps['flu7.0']['top rotation (rbk)'],rot_3ps['flu7.0']['d1 wl'],label='pumped low flu')
plt.legend()

plt.figure(figsize=(15,7))
plt.subplot(1,3,1)
plt.title('amplitude')
plt.errorbar(fluence,ampl[:,0],ampl[:,1],fmt='o')
plt.subplot(1,3,2)
plt.title('rot')
plt.errorbar(fluence,center[:,0],center[:,1],fmt='o')
plt.subplot(1,3,3)
plt.title('sigma')
plt.errorbar(fluence,sigma[:,0],sigma[:,1],fmt='o')



#%%
#string = 'dl'
#keys = [s for s in rot_50mW.keys() if string in s]
#
#fig = plt.figure()
#axe = fig.add_subplot(111)
#for ii in range(len(keys)):
#    axe.plot( rot_50mW[keys[ii]]['top rotation (rbk)'], rot_50mW[keys[ii]]['d2 wol'] )




























