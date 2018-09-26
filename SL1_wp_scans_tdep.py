# -*- coding: utf-8 -*-
"""
Created on Fri Apr  7 11:37:53 2017

@author: esposito_v
"""

import numpy as np
import sys
import matplotlib.pyplot as plt
import pandas as pd

sys.path.append("C:/Users/esposito_v/Documents/Python Scripts/FEMTO analysis/")
import fit_function as fitfct
import functions_FEMTO as fem

plt.close('all')

plot = 1

datapath = 'C:/Users/esposito_v/Documents/PCMO/FEMTO 201506/data/averages2/'

l_area = 0.0520*0.0620 #[cm]
flu_factor = 1./(1000*l_area/np.sin(np.deg2rad(10)))

""" WP scans for PCMO x=0.5 """
wp_PCMO5_dl = dict()
files = np.array([1547,1535,1559,1573,1587,1601])
file_name = dict()
delay = np.array([-100,3,10,20,30,40])

bkg = 10.4 # background from rotation scans
""" WP scans for PCMO x=0.5 at dl = 3 ps """
wp_PCMO5_Tdep = dict()
delay = 3
files = np.array([1535,1917,1946,1957])
file_name = dict()
temp = np.array([100,130,170,200])

popt_T_u = np.zeros([temp.shape[0],4])
perr_T_u = np.zeros([temp.shape[0],4])
popt_T_p = np.zeros([temp.shape[0],4])
perr_T_p = np.zeros([temp.shape[0],4])


if plot:
    fig = plt.figure(102,figsize=(18,6))

for ii in range(files.size):
    file_name[ii] = datapath + 'wp_' + str(files[ii]) + '.dat'
    dataii = pd.read_table(file_name[ii], skiprows=[1,2,3,4,5,6,7,8,9,10,11,12], usecols=[1,2,3,5,6,20,21,23,24])
    dataii['fluence'] = ( 119.9+118.05*np.cos(dataii['Waveplate (rbk)']*0.070255-1.29) ) * flu_factor
    dataii['d1 wl'] = dataii['d1 wl'] - bkg - fem.halo_correction(dataii['d1 wol'],dataii['d1 wl'], \
        delay,dataii['d2 wl'])
    dataii['d1 wol'] = dataii['d1 wol'] - bkg - fem.halo_correction(dataii['d1 wol'],dataii['d1 wol'], \
        delay,dataii['d2 wl'])
    err_p = np.sqrt(np.abs(dataii['d1 wl'])/5.)
    err_u = np.sqrt(np.abs(dataii['d1 wol'])/5.)
                
    wp_PCMO5_Tdep['T'+str( temp[ii] )] = dataii
    
    
    """ Fit step function to data """
    try:
        """ (i) unpumped """
        A = 100
        f0 = 3
        FWHM = 4
        offset = 200
        p0 = np.array([A,f0,FWHM,offset])
        
        popt_T_u[ii,:], perr_T_u[ii,:]  = fitfct.fit_curvefit(p0, dataii['fluence'], dataii['d1 wol'], fitfct.step_fct, \
                    yerr=err_u, absolute_sigma=True)
    except RuntimeError:
        popt_T_u[ii,:] = np.nan
                    
    try:
        """ (ii) pumped """
        A = 180
        f0 = 3
        FWHM = 4
        offset = 150
        p0 = np.array([A,f0,FWHM,offset])
        
        popt_T_p[ii,:], perr_T_p[ii,:]  = fitfct.fit_curvefit(p0, dataii['fluence'], dataii['d1 wl'], fitfct.step_fct, \
                    yerr=err_p, absolute_sigma=True)
    except RuntimeError:
        popt_T_p[ii,:] = np.nan
    
    yfit_p = fitfct.step_fct(dataii['fluence'],*popt_T_p[ii,:])
    yfit_u = fitfct.step_fct(dataii['fluence'],*popt_T_u[ii,:])
    if plot:
        axe = fig.add_subplot(1,4,ii+1)
        axe.errorbar(dataii['fluence'],dataii['d1 wl'],err_p, fmt='o')
        axe.plot( dataii['fluence'],yfit_p )
        axe.errorbar(dataii['fluence'],dataii['d1 wol'],err_u, fmt='o')
        axe.plot( dataii['fluence'],yfit_u )
        axe.set_title('T = %01d K' % temp[ii])
        axe.set_ylim(-10,250)
        plt.xlabel('Fluence (mJ/cm^2)')
        fig.tight_layout()

    wp_PCMO5_Tdep['T'+str( temp[ii] )]['yfit_p'] = yfit_p
    wp_PCMO5_Tdep['T'+str( temp[ii] )]['yfit_u'] = yfit_u
 