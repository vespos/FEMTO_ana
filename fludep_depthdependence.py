# -*- coding: utf-8 -*-
"""
Created on Tue Jan 17 15:34:25 2017

@author: esposito_v
"""
2
"""
Fit depth dependent dynamics of the peak
Assuming that either the fluence is below f_c and nothing happens or it is above and then the layer 
will undergo the full transition, I propose to model it by fitting each fluence with a double peak
at the low and high T position, whose relative amplitude only depends on the number of layers
above / below the critical fluence.
"""

import numpy as np
import sys
import matplotlib.pyplot as plt
import glob
import pandas as pd
import copy
from lmfit import minimize, Parameters
from lmfit.models import GaussianModel, VoigtModel
sys.path.append("C:/Users/esposito_v/Documents/Python Scripts/FEMTO analysis/")

""" custom libraries """
import functions_FEMTO as fem
import tools_diffraction as tldiff
import fit_function as fitfct
import fit_depthdep


plt.close('all')
plot = True
bootstrap = False

rNo_all = np.zeros([6,3])

""" low T """
rNo_all[0,:] = np.arange(2912,2915)
rNo_all[1,:] = np.arange(2915,2918)
rNo_all[2,:] = np.arange(2918,2921)
rNo_all[3,:] = np.arange(2921,2924)
rNo_all[4,:] = np.arange(2924,2927)
rNo_all[5,:] = np.arange(2927,2930)


if not 'imgs_all' in locals():
    imgs_all = np.zeros([rNo_all.shape[0],50,195,487])
    uimgs_all = np.zeros([rNo_all.shape[0],50,195,487])
    imgs_diff = np.zeros([rNo_all.shape[0],50,40,44])
        
    sys.path.append('Z:/Data1')
    procList = glob.glob('Z:\Data1\*\proc\proc*.dat')
    procFile = []
    dfList = []
    set = 2
        
    count = 0
    for ii in range(rNo_all.shape[0]):
        for jj in range(rNo_all.shape[1]):
            rNo = rNo_all[ii,jj]
            string = '%d.dat' %rNo
            procFile.append( [s for s in procList if string in s] )
            string = ''.join(procFile[count])
            temp = pd.read_table(string,usecols=[0,1,9,20,39])
            dfList.append(temp)
            imgs_temp, uimgs_temp =  fem.getPilatusImgs(dfList[count], set)
            imgs_all[ii,:,:,:] = imgs_all[ii,:,:,:] + imgs_temp
            uimgs_all[ii,:,:,:] = uimgs_all[ii,:,:,:] + uimgs_temp
            count+=1
    del(imgs_temp)
    del(uimgs_temp)
    peakPix = [155,92] # pixels position of the peak [x,y]
    roi = np.array([[peakPix[0]-22,peakPix[0]+22],[peakPix[1]-20,peakPix[1]+20]]) # [[xmin,xmax][ymin,ymax]]
    imgs_all = imgs_all[:,:,roi[1][0]:roi[1][1], roi[0][0]:roi[0][1]] # line = y, column = x
    uimgs_all = uimgs_all[:,:,roi[1][0]:roi[1][1], roi[0][0]:roi[0][1]] # line = y, column = x

#d = 74 # distance sample - detector [mm]
#pilH = 2.75 # [mm]
#pilV = -47.5 # [mm]
#xdb = 21.6 - 414.*0.172 # [mm]
#ydb = -58.03 - 188.*0.172 # [mm]
#rot_offset = 329. # offset with respect to the UB matrix
#
#E = 7
#wavelength = 12.3984/E
#alpha = 0.5
#N = np.array([-1,-1,2])
#a = np.array([5.46,5.46,7.56])
#aa = np.array([90,90,90.07])
#a0,a1,a2 = tldiff.vectorFromLengthAndAngles(a,aa)
#a,aa,b,ba = tldiff.lengthAndAngles(a0,a1,a2)
#U,B = tldiff.UBmat(a, aa, N)

for ii in range(imgs_all.shape[0]-1):
    imgs_diff[ii,:,:,:] = imgs_all[ii+1,:,:,:] - imgs_all[0,:,:,:]
#    imgs_plot = np.sum(imgs_diff,1)
#    plt.figure()
#    plt.imshow(imgs_plot[ii,:,:])
#    plt.clim(-900,900)
#    line_plot = np.sum(imgs_plot,2)
#    plt.plot(line_plot[ii,:])

ydata = np.sum(np.sum(imgs_all,3),2)
ydata = np.vstack( [np.sum(np.sum(imgs_all,3),2)[0,:], ydata] )
yerr = np.sqrt(np.abs(ydata))
fit_out = dict()
g1_fit = np.zeros(ydata.shape)
g2_fit = np.zeros(ydata.shape)
g3_fit = np.zeros(ydata.shape)
sigma1 = []
sigma2 = []
integrated_int = []
err_int = np.zeros([rNo_all.shape[0]+1,2])

if plot:
    plt.figure(60,figsize=(16,10))

l_area = 0.0520*0.0620*np.pi/4 #[cm]
flu_factor = 1./(1000*l_area/np.sin(np.deg2rad(10)))
fluence = np.array([0,20,40,60,80,100,120]) * flu_factor

for ii in range(ydata.shape[0]):
    print('ii=%d' %ii)
    
    """ data with background (baseline) subtraction """
    ydata[ii,:] = ydata[ii,:] - np.mean( np.append(ydata[ii,0:4],ydata[ii,-4:]) )
    
    """ Fit the difference with a gaussian. One for each axis (rot, H,V) """
#    x01 = 266
#    x02 = 266.28
#    fit_function = lambda x, A1, sigma1, A2, sigma2: fitfct.double_peak(x,A1,x01,sigma1,A2,x02,sigma2)
    
    """ ROTATION """
    xdata = np.array(dfList[ii]['top rotation (rbk)'])
    
    integrated_int.append(np.trapz(ydata[ii,:],x=xdata))

    model = GaussianModel(prefix='g1_') + GaussianModel(prefix='g2_') + GaussianModel(prefix='g3_')
    params = model.make_params()
    params['g1_amplitude'].value = 500
    params['g1_center'].value = 266
    params['g1_sigma'].value = 0.13
    params['g2_amplitude'].value = 500    
    params['g2_center'].value = 266.26
#    params['g2_center'].value = 266.3
    params['g2_sigma'].value = 0.13
    params['g3_amplitude'].value = 85
    params['g3_center'].value = 266.44
    params['g3_sigma'].value = 0.27

    params['g1_center'].vary = False
    params['g2_center'].vary = False
    params['g2_amplitude'].min = 0
    params['g1_sigma'].vary = False
    params['g2_sigma'].vary = False

    params['g3_amplitude'].vary = False
    params['g3_center'].vary = False
    params['g3_sigma'].vary = False
    
    fit_out[ii] = model.fit(ydata[ii,:], params, x=xdata, weights=yerr[ii,:])
    params = copy.deepcopy(fit_out[ii].params)
    params['g2_amplitude'].value = 0
    params['g3_amplitude'].value = 0
    g1_fit[ii,:] = model.eval(params, x=xdata)
    err_int[ii,0] = np.sqrt(np.pi*params['g1_amplitude'].stderr**2*params['g1_sigma'].value**2 \
        + params['g1_amplitude'].value**2*params['g1_sigma'].stderr**2)
    
    params = copy.deepcopy(fit_out[ii].params)
    params['g1_amplitude'].value = 0
    params['g3_amplitude'].value = 0
    g2_fit[ii,:] = model.eval(params, x=xdata)
    err_int[ii,1] = np.sqrt(np.pi*(params['g2_amplitude'].stderr**2*params['g2_sigma'].value**2 \
        + params['g2_amplitude'].value**2*params['g2_sigma'].stderr**2))
    
    params = copy.deepcopy(fit_out[ii].params)
    params['g1_amplitude'].value = 0
    params['g2_amplitude'].value = 0
    g3_fit[ii,:] = model.eval(params, x=xdata)
    
    sigma1.append([fit_out[ii].params['g1_sigma'].value, fit_out[ii].params['g1_sigma'].stderr])
    sigma2.append([fit_out[ii].params['g2_sigma'].value, fit_out[ii].params['g2_sigma'].stderr])

    if plot:
        plt.figure(60)
        plt.subplot(3,3,ii+1)
        plt.title('%.2f mJ/cm^2' % fluence[ii])
        plt.errorbar(xdata,ydata[ii,:],yerr=yerr[ii,:],fmt='o')
        plt.plot(xdata,fit_out[ii].best_fit)
        plt.plot(xdata,g1_fit[ii,:])
        plt.plot(xdata,g2_fit[ii,:])
        plt.plot(xdata,g3_fit[ii,:])

sigma1 = np.array(sigma1)
sigma2 = np.array(sigma2)
        
#%%
""" sample data """
T = 0.94 # transmission coefficient
z0 = 48. #* np.sin(np.deg2rad(60)) # penetration depth multiplied by the sin of the incident angle (taking into account refraction)
d = 40. # sample thickness [nm]
nc = 465. # critical energy density
fc = 2.5*T # critical fluence
layers = 40 # number of layers
dlayer = d/layers
layers_abovenc = []
layers_abovefc = []

flus = np.arange(0,9,0.2)

for flu in flus:
    f_layer_top = []
    f_layer_bottom = []
    n_layer = []
    for ii in range(layers):
        f_layer_top.append( flu*T*np.exp(-ii*dlayer/z0) )
        f_layer_bottom.append( flu*T*np.exp(-(ii+1)*dlayer/z0) )
        n_layer.append( (f_layer_top[ii]-f_layer_bottom[ii]) / (dlayer*1e-7)/1000 ) # excitation density J/cm^3
    
    plt.figure(11)
    plt.plot(n_layer)
    plt.plot(f_layer_top)
     
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
    ratio = np.array(layers_abovefc)/layers

layers_abovenc = np.array(layers_abovenc)/layers
layers_abovefc = np.array(layers_abovefc)/layers

g1_int = np.trapz(g1_fit,x=xdata)
g2_int = np.trapz(g2_fit,x=xdata)
int_ratio = g2_int/g1_int
""" normalimzed to f=0 peak """
ref = g1_int[0]
g1_int = g1_int/ref
g2_int = g2_int/ref
""" normalized to total intensity at each fluence """
g1_int = g1_int/(g1_int+g2_int)
g2_int = g2_int/(g1_int+g2_int)

print(layers_abovefc)
print(layers_abovenc)

plt.figure(70, figsize=(11,5))
plt.subplot(1,2,1)
plt.title('high T phase intensity')
#plt.plot(fluence,int_ratio,'o')
plt.plot(fluence,g2_int,'o')
plt.plot(flus,layers_abovenc)
plt.plot(flus,layers_abovefc)
plt.xlabel('fluence')

plt.subplot(1,2,2)
plt.title('low T phase intensity')
#plt.plot(fluence,1/int_ratio,'o')
plt.plot(fluence,g1_int,'o')
plt.plot(flus,1-layers_abovenc)
plt.plot(flus,1-layers_abovefc)
plt.xlabel('fluence')





""" other method """
z_c = -np.log(fc/flus/T)*z0
volume_fraction = z_c/d
for ii in range(len(volume_fraction)):
    if volume_fraction[ii] < 0:
        volume_fraction[ii] = 0
    elif volume_fraction[ii] > 1:
        volume_fraction[ii] = 1

volume_fraction2 = 1- volume_fraction
    
plt.figure(80)
plt.plot(fluence,g1_int,'o')
plt.plot(fluence,g2_int,'o')
plt.plot(flus,volume_fraction)
plt.plot(flus,1-volume_fraction)
plt.ylim([-0.2,1.2])



    

#%%

plt.figure(100,figsize=(11,5))
plt.subplot(1,2,1)
plt.subplots_adjust(top=0.85)
plt.title('Volume fraction')
plt.errorbar(fluence,g1_int,yerr=err_int[:,0]/ref,fmt='o')
plt.plot(flus,1-layers_abovefc)
plt.errorbar(fluence,g2_int,yerr=err_int[:,1]/ref,fmt='o')
plt.plot(flus,layers_abovefc)
plt.subplot(1,2,2)
plt.errorbar(fluence,sigma1[:,0],yerr=sigma1[:,1],fmt='o')
plt.errorbar(fluence,sigma2[:,0],yerr=sigma2[:,1],fmt='o')


#plt.figure(100,figsize=(11,5))
#plt.suptitle('Rotation')
#plt.subplot(1,2,1)
#plt.subplots_adjust(top=0.85)
#plt.title('Amplitude')
#plt.errorbar(fluence,popt_rot[:,0],yerr=perr_rot[:,0],fmt='o')
#plt.errorbar(fluence,popt_rot[:,2],yerr=perr_rot[:,2],fmt='o')
#plt.subplot(1,2,2)
#plt.title('Sigma')
#plt.errorbar(fluence,popt_rot[:,1],yerr=perr_rot[:,1],fmt='o')
#plt.errorbar(fluence,popt_rot[:,3],yerr=perr_rot[:,3],fmt='o')
#
#plt.figure(101,figsize=(11,5))
#plt.subplot(1,2,1)
#yerr_prop = np.sqrt((perr_rot[:,0]/popt_rot[:,2])**2 + (1/popt_rot[:,2]*perr_rot[:,2])**2)
#plt.errorbar(fluence, popt_rot[:,0]/popt_rot[:,2],yerr=5*yerr_prop, fmt='o')
#plt.subplot(1,2,2)
#yerr_prop = np.sqrt((perr_rot[:,2]/popt_rot[:,0])**2 + (1/popt_rot[:,0]*perr_rot[:,0])**2)
#plt.errorbar(fluence, popt_rot[:,2]/popt_rot[:,0],yerr=5*yerr_prop, fmt='o')























