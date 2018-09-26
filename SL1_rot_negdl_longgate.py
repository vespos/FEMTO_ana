# -*- coding: utf-8 -*-
"""
Created on Fri Jun  9 11:23:41 2017

@author: esposito_v
"""

import numpy as np
import sys
import matplotlib.pyplot as plt
import glob
import pandas as pd
sys.path.append("C:/Users/esposito_v/Documents/Python Scripts/FEMTO analysis/")

""" custom libraries """
import functions_FEMTO as fem
import tools_diffraction as tldiff
import fit_function as fitfct


plt.close('all')
plot = False
bootstrap = False

rNo_all = np.append(3026,np.arange(3029,3041))
#rNo_all = np.append(3026,np.arange(3029,3031))

l_area = 0.0520*0.0620*np.pi/4 #[cm]
flu_factor = 1./(1000*l_area/np.sin(np.deg2rad(10)))
fluence = np.array([10,20,30,35,40,45,50,60,70,80,100,120,150]) * flu_factor

if not 'imgs_all' in locals():
    imgs_all = np.zeros([rNo_all.shape[0],80,195,487])
    uimgs_all = np.zeros([rNo_all.shape[0],80,195,487])
    imgs_diff = np.zeros([rNo_all.shape[0],80,40,44])
        
    sys.path.append('Z:/Data1')
    procList = glob.glob('Z:\Data1\*\proc\proc*.dat')
    procFile = []
    dfList = []
    set = 2
        
    count = 0
    for ii in range(rNo_all.shape[0]):
        rNo = rNo_all[ii]
        string = '%d.dat' %rNo
        procFile.append( [s for s in procList if string in s] )
        string = ''.join(procFile[count])
        temp = pd.read_table(string,usecols=[0,1,9,20])
        dfList.append(temp)
        imgs_temp, uimgs_temp =  fem.getPilatusImgs(dfList[count], set)
        imgs_all[ii,:,:,:] = imgs_all[ii,:,:,:] + imgs_temp
        count+=1
                
#    peakPix = [87,75] # pixels position of the peak [x,y]
#    roi = np.array([[peakPix[0]-22,peakPix[0]+22],[peakPix[1]-20,peakPix[1]+20]]) # [[xmin,xmax][ymin,ymax]]
#    imgs_all = imgs_all[:,:,roi[1][0]:roi[1][1], roi[0][0]:roi[0][1]] # line = y, column = x




#for ii in range(len(rNo_all)):
##for ii in np.arange(20,60):
#    plt.figure()
#    plt.imshow(np.sum(imgs_all[1,:,:,:],0))
##    plt.imshow(imgs_all[0,ii,:,:])
#    plt.clim(-10,350)
    
popt_rot = np.zeros([rNo_all.shape[0],3])
perr_rot = np.zeros([rNo_all.shape[0],3])
ydata_rot = np.sum(np.sum(imgs_all,3),2)
ydata_rot_2 = np.zeros([rNo_all.shape[0],80])
yerr_rot = np.sqrt(np.abs(ydata_rot))
total_int = np.zeros(rNo_all.shape[0])

for ii in range(imgs_all.shape[0]):
    print('ii=%d' %ii)
    
#    ydata_rot[ii,:] = ydata_rot[ii,:] - np.mean( np.append(ydata_rot[ii,0:4],ydata_rot[ii,-4:]) )
    ydata_rot_2[ii,:] = dfList[ii]['roi8']
    ydata_rot_2[ii,:] = ydata_rot_2[ii,:] - np.mean( np.append(ydata_rot_2[ii,0:4],ydata_rot_2[ii,-4:]) )
    
    """ Fit the difference with a gaussian. One for each axis (rot, H,V) """
    fit_function = fitfct.lorentzian
    
    """ ROTATION """
    xdata_rot = np.array(dfList[ii]['top rotation (rbk)'])
#    rot_calc = xdata_rot - rot_offset
    plt.figure(50)
    plt.plot(xdata_rot, ydata_rot_2[ii,:], label=('%0.1f mJ/cm$^2$' % fluence[ii]))
    plt.legend()
    
#    plt.figure(ii+2000)
#    plt.plot(xdata_rot, ydata_rot[ii,:])
#    plt.plot(xdata_rot, ydata_rot_2[ii,:])
    
    
    A = 3e6
    x0 = -90
    sigma = 0.15  
    p0 = [A,x0,sigma]
    
    try:
#        popt_rot[ii,:], pcov = curve_fit(fitfct.gaussian, xdata_rot, ydata_rot[ii,:], p0=p0)
        if bootstrap:
            popt_rot[ii,:], perr_rot[ii,:] = fitfct.fit_bootstrap(p0, xdata_rot, ydata_rot_2[ii,:], \
                fit_function, yerr_systematic=yerr_rot[ii,:], nboot = 1000)
#            popt_rot[ii,:], perr_rot[ii,:]  = fitfct.fit_curvefit(p0, xdata_rot, ydata_rot[ii,:], fitfct.gaussian_bkg, yerr=None)
        else:
            popt_rot[ii,:], perr_rot[ii,:]  = fitfct.fit_curvefit(p0, xdata_rot, ydata_rot_2[ii,:], fit_function, \
                yerr=yerr_rot[ii,:], absolute_sigma=True)
            
        yfit_rot = fit_function(xdata_rot,*popt_rot[ii,:])
        if plot:
            plt.figure()
            plt.suptitle(ii)
            plt.errorbar(xdata_rot,ydata_rot_2[ii,:],yerr_rot[ii,:],fmt='o')
            plt.plot(xdata_rot,yfit_rot)
            plt.title('rot')
    except RuntimeError:
        popt_rot[ii,:] = np.nan


    total_int[ii] = np.trapz(ydata_rot_2[ii,:], x=xdata_rot)





plt.figure()
plt.title('Integrated intensity')
plt.plot(fluence, total_int, 'o')
#plt.ylim([0,2000000])

""" ROTATION """
plt.figure(100,figsize=(11,5))
plt.suptitle('Rotation')
plt.subplot(1,3,1)
plt.tight_layout(w_pad=4)
plt.subplots_adjust(top=0.85)
plt.title('Amplitude')
plt.errorbar(fluence,popt_rot[:,0],yerr=perr_rot[:,0],fmt='o')
plt.subplot(1,3,2)
plt.title('Rot')
plt.errorbar(fluence,popt_rot[:,1],yerr=perr_rot[:,1],fmt='o')
plt.gca().get_yaxis().get_major_formatter().set_useOffset(False)
plt.subplot(1,3,3)
plt.title('Sigma')
plt.errorbar(fluence,popt_rot[:,2],yerr=perr_rot[:,2],fmt='o')
#plt.subplot(2,2,4)
#plt.title('Background')
#plt.errorbar(time,popt_rot[:10,3],yerr=perr_rot[:10,3],fmt='o',label='T=100K')
plt.subplots_adjust(bottom=0.1)
















