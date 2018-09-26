# -*- coding: utf-8 -*-
"""
Created on Fri Jan 13 17:03:21 2017

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
plot = True

rNo_all = np.zeros([10,3])

""" low T """
rNo_all[0,:] = np.arange(2932,2935)
rNo_all[1,:] = np.arange(2935,2938)
rNo_all[2,:] = np.arange(2938,2941)
rNo_all[3,:] = np.arange(2941,2944)
rNo_all[4,:] = np.arange(2944,2947)
rNo_all[5,:] = np.arange(2947,2950)
rNo_all[6,:] = np.arange(2950,2953)
rNo_all[7,:] = np.arange(2953,2956)
rNo_all[8,:] = np.arange(2956,2959)
rNo_all[9,:] = np.arange(2959,2962)

if not 'imgs_all' in locals():
    imgs_all = np.zeros([rNo_all.shape[0],50,195,487])
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
            temp = pd.read_table(string,usecols=[0,1,9,20])
            dfList.append(temp)
            imgs_temp, uimgs_temp =  fem.getPilatusImgs(dfList[count], set)
            imgs_all[ii,:,:,:] = imgs_all[ii,:,:,:] + imgs_temp
            count+=1
                
    peakPix = [155,92] # pixels position of the peak [x,y]
    roi = np.array([[peakPix[0]-22,peakPix[0]+22],[peakPix[1]-20,peakPix[1]+20]]) # [[xmin,xmax][ymin,ymax]]
    imgs_all = imgs_all[:,:,roi[1][0]:roi[1][1], roi[0][0]:roi[0][1]] # line = y, column = x

d = 74 # distance sample - detector [mm]
pilH = 2.75 # [mm]
pilV = -47.5 # [mm]
xdb = 21.6 - 414.*0.172 # [mm]
ydb = -58.03 - 188.*0.172 # [mm]
rot_offset = 329. # offset with respect to the UB matrix

E = 7
wavelength = 12.3984/E
alpha = 0.5
N = np.array([-1,-1,2])
a = np.array([5.43,5.44,7.49]) # lattice const
aa = np.array([90,90,90.07]) # lattice angles
a0,a1,a2 = tldiff.vectorFromLengthAndAngles(a,aa)
a,aa,b,ba = tldiff.lengthAndAngles(a0,a1,a2)
U,B = tldiff.UBmat(a, aa, N)
    
for ii in range(imgs_all.shape[0]-1):
    imgs_diff[ii,:,:,:] = imgs_all[ii+1,:,:,:] - imgs_all[0,:,:,:]
#    imgs_plot = np.sum(imgs_diff,1)
#    plt.figure()
#    plt.imshow(imgs_plot[ii,:,:])
#    plt.clim(-900,900)
#    line_plot = np.sum(imgs_plot,2)
#    plt.plot(line_plot[ii,:])
   
popt_rot = np.zeros([rNo_all.shape[0],4])
perr_rot = np.zeros([rNo_all.shape[0],4])
ydata_rot = np.sum(np.sum(imgs_all,3),2)
yerr_rot = np.sqrt(np.abs(ydata_rot))

delta_img = np.zeros([ imgs_all.shape[2], imgs_all.shape[3] ])
gamma_img = np.zeros([ imgs_all.shape[2], imgs_all.shape[3] ])
hkl_imgs = np.zeros( [ imgs_all.shape[0],imgs_all.shape[1]*imgs_all.shape[2]*imgs_all.shape[3],3 ])
imgs_vec = np.zeros([ imgs_all.shape[0],imgs_all.shape[1]*imgs_all.shape[2]*imgs_all.shape[3] ])

delta_img, gamma_img, imgs_check = fem.getPixelsAngles(imgs_all[ii,:,:,:], pilH, pilV, xdb, ydb, d, roi = None)

if plot:
    plt.figure(60,figsize=(22,12))

time = np.array([-1,3,6,9,12,15,25,35,50,75])

for ii in range(imgs_all.shape[0]):
    print('ii=%d' %ii)
    
    """ data with background (baseline) subtraction """
    ydata_rot[ii,:] = ydata_rot[ii,:] - np.mean( np.append(ydata_rot[ii,0:4],ydata_rot[ii,-4:]) )
    
    """ Fit the difference with a gaussian. One for each axis (rot, H,V) """
#    x01 = 266
#    x02 = 266.28
#    fit_function = lambda x, A1, sigma1, A2, sigma2: fitfct.double_peak(x,A1,x01,sigma1,A2,x02,sigma2)
    fit_function = fitfct.double_peak_Gauss
    
    """ ROTATION """
    xdata_rot = np.array(dfList[ii]['top rotation (rbk)'])
    rot_calc = xdata_rot - rot_offset
    plt.figure(50)
    plt.plot(xdata_rot, ydata_rot[ii,:])
    
    count = 0
    for jj in range(imgs_all.shape[1]):
        for kk in range(imgs_all.shape[2]):
            for ll in range(imgs_all.shape[3]):
                imgs_vec[ii,count] = imgs_all[ii,jj,kk,ll]
                hkl_imgs[ii,count,:], Q = tldiff.hklFromAngles(E, delta_img[kk,ll], gamma_img[kk,ll], rot_calc[jj], alpha, U, B)
                count+=1
    
    
    A1 = 500
    sigma1 = 0.5
    A2 = 500
    sigma2 = 0.5
    p0 = [A1,sigma1,A2,sigma2]
    param_bounds=([-np.inf,0,-np.inf,0],[np.inf,np.inf,np.inf,np.inf])
    
    try:
#        popt_rot[ii,:], pcov = curve_fit(fitfct.gaussian, xdata_rot, ydata_rot[ii,:], p0=p0)
        popt_rot[ii,:], perr_rot[ii,:]  = fitfct.fit_curvefit(p0, xdata_rot, ydata_rot[ii,:], fit_function, \
                yerr=yerr_rot[ii,:], absolute_sigma=True, bounds=param_bounds)
            
        yfit_rot = fit_function(xdata_rot,*popt_rot[ii,:])
        yfit_peak1 = fitfct.gaussian(xdata_rot, popt_rot[ii,0],266,popt_rot[ii,1])
        yfit_peak2 = fitfct.gaussian(xdata_rot, popt_rot[ii,2],266.28,popt_rot[ii,3])
        if plot:
            plt.figure(60)
            plt.subplot(2,5,ii+1)
            plt.title('dl = %.0f ps' % time[ii])
            plt.errorbar(xdata_rot,ydata_rot[ii,:],yerr_rot[ii,:],fmt='o')
            plt.plot(xdata_rot,yfit_rot)
            plt.plot(xdata_rot,yfit_peak1)
            plt.plot(xdata_rot,yfit_peak2)
    except RuntimeError:
        popt_rot[ii,:] = np.nan
        
        

#%%
""" ROTATION """
time = np.array([-1,3,6,9,12,15,25,35,50,75])

plt.figure(100,figsize=(11,5))
plt.suptitle('Rotation')
plt.subplot(1,2,1)
plt.subplots_adjust(top=0.85)
plt.title('Amplitude')
plt.errorbar(time,popt_rot[:,0],yerr=perr_rot[:,0],fmt='o')
plt.errorbar(time,popt_rot[:,2],yerr=perr_rot[:,2],fmt='o')
plt.subplot(1,2,2)
plt.title('Sigma')
plt.errorbar(time,popt_rot[:,1],yerr=perr_rot[:,1],fmt='o')
plt.errorbar(time,popt_rot[:,3],yerr=perr_rot[:,3],fmt='o')

        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        