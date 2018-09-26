# -*- coding: utf-8 -*-
"""
Created on Mon Jun 20 14:22:14 2016

@author: esposito_v
"""


"""
Fits gaussian of the rotation scans and along the different direction on the detector.
A first fit was made with Gaussians with background.
Now the fit is made with a pure Gaussian and the background is subtracted prior to the fit
by taking the average of a few point in the base line.


A bootstrap method has been implemented to check the error value of the fitted parameters.
This looks fine, compared to the values of the single fit. Systematic experimental errors have 
to be implemented. This can easily be done by giving a yerr to the fitting function. Double check
using the bootstrap is recommended.

A calculation of the hkl position based on the fits is then made.
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

import plotly.plotly as py
import plotly.graph_objs as go




plt.close('all')
plot = True
bootstrap = False

rNo_all = np.zeros([20,3])

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

""" high T """
rNo_all[10,:] = np.arange(2965,2968)
rNo_all[11,:] = np.arange(2968,2971)
rNo_all[12,:] = np.arange(2971,2974)
rNo_all[13,:] = np.arange(2974,2977)
rNo_all[14,:] = np.arange(2977,2980)
rNo_all[15,:] = np.arange(2980,2983)
rNo_all[16,:] = np.arange(2983,2986)
rNo_all[17,:] = np.arange(2986,2989)
rNo_all[18,:] = np.arange(2989,2992)
rNo_all[19,:] = np.arange(2992,2995)


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
        print(rNo_all[ii])
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
    imgs_plot = np.sum(imgs_diff,1)
#    plt.figure()
#    plt.imshow(imgs_plot[ii,:,:])
#    plt.clim(-900,900)
#    line_plot = np.sum(imgs_plot,2)
#    plt.plot(line_plot[ii,:])
   
popt_rot = np.zeros([rNo_all.shape[0],3])
perr_rot = np.zeros([rNo_all.shape[0],3])
ydata_rot = np.sum(np.sum(imgs_all,3),2)
yerr_rot = np.sqrt(np.abs(ydata_rot))

popt_H = np.zeros([rNo_all.shape[0],3])
perr_H = np.zeros([rNo_all.shape[0],3])
ydata_H = np.sum(np.sum(imgs_all,2),1)
yerr_H = np.sqrt(np.abs(ydata_H))

popt_V = np.zeros([rNo_all.shape[0],3])
perr_V = np.zeros([rNo_all.shape[0],3])
ydata_V = np.sum(np.sum(imgs_all,3),1)
yerr_V = np.sqrt(np.abs(ydata_V))

delta_img = np.zeros([ imgs_all.shape[2], imgs_all.shape[3] ])
gamma_img = np.zeros([ imgs_all.shape[2], imgs_all.shape[3] ])
hkl_imgs = np.zeros( [ imgs_all.shape[0],imgs_all.shape[1]*imgs_all.shape[2]*imgs_all.shape[3],3 ])
imgs_vec = np.zeros([ imgs_all.shape[0],imgs_all.shape[1]*imgs_all.shape[2]*imgs_all.shape[3] ])

delta_img, gamma_img, imgs_check = fem.getPixelsAngles(imgs_all[ii,:,:,:], pilH, pilV, xdb, ydb, d, roi = None)

for ii in range(imgs_all.shape[0]):
    print('ii=%d' %ii)
    
    """ data with background (baseline) subtraction """
    ydata_rot[ii,:] = ydata_rot[ii,:] - np.mean( np.append(ydata_rot[ii,0:4],ydata_rot[ii,-4:]) )
    ydata_H[ii,:] = ydata_H[ii,:] - np.mean( np.append(ydata_H[ii,0:4],ydata_H[ii,-4:]) )
    ydata_V[ii,:] = ydata_V[ii,:] - np.mean( np.append(ydata_V[ii,0:4],ydata_V[ii,-4:]) )
    
    """ Fit the difference with a gaussian. One for each axis (rot, H,V) """
    fit_function = fitfct.lorentzian
    
    """ ROTATION """
    xdata_rot = np.array(dfList[ii]['top rotation (rbk)'])
    rot_calc = xdata_rot - rot_offset
    plt.figure(50)
    plt.plot(xdata_rot, ydata_rot[ii,:])
    
#    count = 0
#    for jj in range(imgs_all.shape[1]):
#        for kk in range(imgs_all.shape[2]):
#            for ll in range(imgs_all.shape[3]):
#                imgs_vec[ii,count] = imgs_all[ii,jj,kk,ll]
#                hkl_imgs[ii,count,:], Q = tldiff.hklFromAngles(E, delta_img[kk,ll], gamma_img[kk,ll], rot_calc[jj], alpha, U, B)
#                count+=1
    
    
    A = 1300
    x0 = 266.1
    sigma = 0.15
    bkg = 10    
    p0 = [A,x0,sigma]
    
    try:
#        popt_rot[ii,:], pcov = curve_fit(fitfct.gaussian, xdata_rot, ydata_rot[ii,:], p0=p0)
        if bootstrap:
            popt_rot[ii,:], perr_rot[ii,:] = fitfct.fit_bootstrap(p0, xdata_rot, ydata_rot[ii,:], \
                fit_function, yerr_systematic=yerr_rot[ii,:]+0.001, nboot = 100)
#            popt_rot[ii,:], perr_rot[ii,:]  = fitfct.fit_curvefit(p0, xdata_rot, ydata_rot[ii,:], fitfct.gaussian_bkg, yerr=None)
        else:
            popt_rot[ii,:], perr_rot[ii,:]  = fitfct.fit_curvefit(p0, xdata_rot, ydata_rot[ii,:], fit_function, \
                yerr=yerr_rot[ii,:]+0.0005, absolute_sigma=True)
            
        yfit_rot = fit_function(xdata_rot,popt_rot[ii,0],popt_rot[ii,1],popt_rot[ii,2])
        if plot:
            plt.figure(figsize=(12,5))
            plt.suptitle(ii)
            plt.subplot(1,3,1)
            plt.tight_layout(w_pad=4)
            plt.errorbar(xdata_rot,ydata_rot[ii,:],yerr_rot[ii,:],fmt='o')
            plt.plot(xdata_rot,yfit_rot)
            plt.title('rot')
    except RuntimeError:
        popt_rot[ii,:] = np.nan
        

    """ HORIZONTAL """
    xdata_H = np.arange(peakPix[0]-22,peakPix[0]+22)
    
    A = 4000
    x0 = peakPix[0]
    sigma = 5
    bkg = 10    
    p0 = [A,x0,sigma]
    
    try:
#        popt_H[ii,:], pcov = curve_fit(fitfct.gaussian, xdata_H, ydata_H[ii,:], p0=p0)
        if bootstrap:
            popt_H[ii,:], perr_H[ii,:] = fitfct.fit_bootstrap(p0, xdata_H, ydata_H[ii,:], \
                fit_function, yerr_systematic=yerr_H[ii,:], nboot = 1000)
#            popt_H[ii,:], perr_H[ii,:]  = fitfct.fit_curvefit(p0, xdata_H, ydata_H[ii,:], fitfct.gaussian_bkg, yerr=None)
        else:
            popt_H[ii,:], perr_H[ii,:]  = fitfct.fit_curvefit(p0, xdata_H, ydata_H[ii,:], fit_function, \
                yerr=yerr_H[ii,:], absolute_sigma=True)
            
        yfit_H = fit_function(xdata_H,popt_H[ii,0],popt_H[ii,1],popt_H[ii,2])
        if plot:
            plt.subplot(1,3,2)
            plt.errorbar(xdata_H,ydata_H[ii,:],yerr_H[ii,:],fmt='o')
            plt.plot(xdata_H,yfit_H)
            plt.title('pix H')
    except RuntimeError:
        popt_H[ii,:] = np.nan
        
        
    """ VERTICAL """
    xdata_V = np.arange(peakPix[1]-20,peakPix[1]+20)
    
    A = 4000
    x0 = peakPix[1]
    sigma = 5
    bkg = 10    
    p0 = [A,x0,sigma]
    
    try:
#        popt_V[ii,:], pcov = curve_fit(fitfct.gaussian, xdata_V, ydata_V[ii,:], p0=p0)
        if bootstrap:
            popt_V[ii,:], perr_V[ii,:] = fitfct.fit_bootstrap(p0, xdata_V, ydata_V[ii,:], \
                fit_function, yerr_systematic=0.0, nboot = 1000)
#            popt_V[ii,:], perr_V[ii,:]  = fitfct.fit_curvefit(p0, xdata_V, ydata_V[ii,:], fitfct.gaussian_bkg, yerr=None)
        else:
            popt_V[ii,:], perr_V[ii,:]  = fitfct.fit_curvefit(p0, xdata_V, ydata_V[ii,:], fit_function, \
                yerr=yerr_V[ii,:], absolute_sigma=True)
            
        yfit_V = fit_function(xdata_V,popt_V[ii,0],popt_V[ii,1],popt_V[ii,2])
        if plot:
            plt.subplot(1,3,3)
            plt.errorbar(xdata_V,ydata_V[ii,:],yerr_V[ii,:],fmt='o')
            plt.plot(xdata_V,yfit_V)
            plt.title('pix V')
    except RuntimeError:
        popt_V[ii,:] = np.nan


#%%
time = np.array([-1.,3,6,9,12,15,25,35,50,75])

""" INTEGRATED INTENSITY """
peakint_tot = np.sum(ydata_rot,1)
plt.figure(15)
plt.errorbar(time,peakint_tot[:10],np.sqrt(peakint_tot[:10]),fmt='o',label='100K')
plt.errorbar(time,peakint_tot[10:],np.sqrt(peakint_tot[10:]),fmt='o',label='250K')
plt.legend()
plt.title('Integrated intensity')
plt.xlabel('time [ps]')



""" ROTATION """
plt.figure(100,figsize=(11,5))
plt.suptitle('Rotation')
plt.subplot(1,3,1)
plt.tight_layout(w_pad=4)
plt.subplots_adjust(top=0.85)
plt.title('Amplitude')
plt.errorbar(time,popt_rot[:10,0],yerr=perr_rot[:10,0],fmt='o')
plt.subplot(1,3,2)
plt.title('Rot')
plt.errorbar(time,popt_rot[:10,1],yerr=perr_rot[:10,1],fmt='o')
plt.gca().get_yaxis().get_major_formatter().set_useOffset(False)
plt.subplot(1,3,3)
plt.title('Sigma')
plt.errorbar(time,popt_rot[:10,2],yerr=perr_rot[:10,2],fmt='o')
#plt.subplot(2,2,4)
#plt.title('Background')
#plt.errorbar(time,popt_rot[:10,3],yerr=perr_rot[:10,3],fmt='o',label='T=100K')

plt.figure(100,figsize=(11,5))
plt.suptitle('Rotation')
plt.subplot(1,3,1)
plt.title('Amplitude')
plt.errorbar(time,popt_rot[10:,0],yerr=perr_rot[10:,0],fmt='o')
plt.xlabel('time [ps]')
plt.subplot(1,3,2)
plt.title('Rot')
plt.errorbar(time,popt_rot[10:,1],yerr=perr_rot[10:,1],fmt='o')
plt.gca().get_yaxis().get_major_formatter().set_useOffset(False)
plt.xlabel('time [ps]')
plt.subplot(1,3,3)
plt.title('Sigma')
plt.errorbar(time,popt_rot[10:,2],yerr=perr_rot[10:,2],fmt='o')
plt.xlabel('time [ps]')
#plt.subplot(2,2,4)
#plt.title('Background')
#plt.errorbar(time,popt_rot[10:,3],yerr=perr_rot[10:,3],fmt='o',label='T=250K')
#plt.legend(loc=0)
plt.subplots_adjust(bottom=0.1)



""" HORIZONTAL """
plt.figure(101,figsize=(11,5))
plt.suptitle('Horizontal')
plt.subplot(1,3,1)
plt.tight_layout(w_pad=4)
plt.subplots_adjust(top=0.85)
plt.title('Amplitude')
plt.errorbar(time,popt_H[:10,0],yerr=perr_H[:10,0],fmt='o')
plt.subplot(1,3,2)
plt.title('Horizontal pixel')
plt.errorbar(time,popt_H[:10,1],yerr=perr_H[:10,1],fmt='o')
plt.gca().get_yaxis().get_major_formatter().set_useOffset(False)
plt.subplot(1,3,3)
plt.title('Sigma')
plt.errorbar(time,popt_H[:10,2],yerr=perr_H[:10,2],fmt='o')
#plt.subplot(2,2,4)
#plt.title('Background')
#plt.errorbar(time,popt_H[:10,3],yerr=perr_H[:10,3],fmt='o',label='T=100K')

plt.figure(101,figsize=(11,5))
plt.suptitle('Horizontal')
plt.subplot(1,3,1)
plt.title('Amplitude')
plt.errorbar(time,popt_H[10:,0],yerr=perr_H[10:,0],fmt='o')
plt.xlabel('time [ps]')
plt.subplot(1,3,2)
plt.title('Horizontal pixel')
plt.errorbar(time,popt_H[10:,1],yerr=perr_H[10:,1],fmt='o')
plt.gca().get_yaxis().get_major_formatter().set_useOffset(False)
plt.xlabel('time [ps]')
plt.subplot(1,3,3)
plt.title('Sigma')
plt.errorbar(time,popt_H[10:,2],yerr=perr_H[10:,2],fmt='o')
plt.xlabel('time [ps]')
#plt.subplot(2,2,4)
#plt.title('Background')
#plt.errorbar(time,popt_H[10:,3],yerr=perr_H[10:,3],fmt='o',label='T=250K')
#plt.legend(loc=0)
plt.subplots_adjust(bottom=0.1)



""" VERTICAL """
plt.figure(102,figsize=(11,5))
plt.suptitle('Vertical')
plt.subplot(1,3,1)
plt.tight_layout(w_pad=4)
plt.subplots_adjust(top=0.85)
plt.title('Amplitude')
plt.errorbar(time,popt_V[:10,0],yerr=perr_V[:10,0],fmt='o')
plt.subplot(1,3,2)
plt.title('Vertical pixel')
plt.errorbar(time,popt_V[:10,1],yerr=perr_V[:10,1],fmt='o')
plt.subplot(1,3,3)
plt.title('Sigma')
plt.errorbar(time,popt_V[:10,2],yerr=perr_V[:10,2],fmt='o')
#plt.subplot(2,2,4)
#plt.title('Background')
#plt.errorbar(time,popt_V[:10,3],yerr=perr_V[:10,0],fmt='o',label='T=100K')


plt.figure(102,figsize=(11,5))
plt.suptitle('Vertical')
plt.subplot(1,3,1)
plt.title('Amplitude')
plt.errorbar(time,popt_V[10:,0],yerr=perr_V[10:,0],fmt='o')
plt.xlabel('time [ps]')
plt.subplot(1,3,2)
plt.title('Vertical pixel')
plt.errorbar(time,popt_V[10:,1],yerr=perr_V[10:,1],fmt='o')
plt.gca().get_yaxis().get_major_formatter().set_useOffset(False)
plt.xlabel('time [ps]')
plt.subplot(1,3,3)
plt.title('Sigma')
plt.errorbar(time,popt_V[10:,2],yerr=perr_V[10:,2],fmt='o')
plt.xlabel('time [ps]')
#plt.subplot(2,2,4)
#plt.title('Background')
#plt.errorbar(time,popt_V[10:,3],yerr=perr_V[10:,3],fmt='o',label='T=250K')
#plt.legend(loc=0)
plt.subplots_adjust(bottom=0.1)



#""" K-L plot """
#idx = imgs_vec > 25
#plt.figure(11)
#plt.plot(hkl_imgs[1,idx[1,:],1], hkl_imgs[1,idx[1,:],2], 'o')
#plt.figure(12)
#plt.tricontourf(hkl_imgs[1,:,1], hkl_imgs[1,:,2],imgs_vec[1,:])


""" HKL from fit """
pixels = np.zeros([rNo_all.shape[0],2])
rot = popt_rot[:,1] - rot_offset

loops = 500
hkl = np.zeros([loops,rNo_all.shape[0],3])
Q = np.zeros([loops,rNo_all.shape[0]])

for jj in range(loops):    
    randomPixelsH = np.random.normal(popt_H[:,1], perr_H[:,1])
    randomPixelsV = np.random.normal(popt_V[:,1], perr_V[:,1])
    randrot = rot + np.random.normal(0., perr_rot[:,1])
    pixels[:,0] = randomPixelsH
    pixels[:,1] = randomPixelsV
    
    delta, gamma = fem.getPixelsAngles2(pixels, pilH, pilV, xdb, ydb, d)
    theta = np.rad2deg(np.arccos( np.cos(np.deg2rad(gamma)) * np.cos(np.deg2rad(delta)) )) /2


    for ii in range(rNo_all.shape[0]):
        hkl[jj,ii,:], Q[jj,ii] = tldiff.hklFromAngles(E, delta[ii], gamma[ii], randrot[ii], alpha, U, B)

hkl_err = np.std(hkl,0)
hkl = np.mean(hkl,0)
dhkl = hkl-hkl[0,:]
dhkl2 = dhkl / hkl[0,:]

d002 = wavelength/2/np.sin(np.deg2rad(theta))

monoclinic_angle = np.rad2deg( np.arcsin( np.sqrt(2*d002/a[2]) ))

plt.figure(200, figsize=(14,6))
plt.subplot(131)
plt.title('h')
plt.errorbar(time,hkl[:10,0],hkl_err[:10,0],fmt='o')
plt.errorbar(time,hkl[10:,0],hkl_err[10:,0],fmt='o')
plt.xlabel('time [ps]')

plt.subplot(132)
plt.title('k')
plt.errorbar(time,hkl[:10,1],hkl_err[:10,1],fmt='o')
plt.errorbar(time,hkl[10:,1],hkl_err[10:,1],fmt='o')
plt.xlabel('time [ps]')

plt.subplot(133)
plt.title('l')
plt.errorbar(time,hkl[:10,2],hkl_err[:10,2],fmt='o',label='T=100K')
plt.errorbar(time,hkl[10:,2],hkl_err[10:,2],fmt='o',label='T=250K')
plt.legend(loc=4)
plt.xlabel('time [ps]')


#%%
""" Convert rot scan in Q scan """
hkl_rot = np.zeros([rot_calc.shape[0],3])
Q_rot = np.zeros(rot_calc.shape[0])
Q_rot2 = np.zeros(rot_calc.shape[0])
for ii in range(rot_calc.shape[0]):
        hkl_rot[ii,:], Q_rot[ii] = tldiff.hklFromAngles(E, delta[0], gamma[0], rot_calc[ii], alpha, U, B)
        Q_rot2[ii] = np.linalg.norm( np.dot( U, np.dot(B,hkl_rot[ii,:]) ) )
        
        
        
#%% projection along out-of-plane and in plane direction\
def perpendicular_vector(v):
    if v[1] == 0 and v[2] == 0:
        if v[0] == 0:
            raise ValueError('zero vector')
        else:
            return np.cross(v, [0, 1, 0])
    return np.cross(v, [1, 0, 0])
    

Nnormalized = N*b/np.linalg.norm(N*b) 
hkl_recip = hkl*b
hkl_recip_norm = np.sqrt(hkl_recip[:,0]**2+hkl_recip[:,1]**2+hkl_recip[:,2]**2)
oop_proj = np.dot(hkl_recip,Nnormalized)
ipvec1 = perpendicular_vector(Nnormalized)
ipvec1 = ipvec1/np.linalg.norm(ipvec1)
ipvec2 = np.cross(Nnormalized,ipvec1)
ipvec2 = ipvec2/np.linalg.norm(ipvec2)

ip_proja = np.sqrt( np.dot(hkl_recip,ipvec1)**2 + np.dot(hkl_recip,ipvec2)**2 )
ip_projb = np.sqrt( hkl_recip_norm**2-oop_proj**2 )

err = 0.0001*(hkl_recip[:,0]**2+hkl_recip[:,1]**2+hkl_recip[:,2]**2)**0.5*3

plt.figure()
plt.subplot(1,2,1)
plt.title('100K')
plt.errorbar(time,oop_proj[:10], yerr=err[:10], label='out-of-plane')
plt.errorbar(time,ip_proja[:10], yerr=err[:10], label='in plane')
plt.legend()
        
plt.subplot(1,2,2)
plt.title('250K')
plt.errorbar(time,oop_proj[10:], yerr=err[10:], label='out-of-plane')
plt.errorbar(time,ip_proja[10:], yerr=err[10:], label='in plane')
plt.legend()


#%%
""" 3D plot """
X,Y = np.meshgrid(time,xdata_rot) 
Z = np.transpose(ydata_rot[:10,:])
plt.figure()
plt.contourf(X,Y,Z)


""" with plotly """
#plt.figure()
#XYZ = [go.Contour(z=np.transpose(ydata_rot[:10,:]), x=time, y=xdata_rot)]
#py.iplot(XYZ)
























