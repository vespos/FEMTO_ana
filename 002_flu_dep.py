# -*- coding: utf-8 -*-
"""
Created on Mon Dec 12 12:35:57 2016

@author: esposito_v
"""

"""
Fluence dependence at t = 100ps after excitation. Full analysis, including hkl calculation
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
                
    peakPix = [155,92] # pixels position of the peak [x,y]
    roi = np.array([[peakPix[0]-22,peakPix[0]+22],[peakPix[1]-20,peakPix[1]+20]]) # [[xmin,xmax][ymin,ymax]]
    imgs_all = imgs_all[:,:,roi[1][0]:roi[1][1], roi[0][0]:roi[0][1]] # line = y, column = x
    uimgs_all = uimgs_all[:,:,roi[1][0]:roi[1][1], roi[0][0]:roi[0][1]] # line = y, column = x

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
a = np.array([5.46,5.46,7.56])
aa = np.array([90,90,90.07])
a0,a1,a2 = tldiff.vectorFromLengthAndAngles(a,aa)
a,aa,b,ba = tldiff.lengthAndAngles(a0,a1,a2)
U,B = tldiff.UBmat(a, aa, N)

for ii in range(imgs_all.shape[0]-1):
    imgs_diff[ii,:,:,:] = imgs_all[ii+1,:,:,:] - imgs_all[0,:,:,:]
    imgs_plot = np.sum(imgs_diff,1)
#    plt.figure()
#    plt.imshow(imgs_plot[ii,:,:])
#    plt.clim(-900,900)
#    plt.figure()
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

upopt_rot = np.zeros([rNo_all.shape[0],3])
uperr_rot = np.zeros([rNo_all.shape[0],3])
uydata_rot = np.sum(np.sum(uimgs_all,3),2)
uyerr_rot = np.sqrt(np.abs(uydata_rot))

upopt_H = np.zeros([rNo_all.shape[0],3])
uperr_H = np.zeros([rNo_all.shape[0],3])
uydata_H = np.sum(np.sum(uimgs_all,2),1)
uyerr_H = np.sqrt(np.abs(uydata_H))

upopt_V = np.zeros([rNo_all.shape[0],3])
uperr_V = np.zeros([rNo_all.shape[0],3])
uydata_V = np.sum(np.sum(uimgs_all,3),1)
uyerr_V = np.sqrt(np.abs(uydata_V))

delta_img = np.zeros([ imgs_all.shape[2], imgs_all.shape[3] ])
gamma_img = np.zeros([ imgs_all.shape[2], imgs_all.shape[3] ])
hkl_imgs = np.zeros( [ imgs_all.shape[0],imgs_all.shape[1]*imgs_all.shape[2]*imgs_all.shape[3],3 ])
imgs_vec = np.zeros([ imgs_all.shape[0],imgs_all.shape[1]*imgs_all.shape[2]*imgs_all.shape[3] ])

delta_img, gamma_img, imgs_check = fem.getPixelsAngles(imgs_all[ii,:,:,:], pilH, pilV, xdb, ydb, d, roi = None)


for ii in range(imgs_all.shape[0]):
    print('ii=%d' %ii)
    
    ydata_rot[ii,:] = ydata_rot[ii,:] - np.mean( np.append(ydata_rot[ii,0:4],ydata_rot[ii,-4:]) )
    ydata_H[ii,:] = ydata_H[ii,:] - np.mean( np.append(ydata_H[ii,0:4],ydata_H[ii,-4:]) )
    ydata_V[ii,:] = ydata_V[ii,:] - np.mean( np.append(ydata_V[ii,0:4],ydata_V[ii,-4:]) )
    
    uydata_rot[ii,:] = uydata_rot[ii,:] - np.mean( np.append(uydata_rot[ii,0:4],uydata_rot[ii,-4:]) )
    uydata_H[ii,:] = uydata_H[ii,:] - np.mean( np.append(uydata_H[ii,0:4],uydata_H[ii,-4:]) )
    uydata_V[ii,:] = uydata_V[ii,:] - np.mean( np.append(uydata_V[ii,0:4],uydata_V[ii,-4:]) )
    
    """ Fit the difference with a gaussian. One for each axis (rot, H,V) """
    fit_function = fitfct.gaussian
    
    """ ROTATION """
    xdata_rot = np.array(dfList[ii]['top rotation (rbk)'])
    rot_calc = xdata_rot - rot_offset
    plt.figure(50)
    plt.plot(xdata_rot, ydata_rot[ii,:])
    plt.figure(51)
    plt.plot(xdata_rot, uydata_rot[ii,:])
    
#    plt.figure(ii+2000)
#    plt.plot(xdata_rot, ydata_rot[ii,:])
#    plt.plot(xdata_rot, uydata_rot[ii,:])
    
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
                fit_function, yerr_systematic=yerr_rot[ii,:], nboot = 1000)
#            popt_rot[ii,:], perr_rot[ii,:]  = fitfct.fit_curvefit(p0, xdata_rot, ydata_rot[ii,:], fitfct.gaussian_bkg, yerr=None)
        else:
            popt_rot[ii,:], perr_rot[ii,:]  = fitfct.fit_curvefit(p0, xdata_rot, ydata_rot[ii,:], fit_function, \
                yerr=yerr_rot[ii,:], absolute_sigma=True)
            upopt_rot[ii,:], uperr_rot[ii,:]  = fitfct.fit_curvefit(p0, xdata_rot, uydata_rot[ii,:], fit_function, \
                yerr=uyerr_rot[ii,:], absolute_sigma=True)
        
        popt_rot[ii,1], popt_rot[ii,2] = fitfct.moments(xdata_rot, ydata_rot[ii,:])
        upopt_rot[ii,1], upopt_rot[ii,2] = fitfct.moments(xdata_rot, uydata_rot[ii,:])
            
        yfit_rot = fit_function(xdata_rot,*popt_rot[ii,:])
        uyfit_rot = fit_function(xdata_rot,*upopt_rot[ii,:])
        
        if plot:
            plt.figure(figsize=(12,5))
            plt.suptitle(ii)
            plt.subplot(1,3,1)
            plt.tight_layout(w_pad=4)
            plt.errorbar(xdata_rot,ydata_rot[ii,:],yerr_rot[ii,:],fmt='o')
            plt.errorbar(xdata_rot,uydata_rot[ii,:],uyerr_rot[ii,:],fmt='o')
            plt.plot(xdata_rot,yfit_rot)
            plt.plot(xdata_rot,uyfit_rot)
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
            upopt_H[ii,:], uperr_H[ii,:]  = fitfct.fit_curvefit(p0, xdata_H, uydata_H[ii,:], fit_function, \
                yerr=uyerr_H[ii,:], absolute_sigma=True)
            
        yfit_H = fit_function(xdata_H,*popt_H[ii,:])
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
            upopt_V[ii,:], uperr_V[ii,:]  = fitfct.fit_curvefit(p0, xdata_V, uydata_V[ii,:], fit_function, \
                yerr=uyerr_V[ii,:], absolute_sigma=True)
            
        yfit_V = fit_function(xdata_V,*popt_V[ii,:])
        if plot:
            plt.subplot(1,3,3)
            plt.errorbar(xdata_V,ydata_V[ii,:],yerr_V[ii,:],fmt='o')
            plt.plot(xdata_V,yfit_V)
            plt.title('pix V')
    except RuntimeError:
        popt_V[ii,:] = np.nan



#%%
l_area = 0.0520*0.0620*np.pi/4 #[cm]
flu_factor = 1./(1000*l_area/np.sin(np.deg2rad(10)))
fluence = np.array([20,40,60,80,100,120]) * flu_factor

""" INTEGRATED INTENSITY """
peakint_tot = np.sum(ydata_rot,1)
plt.figure(10)
plt.errorbar(fluence,peakint_tot,np.sqrt(peakint_tot),fmt='o',label='100K')
plt.legend()
plt.title('Integrated intensity')
plt.xlabel('fluence (mJ/cm^2)')
upeakint_tot = np.sum(uydata_rot,1)
plt.errorbar(fluence,upeakint_tot,np.sqrt(upeakint_tot),fmt='o',label='unpumped')
plt.legend()


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
plt.errorbar(0,upopt_rot[0,1], yerr=uperr_rot[0,1], fmt='o')
plt.plot(np.array([0,6.5]),np.array([266.27,266.27]))
plt.gca().get_yaxis().get_major_formatter().set_useOffset(False)
plt.subplot(1,3,3)
plt.title('Sigma')
plt.errorbar(fluence,popt_rot[:,2],yerr=perr_rot[:,2],fmt='o')
#plt.subplot(2,2,4)
#plt.title('Background')
#plt.errorbar(time,popt_rot[:10,3],yerr=perr_rot[:10,3],fmt='o',label='T=100K')
plt.subplots_adjust(bottom=0.1)



""" HORIZONTAL """
plt.figure(101,figsize=(11,5))
plt.suptitle('Horizontal')
plt.subplot(1,3,1)
plt.tight_layout(w_pad=4)
plt.subplots_adjust(top=0.85)
plt.title('Amplitude')
plt.errorbar(fluence,popt_H[:,0],yerr=perr_H[:,0],fmt='o')
plt.subplot(1,3,2)
plt.title('Horizontal pixel')
plt.errorbar(fluence,popt_H[:,1],yerr=perr_H[:,1],fmt='o')
plt.errorbar(0,upopt_H[0,1], yerr=uperr_H[0,1], fmt='o')
plt.gca().get_yaxis().get_major_formatter().set_useOffset(False)
plt.subplot(1,3,3)
plt.title('Sigma')
plt.errorbar(fluence,popt_H[:,2],yerr=perr_H[:,2],fmt='o')
#plt.subplot(2,2,4)
#plt.title('Background')
#plt.errorbar(time,popt_H[:10,3],yerr=perr_H[:10,3],fmt='o',label='T=100K')
plt.subplots_adjust(bottom=0.1)



""" VERTICAL """
plt.figure(102,figsize=(11,5))
plt.suptitle('Vertical')
plt.subplot(1,3,1)
plt.tight_layout(w_pad=4)
plt.subplots_adjust(top=0.85)
plt.title('Amplitude')
plt.errorbar(fluence,popt_V[:,0],yerr=perr_V[:,0],fmt='o')
plt.subplot(1,3,2)
plt.title('Vertical pixel')
plt.errorbar(fluence,popt_V[:,1],yerr=perr_V[:,1],fmt='o')
plt.errorbar(0,upopt_V[0,1], yerr=uperr_V[0,1], fmt='o')
plt.subplot(1,3,3)
plt.title('Sigma')
plt.errorbar(fluence,popt_V[:,2],yerr=perr_V[:,2],fmt='o')
#plt.subplot(2,2,4)
#plt.title('Background')
#plt.errorbar(time,popt_V[:10,3],yerr=perr_V[:10,0],fmt='o',label='T=100K')
plt.subplots_adjust(bottom=0.1)



""" K-L plot """
#idx = imgs_vec > 25
#plt.figure(11)
#plt.plot(hkl_imgs[1,idx[1,:],1], hkl_imgs[1,idx[1,:],2], 'o')
#plt.figure(12)
#plt.tricontourf(hkl_imgs[1,:,1], hkl_imgs[1,:,2],imgs_vec[1,:])


""" HKL from fit """
pixels = np.zeros([rNo_all.shape[0],2])
rot = popt_rot[:,1] - rot_offset
urot = upopt_rot[0,1] - rot_offset

loops = 500
hkl = np.zeros([loops,rNo_all.shape[0],3])
uhkl = np.zeros([loops,3])
Q = np.zeros([loops,rNo_all.shape[0]])
uQ = np.zeros([loops,3])

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
        
    urandomPixelsH = np.random.normal(upopt_H[0,1], uperr_H[0,1])
    urandomPixelsV = np.random.normal(upopt_V[0,1], uperr_V[0,1])
    urandrot = urot + np.random.normal(0., uperr_rot[0,1])
    upixels = np.array([urandomPixelsH, urandomPixelsV])
    
    delta, gamma = fem.getPixelsAngles2(upixels, pilH, pilV, xdb, ydb, d)
    theta = np.rad2deg(np.arccos( np.cos(np.deg2rad(gamma)) * np.cos(np.deg2rad(delta)) )) /2

    uhkl[jj], uQ[jj] = tldiff.hklFromAngles(E, delta, gamma, urandrot, alpha, U, B)

hkl_err = np.std(hkl,0)
hkl = np.mean(hkl,0)
dhkl = hkl-hkl[0,:]
dhkl2 = dhkl / hkl[0,:]
uhkl_err = np.std(uhkl,0)
uhkl = np.mean(uhkl,0)

d002 = wavelength/2/np.sin(np.deg2rad(theta))

monoclinic_angle = np.rad2deg( np.arcsin( np.sqrt(2*d002/a[2]) ))

plt.figure(200, figsize=(14,6))
plt.subplot(131)
plt.title('h')
plt.errorbar(fluence,hkl[:,0],hkl_err[:,0],fmt='o')
plt.xlabel('time [ps]')

plt.subplot(132)
plt.title('k')
plt.errorbar(fluence,hkl[:,1],hkl_err[:,1],fmt='o')
plt.xlabel('time [ps]')

plt.subplot(133)
plt.title('l')
plt.errorbar(fluence,hkl[:,2],hkl_err[:,2],fmt='o',label='T=100K')
plt.legend(loc=4)
plt.xlabel('time [ps]')


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
uhkl_recip = uhkl*b
hkl_recip_norm = np.sqrt(hkl_recip[:,0]**2+hkl_recip[:,1]**2+hkl_recip[:,2]**2)
oop_proj = np.dot(hkl_recip,Nnormalized)
uoop_proj = np.dot(uhkl_recip,Nnormalized)
ipvec1 = perpendicular_vector(Nnormalized)
ipvec1 = ipvec1/np.linalg.norm(ipvec1)
ipvec2 = np.cross(Nnormalized,ipvec1)
ipvec2 = ipvec2/np.linalg.norm(ipvec2)

ip_proja = np.sqrt( np.dot(hkl_recip,ipvec1)**2 + np.dot(hkl_recip,ipvec2)**2 )
uip_proja = np.sqrt( np.dot(uhkl_recip,ipvec1)**2 + np.dot(uhkl_recip,ipvec2)**2 )
ip_projb = np.sqrt( hkl_recip_norm**2-oop_proj**2 )

err = 0.0001*(hkl_recip[:,0]**2+hkl_recip[:,1]**2+hkl_recip[:,2]**2)**0.5*3

plt.figure(1000)
plt.errorbar(fluence,oop_proj, yerr=err, label='out-of-plane')
plt.errorbar(fluence,ip_proja, yerr=err, label='in plane')
plt.legend()






























 