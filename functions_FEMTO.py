# -*- coding: utf-8 -*-
"""
Created on Tue Apr 12 11:43:51 2016

@author: esposito_v
"""

import matplotlib.pyplot as plt
import numpy as np
import tools_diffraction as tldiff
import pandas as pd
import scipy.stats as stats
#import Tkinter as tk
#import tkFileDialog


def import_FEMTO(datapath=None, det = 'APD'):
    """
    Import data from the given average file.
    The detector can be either 'Pilatus' or 'APD'
    Note: if the dta were taken with the Pilatus, this doesn't load or look at the images, but just loads the data from the ROIs 
    that were set during the experiment. This is good to have a quick look at scans, but probably not what you want to use for
    a thorough data analysis.
    """
    
#    if datapath == None:
#        root = tk.Tk()
#        root.withdraw()
#        datapath = tkFileDialog.askopenfilename()


    if det == 'Pilatus':
        """ check for bad lines (starting with '#') and skip them """    
        data_raw = pd.read_table(datapath, header=None, delimiter=' ')
        matches = [s for s in data_raw[0] if '#' in s]
        skip_lines = len(matches)+1
        
        """ set headers manually """
        header_names = ['mot', 'mot rbk', 'roi1', 'roi2', 'roi3', 'roi4', 'roi5', 'roi6', 'roi7', 'roi8', 'roi9', 
                        'roim1', 'roim2', 'roim3', 'roim4', 'roim5', 'roim6', 'roim7', 'roim8', 'roim9','fileid',
                        'uroi1', 'uroi2', 'uroi3', 'uroi4', 'uroi5', 'uroi6', 'uroi7', 'uroi8', 'uroi9',
                        'uroim1', 'uroim2', 'uroim3', 'uroim4', 'uroim5', 'uroim6', 'uroim7', 'uroim8', 'uroim9', 'fileuid']
    
        """ load data without bad column """
        data = pd.read_table(datapath, header=None, delimiter=' ', skiprows=skip_lines, names=header_names)
    
    
    elif det == 'APD':
        """ check for bad lines (starting with '#') and skip them """
        data_raw = pd.read_table(datapath, header=None, delimiter=' ')
        matches = [s for s in data_raw[0] if '#' in s]
        skip_lines = len(matches)+1
        
        header_names = ['mot', 'mot rbk', 'wl1', 'wol1', 'd1', 'wl2', 'wol2', 'd2', 'wl3', 'wol3', 'd3', 'wl4', 'wol4', 'd4', 
                        'wl5', 'wol5', 'd5', 'wl6', 'wol6', 'd6',  
                        'wl1e', 'wol1e', 'd1e', 'wl2e', 'wol2e', 'd2e', 'wl3e', 'wol3e', 'd3e', 'wl4e', 'wol4e', 'd4e',
                        'wl5e', 'wol5e', 'd5e', 'wl6e', 'wol6e', 'd6e']
        data = pd.read_table(datapath, header=None, delimiter=' ', skiprows=skip_lines, names=header_names)
    
    return data
    
    
    
    


def getPilatusImgs(df, set, baseDir='Z:/Data1', imgDir='/Pilatus-Datas/'):
    """
    Fetch the images corresponding to the fileid found in the dataframe df
    """
    
    imgDir = baseDir + imgDir
    imgs = []
    uimgs = []
    
    try:
        imgID = df.fileid
    except AttributeError:
        try:
            imgID = df.fileuid
        except AttributeError:
            try:
                imgID = df.pfileid
            except AttributeError:
                try:
                    imgID = df.pfileuid
                except AttributeError:
                    print('img IDs not found')
        
    if set == 1:
        for jj in range(len(imgID)):
            try:
                image = plt.imread(imgDir + 'PCMO201506/' + 'PCMO201506b_%d.tif' %imgID[jj])
            except IOError:
                try:
                    image = plt.imread(imgDir + 'PCMO201506a/' + 'PCMO201506c_%d.tif' %imgID[jj])
                except IOError:
                    try:
                        image = plt.imread(imgDir + 'PCMO201506b/' + 'PCMO201506c_%d.tif' %imgID[jj])
                    except IOError:
                        print ('No file found')
                        
            imgs.append(image[:,:,0])
                        
    elif set == 2:
        for jj in range(len(imgID)):
            try:
                image = plt.imread(imgDir + 'PCMO201506c/' + 'PCMO201506d_%d.tif' %imgID[jj])
            except IOError:
                try:
                    image = plt.imread(imgDir + 'PCMO201506d/' + 'PCMO201506d_%d.tif' %imgID[jj])
                except IOError:
                        print ('No file found')
                        
            imgs.append(image[:,:,0])
    
    imgs = np.array(imgs)
    
    
    
    """ check for unpumped images and import them if they exist """
    if 'ufileid' in df.columns:
        imgID = df.ufileid
        if set == 1:
            for jj in range(len(imgID)):
                try:
                    image = plt.imread(imgDir + 'PCMO201506/' + 'PCMO201506b_%d.tif' %imgID[jj])
                except IOError:
                    try:
                        image = plt.imread(imgDir + 'PCMO201506a/' + 'PCMO201506c_%d.tif' %imgID[jj])
                    except IOError:
                        try:
                            image = plt.imread(imgDir + 'PCMO201506b/' + 'PCMO201506c_%d.tif' %imgID[jj])
                        except IOError:
                            print ('No file found')
                            
                uimgs.append(image[:,:,0])
                            
        elif set == 2:
            for jj in range(len(imgID)):
                try:
                    image = plt.imread(imgDir + 'PCMO201506c/' + 'PCMO201506d_%d.tif' %imgID[jj])
                except IOError:
                    try:
                        image = plt.imread(imgDir + 'PCMO201506d/' + 'PCMO201506d_%d.tif' %imgID[jj])
                    except IOError:
                            print ('No file found')
                            
                uimgs.append(image[:,:,0])
        
        uimgs = np.array(uimgs)
        
    else:
        uimgs = None
    
    return imgs, uimgs
    
    
    
    


def getPixelsAngles(imgs, pilH, pilV, xdb, ydb, d, roi = None):
    """
    Calculate the angles delta and gamma for all pixels in a ROI, based on the position 
    of the dectector and the direct beam
    input:
        rot: dataframe containing the rotation angle (or other motor)
        imgs: stack of images. one for each motor position imgs(i,j,k), i=image index
        pilH, pilV: Pilatus position
        xdb, ydb: direct beam position(pixel aready taken into account)
        d: distance sample - detector
        img_range: roi of the input images on the detector

    output:
        array gamma, delta corresponding to the pixel in the ROI
        
    the index x (y) always refer to the horizontal (vertical) axis of the picture
    """
    
    """ constants """
    pixSize = 0.172 # pixel size [mm]
    
    if roi is None:
        roi = np.array([ [0,imgs.shape[1]], [0,imgs.shape[2]] ])
    
    delta = np.zeros([ roi[0,1]-roi[0,0], roi[1,1]-roi[1,0] ])
    gamma = np.zeros([ roi[0,1]-roi[0,0], roi[1,1]-roi[1,0] ])
    
    for ny in range(delta.shape[0]): # /!\ in array, the second index is the x (horizontal) axis (line/column VS x/y)
        ypix = roi[1][0]+ny
        y = pilV - pixSize*ypix
        for nx in range(delta.shape[1]): # /!\ in array, the first index is the y (vertical) axis (line/column VS x/y)
            xpix = roi[0][0]+nx
            x = pilH - pixSize*xpix
            
            delta_temp, gamma_temp = tldiff.anglesFromPos(x,y,xdb,ydb,d)
            delta[ny,nx] = delta_temp
            gamma[ny,nx] = gamma_temp
            
    imgs = imgs[:,roi[1,0]:roi[1,1], roi[0,0]:roi[0,1]]
                
    return delta, gamma, imgs
    
    
    


    
def getPixelsAngles2(pixels, pilH, pilV, xdb, ydb, d, pixSize = 0.172):
    """
    Calculate the angles from pixels position
    """
    
    """ constants """
#    pixSize = 0.172 # pixel size [mm]
    
    
    if pixels.ndim == 1:
        xpix = pixels[0]
        x = pilH - pixSize*xpix
        ypix = pixels[1]
        y = pilV - pixSize*ypix
        delta, gamma = tldiff.anglesFromPos(x,y,xdb,ydb,d)

    else:
        
        delta = np.zeros(pixels.shape[0])
        gamma = np.zeros(pixels.shape[0])
        for ii in range(pixels.shape[0]):
            xpix = pixels[ii,0]
            x = pilH - pixSize*xpix
            ypix = pixels[ii,1]
            y = pilV - pixSize*ypix
                        
            delta_temp, gamma_temp = tldiff.anglesFromPos(x,y,xdb,ydb,d)
            delta[ii] = delta_temp
            gamma[ii] = gamma_temp
                
    return delta, gamma
    
    
    

def calibDetAngles(pixels, pilH, pilV, xdb, ydb, d, tilt):
    """
    Calculate angles from detector and pixel position, taking into account an 
    eventual tilt of the detector stage.
    
    input:
        pixel and detector position (pilH, pilV)
        direct beam position (xdb, ydb)
        distance sample - dtetector (d)
        tilt angle of the detector
    """
    pixSize = 0.172 # pixel size [mm]
    tilt = np.deg2rad(tilt)
#    tiltV = np.deg2rad(tiltV)

    xpix = pixels[0]
    x = pilH - pixSize*xpix
    d = d + x*np.sin(tilt)
    x = x*np.cos(tilt)
        
    ypix = pixels[1]
    y = pilV - pixSize*ypix
#    d = d + np.sin(tilt)
#    y = y*np.cos(tilt)
    
    delta, gamma = tldiff.anglesFromPos(x,y,xdb,ydb,d)
    
    return delta, gamma
    
    
    
    
    
def bin_data(df,motor='delay'):
    """
    TAKEN FROM FEL (SACLA) ANALYSIS CODE
    bin data according to motor, without any timing tool correction
    Basically averages the intensity and bkg at each motor position. It is made this way, because 
    it is basically a copy from a timing tool binning function.
    """
    
    # create corrected delay
    df['scan_motor'] = df[motor]
    bin_center = df[motor].unique()
    bin_center = sorted(bin_center)
    bin_size = min(np.diff(bin_center))
    
    bin_edges = np.append(bin_center[0]-0.5*bin_size, bin_center+0.5*bin_size)
    
    df_out = pd.DataFrame(bin_center, columns=[motor])

    binned_int = stats.binned_statistic(df.scan_motor,df.intensity, bins=bin_edges, statistic='mean')
    binned_bkg = stats.binned_statistic(df.scan_motor,df.bkg, bins=bin_edges, statistic='mean')
    df_out['intensity'] = binned_int.statistic
    df_out['bkg'] = binned_bkg.statistic
    binned_int_lon_std = stats.binned_statistic(df.scan_motor,df.intensity, bins=bin_edges, statistic='std')
    binned_bkg_lon_std = stats.binned_statistic(df.scan_motor,df.bkg, bins=bin_edges, statistic='std')
    df_out['intensity_std'] = binned_int_lon_std.statistic
    df_out['bkg_std'] = binned_bkg_lon_std.statistic

#    else: print('No laser ON shots')
#    
#    if len(df_loff) != 0:
#        binned_int_loff = stats.binned_statistic(df_loff.scan_motor,df_loff.intensity, bins=bin_edges, statistic='mean')
#        binned_bkg_loff = stats.binned_statistic(df_loff.scan_motor,df_loff.bkg, bins=bin_edges, statistic='mean')
#        binned_I0_loff = stats.binned_statistic(df_loff.scan_motor,df_loff.I0, bins=bin_edges, statistic = 'mean')
#        df_out['I0_loff'] = binned_I0_loff.statistic
#        df_out['bkg_loff'] = binned_bkg_loff.statistic
#        df_out['intensity_loff'] = binned_int_loff.statistic
#        binned_int_loff_std = stats.binned_statistic(df_loff.scan_motor,df_loff.intensity, bins=bin_edges, statistic='std')
#        binned_bkg_loff_std = stats.binned_statistic(df_loff.scan_motor,df_loff.bkg, bins=bin_edges, statistic='std')
#        binned_I0_loff_std = stats.binned_statistic(df_loff.scan_motor,df_loff.I0, bins=bin_edges, statistic = 'std')
#        df_out['I0_loff_std'] = binned_I0_loff_std.statistic
#        df_out['bkg_loff_std'] = binned_bkg_loff_std.statistic
#        df_out['intensity_loff_std'] = binned_int_loff_std.statistic
#    else: print('No laser OFF shots')
    
    return df_out
    
    
    
def gaussian(x,A,x0,sigma):
    return  A*np.exp(-(x-x0)**2/2/sigma**2)





def halo_correction(int0, int, dl, halo_signal):
    """
    Halo correction for slicing data
    inputs:
        int0: intensity of the signal before t0
        int: intensity of the signal of the data point considered
        dl: dl at which the data point is taken. This is important to know the pumped fraction of the halo
        halo_signal: halo measurement ot the data point
    
    The code works in two steps:
        i) the ratio of the pumped and unpumped halo is calculated
        ii) the correction is composed of two parts; (1-ratio) of the halo (unpumped) and ratio*int/int0 of the halo
            pumped, assuming the same drop than in the signal).
    Ideally one would want to iterate between these steps, as the drop in signal is biased by the halo, but this is not done,
    and probably does not change much to the result.
    """
    
    FWHM = 40
    sigma = FWHM /2/np.sqrt(2*np.log(2))
    
    t0 = dl
    t = np.arange(-200,300,0.2) # [ps] t=0 at t[1000]
    halo = gaussian(t,1,t0,sigma)
    int0_halo = np.trapz(halo,x=t)
    int_halo = np.trapz(halo[1000:], x=t[1000:])
    ratio = int_halo/int0_halo
    
    correction = (1-ratio)*halo_signal + ratio*int/int0*halo_signal
    
    return correction
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    