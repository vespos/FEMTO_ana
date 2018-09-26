# -*- coding: utf-8 -*-
"""
Created on Fri Apr 15 15:55:00 2016

@author: esposito_v
"""

import numpy as np
import pandas as pd
import functions_FEMTO as fem
import tools_imgana as tlimg
import tools_diffraction as tldiff
import matplotlib.pyplot as plt

def analyzeCDM_motion(rNo, procList):    
    """ load data and images """
    procFile = []
    set = 2
    for ii in range(len(rNo)):
        str = '%d.dat' %rNo[ii]
        procFile.append( [s for s in procList if str in s] )
        str = ''.join(procFile[ii])
        df = pd.read_table(str,usecols=[0,1,9,20])
        
        if not 'imgs' in locals():
            imgs = fem.getPilatusImgs(df, rNo[ii], set)
        else:
            imgs = imgs + fem.getPilatusImgs(df, rNo[ii], set)
    
    
    
    """ calculate angles for each pixels of each images """    
    E = 7
    alpha = 0.5
    
    pilH = 2.75 # [mm]
    pilV = -47.5 # [mm]
    
    xdb = 21.6 - 414.*0.170 # [mm]
    ydb = -58.03 - 188.*0.17 # [mm]
    
    d = 74
    
    img_range = [12,10]    
    
    delta, gamma, images = fem.getPixelsAngles(imgs,pilH,pilV,xdb,ydb,d, img_range)
    
    E = 7
    alpha = 0.5
    N= np.array([-1,-1,2])
    omega = -63.66
    omega_offset = 0 # offset with respect to the UB matrix
    
    rot = np.array(df['# top rotation']) -omega_offset
    
    a = np.array([5.39,5.40,7.61])
    aa = np.array([90,90,90.07])
    
    U,B = tldiff.UBmat(a, aa, N)
    
    
    """ find center of mass of the peak """
    CDMidx, CDMmaxidx = tlimg.centerOfMass(images)
    
    idxup = np.ceil(CDMidx)
    idxdown = np.floor(CDMidx)
    
    CDM_omega = np.interp(CDMidx[0], np.array([idxdown[0],idxup[0]]), 
                          np.array([rot[idxdown[0]],rot[idxup[0]]] ) )
    omegaMax = rot[CDMmaxidx[0]]
    CDM_delta = np.interp(CDMidx[0], np.array([idxdown[0],idxup[0]]), 
                          np.array([delta[idxdown[1],idxdown[2]],delta[idxup[1],idxup[2]]] ) )
    CDM_gamma = np.interp(CDMidx[0], np.array([idxdown[0],idxup[0]]), 
                          np.array([gamma[idxdown[1],idxdown[2]],gamma[idxup[1],idxup[2]]] ) )
    
    CDMhkl, CDMQ = tldiff.hklFromAngles(E, CDM_delta, CDM_gamma, CDM_omega, alpha, U, B)
    
    images = np.sum(images,axis=0)
    
    return images, CDMhkl, CDM_omega, CDM_delta, CDM_gamma, procFile, omegaMax
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    