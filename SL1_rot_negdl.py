# -*- coding: utf-8 -*-
"""
Created on Fri Mar 10 17:25:25 2017

@author: esposito_v
"""

import numpy as np
import sys
import matplotlib.pyplot as plt
import pandas as pd
import glob
from lmfit import minimize, Parameters
from lmfit.models import GaussianModel, VoigtModel, LorentzianModel

sys.path.append("C:/Users/esposito_v/Documents/Python Scripts/FEMTO analysis/")
import functions_FEMTO as fem

plt.close('all')

plot = 1

datapath = 'C:/Users/esposito_v/Documents/PCMO/FEMTO 201506/data/averages2/'

l_area = 0.0520*0.0620 #[cm]
flu_factor = 1./(1000*l_area/np.sin(np.deg2rad(10)))

""" low T """
rNo_all = np.append(3026,np.arange(3029,3041))


if not 'imgs_all' in locals():
    imgs_all = np.zeros([rNo_all.shape[0],80,195,487])
    imgs_diff = np.zeros([rNo_all.shape[0],80,40,44])
        
    sys.path.append('Z:/Data1')
    procList = glob.glob('Z:\Data1\*\proc\proc*.dat')
    procFile = []
    dfList = []
    set = 2
        
    count = 0
    for jj in range(rNo_all.size):
        rNo = rNo_all[jj]
        string = '%d.dat' %rNo
        procFile.append( [s for s in procList if string in s] )
        string = ''.join(procFile[count])
        temp = pd.read_table(string,usecols=[0,1,9,20])
        dfList.append(temp)
        imgs_temp, uimgs_temp =  fem.getPilatusImgs(dfList[count], set)
        imgs_all[jj,:,:,:] = imgs_temp
        count+=1
                
    peakPix = [155,92] # pixels position of the peak [x,y]
    roi = np.array([[peakPix[0]-22,peakPix[0]+22],[peakPix[1]-20,peakPix[1]+20]]) # [[xmin,xmax][ymin,ymax]]
    imgs_all = imgs_all[:,:,roi[1][0]:roi[1][1], roi[0][0]:roi[0][1]] # line = y, column = x