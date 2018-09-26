# -*- coding: utf-8 -*-
"""
Created on Thu Jan 26 11:20:35 2017

@author: esposito_v
"""

""" Analysis of the recevery dynamics """

import numpy as np
import sys
import matplotlib.pyplot as plt
import pandas as pd
import glob
from lmfit import Model, Parameters

sys.path.append("C:/Users/esposito_v/Documents/Python Scripts/FEMTO analysis/")
import functions_FEMTO as fem
import fit_function as fitfun

sys.path.append("C:/Users/esposito_v/Documents/Python Scripts/diffraction_imgAna/")
import tools_imgana as imgana


plt.close('all')

datapath = 'C:/Users/esposito_v/Documents/PCMO/FEMTO 201506/data/averages2/'

l_area = 0.0520*0.0620*np.pi/4 #[cm]
flu_factor = 1./(1000*l_area/np.sin(np.deg2rad(10)))

rNo_pilgate = dict()
rNo_timing = dict()

""" pilgate """
rNo_pilgate[0] = np.arange(3098,3107)
rNo_pilgate[1] = np.arange(3108,3124)
rNo_pilgate[2] = np.arange(3124,3135)
rNo_pilgate[3] = np.arange(3136,3144)
rNo_pilgate[4] = np.arange(3150,3162)
rNo_pilgate[5] = np.arange(3083,3095)
rNo_pilgate[6] = np.arange(3146,3150)

""" laser timing """
rNo_timing[0] = np.arange(3183,3191)
rNo_timing[1] = np.arange(3202,3210)
rNo_timing[2] = np.arange(3174,3182)
rNo_timing[3] = np.arange(3211,3217)
rNo_timing[4] = np.arange(3165,3173)
rNo_timing[5] = np.arange(3192,3196)

file_name = dict()
fluence = np.array([20,50,60,70,90,120,140]) * flu_factor

if not 'procList' in locals():        
    sys.path.append('Z:/Data1')
    procList = glob.glob('Z:\Data1\*\proc\proc*.dat')

procFile_pilgate = []
set = 2

if not 'data_pilgate' in locals():
    data_pilgate = dict()
    data_pilgate_ave = dict()
    count = 0
    for ii in range(len(rNo_pilgate)):
        print('pilgate ii = %d' % ii)
        data_pilgate[ii] = pd.DataFrame()
    
        for jj in range(rNo_pilgate[ii].size):
            rNo = rNo_pilgate[ii][jj]
            string = '%d.dat' %rNo
            procFile_pilgate.append( [s for s in procList if string in s] )
            string = ''.join(procFile_pilgate[count])
            temp = pd.read_table(string,usecols=[0,1,9,18,20])
            imgs_temp, uimgs_temp = fem.getPilatusImgs(temp, set)
    
            int_roi = np.zeros(imgs_temp.shape[0])
            int_bkg = np.zeros(imgs_temp.shape[0])
            total_int = np.zeros(131)
            for kk in range(imgs_temp.shape[0]):
    #            roi = [[0,178],[0,212]] # [[xmin, xmax], [ymin, ymax]]
                roi = [[94-20,94+20],[165-25,165+25]] # [[xmin, xmax], [ymin, ymax]]
                # note: the import routine create an array from the images, which invert the axis
                bkgroi = np.add(roi , [[-30,30],[-30,30]] )
                
                temp_int_roi, temp_int_bkg = imgana.roi_bkgRoi(imgs_temp[kk], roi, bkgroi)
                int_roi[kk] = temp_int_roi
                int_bkg[kk] = temp_int_bkg
                
    #            if kk % 10 == 0:
    #                plt.figure(1000)
    #                plt.subplot(4,4,kk/10+1)
    #                plt.imshow(imgs_temp[ kk ])
            
            temp['intensity'] = int_roi
            temp['bkg'] = int_bkg
            data_pilgate[ii] = data_pilgate[ii].append(temp)
            count+=1
        
for ii in range(len(rNo_pilgate)):
    data_pilgate_ave[ii] = fem.bin_data(data_pilgate[ii], motor=data_pilgate[ii].columns[0])
    data_pilgate_ave[ii][data_pilgate_ave[ii].columns[0]] = data_pilgate_ave[ii][data_pilgate[ii].columns[0]]/1000
    data_pilgate_ave[ii]['intensity-bkg'] = data_pilgate_ave[ii]['intensity'] - data_pilgate_ave[ii]['bkg']
    data_pilgate_ave[ii]['normalized'] = data_pilgate_ave[ii]['intensity-bkg'] / np.mean(data_pilgate_ave[0]['intensity-bkg'][-20:])
    data_pilgate_ave[ii]['normalized_2'] = data_pilgate_ave[ii]['intensity-bkg'] / np.mean(data_pilgate_ave[ii]['intensity-bkg'][-20:])
#    plt.figure(ii)
#    plt.plot(data_ave[ii][data[ii].columns[0]], data_ave[ii]['intensity'])
#    plt.plot(data[ii][data[ii].columns[0]],data[ii][data[ii].columns[5]], 'o')
    
#    plt.figure(100)
#    plt.title('intensity')
#    plt.plot(data_pilgate_ave[ii][data_pilgate_ave[ii].columns[0]], data_pilgate_ave[ii]['intensity'])
    plt.figure(101)
    plt.title('background corrected')
#    plt.plot(data_pilgate_ave[ii][data_pilgate_ave[ii].columns[0]], data_pilgate_ave[ii]['normalized'], label=('%d' %ii))
    plt.plot(data_pilgate_ave[ii][data_pilgate_ave[ii].columns[0]], data_pilgate_ave[ii]['normalized_2'], label=('%d' %ii))
    plt.xlabel('Time [us]')
    plt.legend()
    
    
    
procFile_timing = []
set = 2    
    
if not 'data_timing' in locals():
    data_timing = dict()
    data_timing_ave = dict()
    count = 0
    for ii in range(len(rNo_timing)):
        print('timing ii = %d' % ii)
        data_timing[ii] = pd.DataFrame()
    
        for jj in range(rNo_timing[ii].size):
            rNo = rNo_timing[ii][jj]
            string = '%d.dat' %rNo
            procFile_timing.append( [s for s in procList if string in s] )
            string = ''.join(procFile_timing[count])
            temp = pd.read_table(string,usecols=[0,1,9,18,20])
            imgs_temp, uimgs_temp = fem.getPilatusImgs(temp, set)

            int_roi = np.zeros(imgs_temp.shape[0])
            int_bkg = np.zeros(imgs_temp.shape[0])
            total_int = np.zeros(131)
            for kk in range(imgs_temp.shape[0]):
    #            roi = [[0,178],[0,212]] # [[xmin, xmax], [ymin, ymax]]
                roi = [[94-20,94+20],[165-25,165+25]] # [[xmin, xmax], [ymin, ymax]]
                # note: the import routine create an array from the images, which invert the axis
                bkgroi = np.add(roi , [[-30,30],[-30,30]] )
                
                temp_int_roi, temp_int_bkg = imgana.roi_bkgRoi(imgs_temp[kk], roi, bkgroi)
                int_roi[kk] = temp_int_roi
                int_bkg[kk] = temp_int_bkg
                
    #            if kk % 10 == 0:
    #                plt.figure(1000)
    #                plt.subplot(4,4,kk/10+1)
    #                plt.imshow(imgs_temp[ kk ])
            
            temp['intensity'] = int_roi
            temp['bkg'] = int_bkg
            data_timing[ii] = data_timing[ii].append(temp)
            count+=1
            
ratio = []
ratio2 = []
negdl = []
for ii in range(len(rNo_timing)):
    data_timing_ave[ii] = fem.bin_data(data_timing[ii], motor=data_timing[ii].columns[0])
    data_timing_ave[ii][data_timing_ave[ii].columns[0]] = 3867.45 - data_timing_ave[ii][data_timing[ii].columns[0]]
    data_timing_ave[ii]['intensity-bkg'] = data_timing_ave[ii]['intensity'] - data_timing_ave[ii]['bkg']
    data_timing_ave[ii]['normalized'] = data_timing_ave[ii]['intensity-bkg'] / np.mean(data_timing_ave[0]['intensity-bkg'][0:2])
    data_timing_ave[ii]['normalized_2'] = data_timing_ave[ii]['intensity-bkg'] / np.mean(data_timing_ave[ii]['intensity-bkg'][-3:])
#    plt.figure(ii)
#    plt.plot(data_ave[ii][data[ii].columns[0]], data_ave[ii]['intensity'])
#    plt.plot(data[ii][data[ii].columns[0]],data[ii][data[ii].columns[5]], 'o')
    
#    plt.figure(103)
#    plt.title('intensity')
#    plt.plot(data_timing_ave[ii][data_timing_ave[ii].columns[0]], data_timing_ave[ii]['intensity'])
    plt.figure(104)
    plt.title('background corrected')
#    plt.plot(data_timing_ave[ii][data_timing_ave[ii].columns[0]], data_timing_ave[ii]['normalized'])
    plt.plot(data_timing_ave[ii][data_timing_ave[ii].columns[0]], data_timing_ave[ii]['normalized_2'])
    plt.xlabel('Time [ns]')
    
    ratio.append( (np.mean(data_timing_ave[ii]['normalized'][:5]) - np.mean(data_timing_ave[ii]['normalized'][-3:]))\
        /np.mean(data_timing_ave[ii]['normalized'][-3:]) )
    ratio2.append( (np.mean(data_pilgate_ave[ii]['normalized'][-5:]) - np.mean(data_timing_ave[ii]['normalized'][-3:]))\
        /np.mean(data_timing_ave[ii]['normalized'][-3:]) )
    negdl.append(np.mean(data_timing_ave[ii]['normalized'][-3:]))
    
plt.figure(105)
plt.plot(fluence[:6], ratio,'o',label='ratio ns')
plt.plot(fluence[:6], ratio2,'o',label='ratio um')
plt.plot(fluence[:6], np.array(negdl)-1,'o',label='negative delay drop')
plt.legend()
plt.legend(loc=0,numpoints=1)


""" Fit of a streched exponential """
plt.figure()

model = Model(fitfun.stretched_exp)
params = model.make_params()
params['A'].value = 0.5
params['A'].min = 0
#params['A'].max = 1
params['tau'].value = 1000
params['tau'].min = 0
params['beta'].value = 0.6
params['beta'].min = 0

xx = dict()
yfit = dict()
results = dict()
A = np.zeros(len(data_timing_ave))
tau = np.zeros(len(data_timing_ave))
beta = np.zeros(len(data_timing_ave))

for ii in range(len(data_timing_ave)):
    xx[ii] = np.append( data_timing_ave[ii][data_timing_ave[ii].columns[0]][:-7],  data_pilgate_ave[ii][data_pilgate_ave[ii].columns[0]][2:]*1000)
    yy = np.append( data_timing_ave[ii]['normalized'][:-7], data_pilgate_ave[ii]['normalized'][2:] )
    
    plt.plot(xx[ii],yy,'o')
    plt.xscale('log')
#    plt.yscale('log')
    
    results[ii] = model.fit(yy, params, t=xx[ii])
    yfit[ii] = model.eval(results[ii].params, t=xx[ii])
    A[ii] = results[ii].params['A'].value
    tau[ii] = results[ii].params['tau'].value
    beta[ii] = results[ii].params['beta'].value
      
    print(results[ii].fit_report())
    print('\n')
    
    plt.plot(xx[ii],yfit[ii])

    





















