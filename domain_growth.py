# -*- coding: utf-8 -*-
"""
Created on Tue Jan 17 11:54:24 2017

@author: esposito_v
"""

"""
Fit of the domain growth dynamics to the peak width 
First run the file that analyzes the 002 time dependent behavior to get the fit parameters
Fit of the form 1/sqrt(t) based on PRB 50 (1994) by Long-Qing Chen
"""


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


time_fit = time[4:]
corr_length = 1/popt_rot[4:10,2]

growth_fct = lambda t, A, offset: A*t**0.5+offset
#growth_fct = lambda t, A: A*t**0.5

p0 = [1,0]
#p0 = 1

popt_growth, perr_growth = fitfct.fit_curvefit(p0, time_fit, corr_length, growth_fct)

time_plot = np.arange(0,100,1)
y_fit = growth_fct(time_plot, popt_growth[0], popt_growth[1])
#y_fit = growth_fct(time_plot, popt_growth[0])

plt.figure()
plt.title('Sigma')
plt.errorbar(time,1/popt_rot[:10,2],yerr=perr_rot[:10,2]/popt_rot[:10,2]**2,fmt='o')
plt.plot(time_plot, y_fit)