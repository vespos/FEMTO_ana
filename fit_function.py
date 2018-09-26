# -*- coding: utf-8 -*-
"""
Created on Tue Jun 21 09:16:09 2016

@author: esposito_v

"""


import numpy as np
from scipy import optimize
from scipy.special import erf
from time import sleep
import matplotlib.pyplot as plt
import lmfit



""" FITTING METHODS """
 
def fit_leastsq(p0, datax, datay, function):
    """
    Uses optimize.leastsq method to fit curve to data
    """

    errfunc = lambda p, x, y: function(x,p) - y

    pfit, pcov, infodict, errmsg, success = \
        optimize.leastsq(errfunc, p0, args=(datax, datay), \
                          full_output=1, epsfcn=0.0001)

    if (len(datay) > len(p0)) and pcov is not None:
        s_sq = (errfunc(pfit, datax, datay)**2).sum()/(len(datay)-len(p0))
        pcov = pcov * s_sq
    else:
        pcov = np.inf

    error = [] 
    for i in range(len(pfit)):
        try:
          error.append(np.absolute(pcov[i][i])**0.5)
        except:
          error.append( 0.00 )
    pfit_leastsq = pfit
    perr_leastsq = np.array(error) 
    return pfit_leastsq, perr_leastsq
    



    
def fit_curvefit(p0, datax, datay, function, yerr=None, **kwargs):
    """
    Uses optimize.curvefit to fit curve to data. 
    Can take yerr into account.
    Minimizes P = np.sum( ((f(xdata, p0) - ydata) / sigma)**2 )
    """

    pfit, pcov = \
         optimize.curve_fit(function,datax,datay,p0=p0,\
                            sigma=yerr, **kwargs)
    error = []
    for i in range(len(pfit)):
        try:
          error.append(np.absolute(pcov[i][i])**0.5)
        except:
          error.append( 0.00 )
    pfit_curvefit = pfit
    perr_curvefit = np.array(error)
    return pfit_curvefit, perr_curvefit
    
    
    
    
    
def fit_bootstrap(p0, datax, datay, function, yerr_systematic = 0.0, nboot = 1000):
    """
    Bootstrap method to evaluate errors on fitting parameters, taking into account y-error on the data points.
    It basically evaluate the parameters for nboot randomly generated sets of data
    and compute the mean and variance of the fitted parameters. It is usually the best method to
    use, but might require a lot more time / computing power.
    As usually, the fit function 'function' should be defined as f(x, [params]) or f(x, param1, param2, ...)
    """

    errfunc = lambda p, x, y: function(x,*p) - y

    # Fit first time
    pfit, perr = optimize.leastsq(errfunc, p0, args=(datax, datay), full_output=0)


    # Get the stdev of the residuals
    residuals = errfunc(pfit, datax, datay)
    sigma_res = np.std(residuals)

    sigma_err_total = np.sqrt(sigma_res**2 + yerr_systematic**2)

    # nboot random data sets are generated and fitted
    ps = []
    for i in range(nboot):

        randomDelta = np.random.normal(0., sigma_err_total, len(datay))
        randomdataY = datay + randomDelta

        randomfit, randomcov = \
            optimize.leastsq(errfunc, p0, args=(datax, randomdataY),\
                             full_output=0)

        ps.append(randomfit) 

    ps = np.array(ps)
    mean_pfit = np.mean(ps,0)

    # You can choose the confidence interval that you want for your
    # parameter estimates: 
    Nsigma = 1. # 1sigma gets approximately the same as methods above
                # 1sigma corresponds to 68.3% confidence interval
                # 2sigma corresponds to 95.44% confidence interval
    err_pfit = Nsigma * np.std(ps,0) 

    pfit_bootstrap = mean_pfit
    perr_bootstrap = err_pfit
    return pfit_bootstrap, perr_bootstrap 
    
    












""" SHORT FIT FUNCTIONS """
def moments(x,y):
    mean = np.sum(x*y)/np.sum(y)
    sigma = 2*np.sum( (x-mean)**2*y )/y.sum()
    return mean, sigma

def gaussian_diff(x, A1, x10, sigma1, A2, x20, sigma2):
    gauss1 = A1*np.exp(-(x-x10)**2/2/sigma1**2)
    gauss2 = A2*np.exp(-(x-x20)**2/2/sigma2**2)
    
    return gauss2-gauss1
    

def gaussian(x,A,x0,sigma):
    return  A*np.exp(-(x-x0)**2/2/sigma**2)


def gaussian2(x,p):
    A = p[0]
    x0 = p[1]
    sigma = p[2]
    return  A*np.exp(-(x-x0)**2/2/sigma**2)
    
    
def gaussian_bkg(x,A,x0,sigma,bkg):
    return  A*np.exp(-(x-x0)**2/2/sigma**2)+bkg
    

def lorentzian(x,A,x0,gamma):
    return A*gamma**2/( (x-x0)**2+gamma**2 )
    
    
def step_fct(x,A,x0,FWHM,offset):
    sigma = FWHM/2./np.sqrt(2.*np.log(2));
    y = 0.5*(erf((x-x0)/sigma) + 1)
    y = 1-A*y + (offset-1)
    return y
    
    
def double_peak_Gauss(x,A1,sigma1,A2,sigma2):
    x01 = 266
    x02 = 266.27
    return A1*np.exp(-(x-x01)**2/2/sigma1**2) + A2*np.exp(-(x-x02)**2/2/sigma2**2)

    
#def double_peak_Lorentz(x,A1,gamma1,A2,gamma2):
#    x01 = 266
#    x02 = 266.28
#    return A1*gamma1**2/( (x-x01)**2+gamma1**2 ) + A2*gamma2**2/( (x-x02)**2+gamma2**2 )
    
    
def stretched_exp(t,A,tau,beta):
    return 1-A*np.exp(-(t/tau)**beta)
    
    
def incoh(params, t, data=None, eps=None, sigma=0.07):
    # unpack parameters:
    #  extract .value attribute for each parameter
    parvals = params.valuesdict()
    A_fast = parvals['A_fast']
    tau_fast = parvals['tau_fast']
    A_slow = parvals['A_slow']
    tau_slow = parvals['tau_slow']
    t0 = parvals['t0']
    
    y1 = A_fast * np.exp(-(t-t0)/tau_fast)
    y2 = A_slow * (1 - np.exp(-(t-t0)/tau_fast)) * np.exp(-(t-t0)/tau_slow)
    step = 0.5*(erf((t-t0)/sigma) + 1)
    yfit = step*(y1+y2)
    
#    t_conv = t[t>-0.6]
#    t_conv = t_conv[t_conv<0.6]
#    gauss_conv = np.exp(-np.power((t_conv)/sigma, 2.)/2)
#    yfit = np.convolve(yfit, gauss_conv, 'same')
    
    if data is None:
        return yfit
    if eps is None:
        return (yfit - data)
    return (yfit - data)/eps
    
    
def incoh_offset(params, t, data=None, eps=None, sigma=0.07):
    # unpack parameters:
    #  extract .value attribute for each parameter
    parvals = params.valuesdict()
    A_fast = parvals['A_fast']
    tau_fast = parvals['tau_fast']
    A_slow = parvals['A_slow']
    tau_slow = parvals['tau_slow']
    offset = parvals['offset']
    t0 = parvals['t0']
    
    y1 = A_fast * np.exp(-(t-t0)/tau_fast)
    y2 = A_slow * (1 - np.exp(-(t-t0)/tau_fast)) * np.exp(-(t-t0)/tau_slow)
    step = 0.5*(erf((t-t0)/sigma) + 1)
    yfit = step*(y1+y2+offset)
    
    if data is None:
        return yfit
    if eps is None:
        return (yfit - data)
    return (yfit - data)/eps
    

def displacive(params, t, data=None, eps=None ,t0=0, sigma=0.1):
    # unpack parameters:
    #  extract .value attribute for each parameter
    parvals = params.valuesdict()
    A = parvals['A']
    tau = parvals['tau']
    freq = parvals['freq']
    A_fast = parvals['A_fast']

    yfit = A*(1-np.cos(2*np.pi*freq*t)*np.exp(-t/tau))+A_fast

    step = 0.5*(erf((t-t0)/sigma) + 1)
#    yfit[t<0] = 0
    yfit = step*yfit
    
    if data is None:
        return yfit
    if eps is None:
        return (yfit - data)
    return (yfit - data)/eps


def incoh_and_coh_t0(params, t, data=None, eps=None, sigma=0.09):
    # unpack parameters:
    #  extract .value attribute for each parameter
    parvals = params.valuesdict()
    A_fast = parvals['A_fast']
    tau_fast = parvals['tau_fast']
    A_slow = parvals['A_slow']
    tau_slow = parvals['tau_slow']
    A_ph = parvals['A_ph']
    freq = parvals['freq']
    tau_ph = parvals['tau_ph']
    t0 = parvals['t0']

    
    y1 = A_fast * np.exp(-(t-t0)/tau_fast)
    y2 = A_slow * (1 - np.exp(-(t-t0)/tau_fast)) * np.exp(-(t-t0)/tau_slow)
    y3 = A_ph * np.cos(2*np.pi*freq*t)*np.exp(-t/tau_ph)
    step = 0.5*(erf((t-t0)/sigma) + 1)
    yfit = step*(y1+y2+y3)
    
#    t_conv = t[t>-0.6]
#    t_conv = t_conv[t_conv<0.6]
#    gauss_conv = np.exp(-np.power((t_conv)/sigma, 2.)/2)
#    yfit = np.convolve(yfit, gauss_conv, 'same')
    
    if data is None:
        return yfit
    if eps is None:
        return (yfit - data)
    return (yfit - data)/eps


def incoh_and_coh(params, t, data=None, eps=None, sigma=0.09):
    t0=0
    # unpack parameters:
    #  extract .value attribute for each parameter
    parvals = params.valuesdict()
    A_fast = parvals['A_fast']
    tau_fast = parvals['tau_fast']
    A_slow = parvals['A_slow']
    tau_slow = parvals['tau_slow']
    A_ph = parvals['A_ph']
    freq = parvals['freq']
    tau_ph = parvals['tau_ph']

    omega = np.sqrt( (2*np.pi*freq)**2-(1/tau_ph)**2 )
    
    y1 = A_fast * np.exp(-(t-t0)/tau_fast)
    y2 = A_slow * (1 - np.exp(-(t-t0)/tau_fast)) * np.exp(-(t-t0)/tau_slow)
    y3 = A_ph * np.cos(omega*t)*np.exp(-t/tau_ph)
    step = 0.5*(erf((t-t0)/sigma) + 1)
    yfit = step*(y1+y2+y3)
    
#    t_conv = t[t>-0.6]
#    t_conv = t_conv[t_conv<0.6]
#    gauss_conv = np.exp(-np.power((t_conv)/sigma, 2.)/2)
#    yfit = np.convolve(yfit, gauss_conv, 'same')
    
    if data is None:
        return yfit
    if eps is None:
        return (yfit - data)
    return (yfit - data)/eps


def coh(params, t, data=None, eps=None, sigma=0.09):
    # unpack parameters:
    #  extract .value attribute for each parameter
    parvals = params.valuesdict()
    A_ph = parvals['A_ph']
    freq = parvals['freq']
    tau_ph = parvals['tau_ph']
    
    omega = np.sqrt( (2*np.pi*freq)**2-(1/tau_ph)**2 )

    yfit = A_ph * np.cos(omega*t)*np.exp(-t/tau_ph)
    
#    t_conv = t[t>-0.6]
#    t_conv = t_conv[t_conv<0.6]
#    gauss_conv = np.exp(-np.power((t_conv)/sigma, 2.)/2)
#    yfit = np.convolve(yfit, gauss_conv, 'same')
    
    if data is None:
        return yfit
    if eps is None:
        return (yfit - data)
    return (yfit - data)/eps


def coh_phase(params, t, data=None, eps=None, sigma=0.09):
    # unpack parameters:
    #  extract .value attribute for each parameter
    parvals = params.valuesdict()
    A_ph = parvals['A_ph']
    freq = parvals['freq']
    tau_ph = parvals['tau_ph']
    phase = parvals['phase']


    yfit = A_ph * np.cos(2*np.pi*freq*t+phase)*np.exp(-t/tau_ph)
    
#    t_conv = t[t>-0.6]
#    t_conv = t_conv[t_conv<0.6]
#    gauss_conv = np.exp(-np.power((t_conv)/sigma, 2.)/2)
#    yfit = np.convolve(yfit, gauss_conv, 'same')
    
    if data is None:
        return yfit
    if eps is None:
        return (yfit - data)
    return (yfit - data)/eps




def order_param(params, fluence, data=None, eps=None):
    
    parvals = params.valuesdict()
    fc = parvals['fc']
    gamma = parvals['gamma']
    scale = parvals['scale']

    z0 = 48 # penetration depth
    d = 40. # sample thickness [nm]
    layers = 40 # number of layers
    dlayer = d/layers
    T = 0.94
    
    F=0
    
    for ii in range(layers):
        f_layer_top = fluence*T*np.exp(-ii*dlayer/z0)
        eta = np.sqrt(1-f_layer_top/fc)**gamma
        eta[eta!=eta] = 0
        F = F+eta

    F = 1/layers * F
    yfit = scale * F**2 + (1-scale)
#    yfit = F**2
    
    if data is None:
        return yfit
    if eps is None:
        return (yfit - data)
    return (yfit - data)/eps


def double_exp(params, t, data=None, eps=None, sigma=0.08):
    # unpack parameters:
    #  extract .value attribute for each parameter
    parvals = params.valuesdict()
    A_fast = parvals['A_fast']
    tau_fast = parvals['tau_fast']
    A_slow = parvals['A_slow']
    tau_slow = parvals['tau_slow']
    t0 = parvals['t0']
    
    y1 = A_fast * np.exp(-(t-t0)/tau_fast)
    y2 = A_slow * np.exp(-(t-t0)/tau_slow)
    step = 0.5*(erf((t-t0)/sigma) + 1)
    yfit = step*(y1+y2)
    
#    t_conv = t[t>-0.6]
#    t_conv = t_conv[t_conv<0.6]
#    gauss_conv = np.exp(-np.power((t_conv)/sigma, 2.)/2)
#    yfit = np.convolve(yfit, gauss_conv, 'same')
    
    if data is None:
        return yfit
    if eps is None:
        return (yfit - data)
    return (yfit - data)/eps


def exp_offset(params, t, data=None, eps=None, sigma=0.075):
    # unpack parameters:
    #  extract .value attribute for each parameter
    if isinstance(params, lmfit.parameter.Parameters):
        parvals = params.valuesdict()
        A_fast = parvals['A_fast']
        tau_fast = parvals['tau_fast']
        offset = parvals['offset']
        t0 = parvals['t0']
    else:
        A_fast = params[0]
        tau_fast = params[1]
        offset = params[2]
        t0 = params[3]
    
    y1 = A_fast * np.exp(-(t-t0)/tau_fast)
    step = 0.5*(erf((t-t0)/sigma) + 1)
    yfit = step*(y1+offset)
    
#    t_conv = t[t>-0.6]
#    t_conv = t_conv[t_conv<0.6]
#    gauss_conv = np.exp(-np.power((t_conv)/sigma, 2.)/2)
#    yfit = np.convolve(yfit, gauss_conv, 'same')
    
    if data is None:
        return yfit
    if eps is None:
        return (yfit - data)
    return (yfit - data)/eps




def double_exp_offset(params, t, data=None, eps=None, sigma=0.08):
    # unpack parameters:
    #  extract .value attribute for each parameter
    parvals = params.valuesdict()
    A_fast = parvals['A_fast']
    tau_fast = parvals['tau_fast']
    A_slow = parvals['A_slow']
    tau_slow = parvals['tau_slow']
    offset = parvals['offset']
    t0 = parvals['t0']
    
    y1 = A_fast * np.exp(-(t-t0)/tau_fast)
    y2 = A_slow * np.exp(-(t-t0)/tau_slow)
    step = 0.5*(erf((t-t0)/sigma) + 1)
    yfit = step*(y1+y2+offset)
    
#    t_conv = t[t>-0.6]
#    t_conv = t_conv[t_conv<0.6]
#    gauss_conv = np.exp(-np.power((t_conv)/sigma, 2.)/2)
#    yfit = np.convolve(yfit, gauss_conv, 'same')
    
    if data is None:
        return yfit
    if eps is None:
        return (yfit - data)
    return (yfit - data)/eps



def order_param_opt(fluence, fc, gamma, scale):
    
#    parvals = paramss.valuesdict()
#    fc = parvals['fc']
#    gamma = parvals['gamma']
#    scale = parvals['scale']

    T = 0.85 # optical
    z0 = 60 # penetration depth
    d = 40. # sample thickness [nm]
    nlayers = 40 # number of layers
    dlayer = d/nlayers

    F=0
    
    nc = fc
    
    norm = 0
    for ii in range(nlayers):
        flu_top = fluence*T*np.exp(-ii*dlayer/z0)
        flu_bottom = fluence*T*np.exp(-(ii+1)*dlayer/z0)
        n0 = (flu_top-flu_bottom)/(dlayer*1E-7)/1000
        eta = (1-n0/nc)**gamma

#        f_layer_top = fluence*T*np.exp(-ii*dlayer/z0)
#        eta = (1-f_layer_top/fc)**gamma

        eta[eta!=eta] = 0
#        eta = eta**2

        weight = np.exp(-ii*dlayer/z0) # weight for the probe
        eta = eta * weight
        norm = norm + weight
        
        F = F + eta

#    F = 1/nlayers*F
    F = 1/norm*F
    yfit = scale * F**2 + (1-scale)
#    yfit = scale * F + (1-scale)

    return yfit




def order_param_xray(fluence, fc, gamma, scale):
    
#    parvals = paramss.valuesdict()
#    fc = parvals['fc']
#    gamma = parvals['gamma']
#    scale = parvals['scale']

    T = 0.75 # pump (x-ray)
    z0 = 60*np.cos(np.deg2rad(25)) # penetration depth
    d = 40. # sample thickness [nm]
    nlayers = 40 # number of layers
    dlayer = d/nlayers

    F=0
    
    nc = fc
    
    for ii in range(nlayers):
        flu_top = fluence*T*np.exp(-ii*dlayer/z0)
        flu_bottom = fluence*T*np.exp(-(ii+1)*dlayer/z0)
        n0 = (flu_top-flu_bottom)/(dlayer*1E-7)/1000

#        f_layer_top = fluence*T*np.exp(-ii*dlayer/z0)
#        eta = (1-f_layer_top/fc)**gamma
        eta = (1-n0/nc)**gamma
        eta[eta!=eta] = 0
#        eta = eta**2
        F = F+eta

    F = 1/nlayers * F
    yfit = scale * F**2 + (1-scale)
#    yfit = scale * F + (1-scale)
    
#    yfit = 0.75 * F**2 + (1-0.75)

    return yfit
    
    
    

    
    
def order_param_xray_time(t, fluences, nc, gamma, tau, t0):
    """ fit the order parameter model from the Nat Mat paper """
    
    T = 0.75 # x-ray
    z0 = 60*np.cos(np.deg2rad(25)) # penetration depth
    d = 40. # sample thickness [nm]
    nlayers = 40 # number of layers
    dlayer = d/nlayers
    
    t_exp = t
    t_fit = np.arange(-5,15,0.05)
#    t_fit = np.arange(t[0]-4,t[-1]+4,0.01)
    
    for fluence in fluences:
        eta = 0
        for ii in range(nlayers):
            flu_top = fluence*T*np.exp(-ii*dlayer/z0)
            flu_bottom = fluence*T*np.exp(-(ii+1)*dlayer/z0)
            n0 = (flu_top-flu_bottom)/(dlayer*1E-7)/1000 # absorbed energy density J/cm^3
            
            tau_pump = 0.05
            exc = 0.5+erf( (t_fit-t0) /tau_pump*np.sqrt(4*np.log(2)))/2
    
            if n0 < nc:
                alpha = 1-(1-exc*n0/nc)**(2*gamma)
#                tau2 = tau/np.sqrt(1-n0/nc)
                tau2 = tau
                n = ((exc*n0-alpha*nc)*np.exp(-(t_fit-t0)/tau2) + alpha*nc) /nc
            else:
                n = exc
            
            eta_layer = np.sqrt(1-n)
            eta_layer[eta_layer!=eta_layer] = 0
#            eta_layer = eta_layer**2
            eta = eta+1/nlayers*eta_layer
        
        if 'eta_square' in locals():
            eta_square = np.vstack((eta_square, eta**2))
#            eta_square = np.vstack((eta_square, eta))
        else:
            eta_square = np.asarray(eta**2)
#            eta_square = np.asarray(eta)
            
#    plt.figure('FIT')
#    plt.plot(t,eta_square.transpose())
#    sleep(0.5)
#    plt.close('FIT')

    print('nc = ' + str(nc) + ', gamma = ' + str(gamma) + ', tau = ' + str(tau))
        
    eta_square = eta_square.transpose() 
    
    eta_square = 0.75 * eta_square + (1-0.75)
    
    
    """ convolution with time resolution """
    output = np.zeros((t_exp.shape[0],eta_square.shape[1]))
    FWHM = 1.3
    sigma = FWHM/(2*np.sqrt(2*np.log(2)))
    t_conv = np.arange(-3*FWHM,3*FWHM,0.05)
    t_res = gaussian(t_conv,1,0,sigma) / np.sum(gaussian(t_conv,1,0,sigma))
    for ii in range(eta_square.shape[1]):
        eta_square[:,ii] = np.convolve(eta_square[:,ii], t_res, mode='same')
        output[:,ii] = np.interp(t_exp,t_fit,eta_square[:,ii])
    
#    output = eta_square
        
    return output
    
    
    
    
    
    

def order_param_optical_time(t, fluences, nc, gamma, tau, tau_exp, t0):
    """ fit the order parameter model from the Nat Mat paper """
    
    T = 0.82 # optical
    z0 = 60 # penetration depth
    d = 40. # sample thickness [nm]
    nlayers = 40 # number of layers
    dlayer = d/nlayers
    
    for fluence in fluences:
        eta = 0
        for ii in range(nlayers):
            flu_top = fluence*T*np.exp(-ii*dlayer/z0)
            flu_bottom = fluence*T*np.exp(-(ii+1)*dlayer/z0)
            n0 = (flu_top-flu_bottom)/(dlayer*1E-7)/1000 # absorbed energy density J/cm^3
            
            tau_pump = 0.8
            exc = 0.5+erf( (t-t0) /tau_pump*np.sqrt(4*np.log(2)))/2
    
            if n0 < nc:
                alpha = 1-(1-exc*n0/nc)**(2*gamma)
                tau2 = tau/(1-n0/nc)**tau_exp
#                tau2 = tau
                n = ((exc*n0-alpha*nc)*np.exp(-(t-t0)/tau2) + alpha*nc) /nc
            else:
                n = exc
            
            eta_layer = np.sqrt(1-n)
            eta_layer[eta_layer!=eta_layer] = 0
            eta = eta+1/nlayers*eta_layer
        
        if 'eta_square' in locals():
            eta_square = np.vstack((eta_square, eta**2))
        else:
            eta_square = np.asarray(eta**2)
            
#    plt.figure('FIT')
#    plt.plot(t,eta_square.transpose())
#    sleep(0.5)
#    plt.close('FIT')

    print('nc = ' + str(nc) + ', gamma = ' + str(gamma) + ', tau = ' + str(tau))
        
    eta_square = eta_square.transpose() 
    
    eta_square = 0.1 * eta_square + (1-0.1)
            
    return eta_square-1





















