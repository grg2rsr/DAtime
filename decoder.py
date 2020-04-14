import sys,os
from time import time
import pickle
from tqdm import tqdm
from copy import copy
import scipy as sp
import numpy as np
import pandas as pd
import neo
import elephant as ele
import quantities as pq
from pathlib import Path

from helpers import *

from scipy.interpolate import interp1d


"""
 
   ____  _____ ____ ___  ____  _____ ____  
  |  _ \| ____/ ___/ _ \|  _ \| ____|  _ \ 
  | | | |  _|| |  | | | | | | |  _| | |_) |
  | |_| | |__| |__| |_| | |_| | |___|  _ < 
  |____/|_____\____\___/|____/|_____|_| \_\
                                           
 
"""

def calc_Prt_direct(Rates_train, tt_dc, rr):
    nUnits = Rates_train.shape[0]
    nTrials = Rates_train.shape[1]
    tvec = Rates_train[0,0].times.rescale('s').magnitude.flatten()

    # get the rates out of neo objects
    R_train = sp.zeros((tvec.shape[0],nUnits,nTrials))
    for u in range(nUnits):
        for j in range(nTrials):
            R_train[:,u,j] = Rates_train[u,j].magnitude.flatten()

    Prt = sp.zeros((tt_dc.shape[0],rr.shape[0],nUnits))
    for u in range(nUnits):
        for i,t in enumerate(tt_dc):
            start_ind = sp.argmin(sp.absolute(tvec - t))
            stop_ind = sp.argmin(sp.absolute(tvec - (t + dt)))
            samples = [sp.average(R_train[start_ind:stop_ind,u,j],axis=0) for j in range(nTrials)]
            Prt[i,:-1,u] = sp.histogram(samples,bins=rr)[0]
    return Prt

# def calc_Prt_scipy(Rates_train, tt_dc, bandwidth=None):
#     """ calculates a KDE for rates from all trail in Rates_train
#     for each timepoint in tt_dc (in unit seconds) """

#     nUnits = Rates_train.shape[0]
#     nTrials = Rates_train.shape[1]
#     tvec = Rates_train[0,0].times.rescale('s').magnitude.flatten()

#     # get the rates out of neo objects
#     R_train = sp.zeros((tvec.shape[0],nUnits,nTrials))
#     for u in range(nUnits):
#         for j in range(nTrials):
#             R_train[:,u,j] = Rates_train[u,j].magnitude.flatten()

#     # fill Prt
#     Prt = sp.zeros((tt_dc.shape[0],rr.shape[0],nUnits))
#     for u in range(nUnits):
#         for i,t in enumerate(tt_dc):
#             start_ind = sp.argmin(sp.absolute(tvec - t))
#             stop_ind = sp.argmin(sp.absolute(tvec - (t + dt)))
#             samples = sp.array([sp.average(R_train[start_ind:stop_ind,u,j],axis=0) for j in range(nTrials)])
            
#             pdf = sp.stats.gaussian_kde(samples,bw_method=bandwidth).evaluate(rr)
#             Prt[i,:,u] = pdf * dr # scaling?

#     return Prt

def calc_Prt_kde_sklearn(Rates_train, tt_dc, rr, bandwidth=None):
    """ calculates a KDE for rates from all trail in Rates_train
    for each timepoint in tt_dc (in unit seconds) """
    from sklearn.neighbors import KernelDensity
    if bandwidth is not None:
        kde_skl = KernelDensity(bandwidth=bandwidth*2.2)# factor empirically decuded see below
    else:
        kde_skl = KernelDensity()

    nUnits = Rates_train.shape[0]
    nTrials = Rates_train.shape[1]
    tvec = Rates_train[0,0].times.rescale('s').magnitude.flatten()
    dt = sp.diff(tt_dc)[0]
    dr = sp.diff(rr)[0]

    # get the rates out of neo objects
    R_train = sp.zeros((tvec.shape[0],nUnits,nTrials))
    for u in range(nUnits):
        for j in range(nTrials):
            R_train[:,u,j] = Rates_train[u,j].magnitude.flatten()

    # fill Prt
    Prt = sp.zeros((tt_dc.shape[0],rr.shape[0],nUnits))
    for u in range(nUnits):
        for i,t in enumerate(tt_dc):
            start_ind = sp.argmin(sp.absolute(tvec - t))
            stop_ind = sp.argmin(sp.absolute(tvec - (t + dt)))
            samples = sp.array([sp.average(R_train[start_ind:stop_ind,u,j],axis=0) for j in range(nTrials)])
            
            # kde
            kde_skl.fit(samples[:,sp.newaxis])
            res = kde_skl.score_samples(rr[:,sp.newaxis])
            pdf = sp.exp(res)
            Prt[i,:,u] = pdf * dr # scaling?

    return Prt

def decode(R, Prt, tt_dc, rr):
    """ decodes rates matrix of shape unit x time """
    nUnits = R.shape[1]
    Ls = sp.zeros((tt_dc.shape[0],tt_dc.shape[0]))
    for i,t in enumerate(tt_dc):
        L = sp.zeros((tt_dc.shape[0],nUnits))
        for u in range(nUnits):
            # find closest present rate
            r_ind = sp.argmin(sp.absolute(rr-R[i,u]))
            l = copy(Prt[:,r_ind,u])
            L[:,u] = l

        # sum the logs instead of prod the p
        L = sp.sum(sp.log(L),axis=1)

        # normalize
        L -= max(L)

        # convert back to p
        L = sp.exp(L)

        Ls[i,:] = L # input time on first axis
    return Ls

def calc_Prt(Rates_train, tt_dc, rr, estimator='sklearn_kde', bandwidth=None):
    if estimator is None:
        Prt = calc_Prt_direct(Rates_train, tt_dc, rr)
    
    # if estimator == "scipy_kde":
    #     Prt = calc_Prt_scipy(Rates_train, tt_dc, rr, bandwidth=bandwidth)

    if estimator == "sklearn_kde":
        Prt = calc_Prt_kde_sklearn(Rates_train, tt_dc, rr, bandwidth=bandwidth)
    
    return Prt

def decode_trials(Prt, Rates_test, tt_dc, rr):
    """ returns trial by trial decoding tt_dc x tt_dc x nTrials matrix """
    
    nUnits = Rates_test.shape[0]
    nTrials_test = Rates_test.shape[1]

    # decode trial by trial
    R = Rates_test[0,0]
    dt = R.sampling_period.rescale('s').magnitude
    tt = sp.arange(R.t_start.rescale('s').magnitude,R.t_stop.rescale('s').magnitude,dt)

    # get trial firing rates
    Rs = sp.zeros((tt.shape[0],nUnits,nTrials_test))
    for u in range(nUnits):
        for j in range(nTrials_test):
            Rs[:,u,j] = Rates_test[u,j].magnitude.flatten()

    # reinterpolate average rates to the decodable times
    Rsi = interp1d(tt,Rs,axis=0)(tt_dc)

    # decode
    Ls = sp.zeros((tt_dc.shape[0],tt_dc.shape[0], nTrials_test))
    for i in range(nTrials_test):
        Ls[:,:,i] = decode(Rsi[:,:,i], Prt, tt_dc, rr)

    return Ls
