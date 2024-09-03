import sys,os
from time import time
import pickle
from tqdm import tqdm
from copy import copy
import numpy as np
import pandas as pd
from pathlib import Path
from scipy.interpolate import interp1d
from sklearn.neighbors import KernelDensity

import logging
logger = logging.getLogger()

# fmt = "%(asctime)s - %(levelname)s - %(message)s"
# logging.basicConfig(format=fmt, level=logging.INFO)

# the logic for logging here should be:
# use of get_logger()
# and the logger is created in a notebook from the path etc

def kfold_split(inds, k_folds):
    """ FIXME I feel like there must me a sklearn based function that does exactly this """
    n_inds = inds.shape[0]

    # shuffle
    ix_shuff = np.arange(n_inds)
    np.random.shuffle(ix_shuff)
    inds = inds[ix_shuff]

    max_ind = n_inds - n_inds % k_folds
    inds = inds[:max_ind]

    logger.info("kfold-split: dropping %i indices" % (n_inds - max_ind))

    splits = np.array(np.sort(np.split(inds, k_folds)))

    train_inds = []
    test_inds = []
    for k in range(k_folds):
        train_k = list(range(k_folds))
        train_k.remove(k)
        test_k = k

        train_inds.append( np.concatenate([splits[i] for i in train_k]) )
        test_inds.append( splits[test_k])

    return train_inds, test_inds

def get_likelihood(Rates, tvec, tvec_decode, rvec_decode, bandwidth=None):
    """
    constructs the matrix P(r|t) = likelihood

    arguments:
        Rates = array of time x trials - this is for one unit
        tvec = the corresponding time vector
        tvec_decode = the requested times at which the decoder should decode
        rvec_decode = the same with rates
        bandwidth = passed to KDE
    
    returns:
        Prt = the array of likelihood values, with time on the first axis, rate on the 2nd
    """

    n_trials = Rates.shape[1]

    dt = np.diff(tvec_decode)[0]
    dr = np.diff(rvec_decode)[0]

    # "build" P(r|t)
    kde_skl = KernelDensity(bandwidth=bandwidth)
    Prt = np.zeros((tvec_decode.shape[0], rvec_decode.shape[0]))

    for i,t in enumerate(tvec_decode):
        start_ind = np.argmin(np.absolute(tvec - t))
        stop_ind = np.argmin(np.absolute(tvec - (t + dt)))
        samples = np.array([np.average(Rates[start_ind:stop_ind, j],axis=0) for j in range(n_trials)])
        
        # kde
        kde_skl.fit(samples[:,np.newaxis])
        res = kde_skl.score_samples(rvec_decode[:,np.newaxis])
        pdf = np.exp(res)
        Prt[i,:] = pdf * dr # scaling

    return Prt

def train(Rates, tvec, tvec_decode, rvec_decode, bandwidth=None):
    """
    trains the decoder for all units

    arguments:
        Rates = array of time x unit x trial
        tvec = the corresponding time vector
        tvec_decode = the requested times at which the decoder should decode
        rvec_decode = the same with rates
        bandwidth = passed to KDE
    
    returns:
        Prts = the array of likelihood values
            with time on the first axis
            rate on the 2nd
            units on the third
    """

    n_units = Rates.shape[1]
    Prts = np.zeros((tvec_decode.shape[0], rvec_decode.shape[0], n_units))

    for i in range(n_units):
        Prts[:,:,i] = get_likelihood(Rates[:,i,:], tvec, tvec_decode, rvec_decode, bandwidth=bandwidth)
    
    return Prts

def decode(Rates, Prts, tvec_decode, rvec_decode):
    """
    decodes a single trial

    arguments:
        Rates = array of time x units

            TODO REQUIREMENT that Rates is reinterpolated to tvec_decode??
            NOTE this only makes sense if Prts is the same time basis as Rates
            (=> which is probably tvec_decode)

        Prts = the likelihoods of the units
        tvec_decode = the requested times at which the decoder should decode
        rvec_decode = the same with rates
    
    returns:
        Ls = np.array of shape tvec_decode[0] x tvec_decode.shape[0]
            these are the posterior values P(t|r(t))
            with r(t) on the first axis (=input time)
            and decoded time on 2nd

            TODO-REFACTOR this should definitely not be called L as this confuses with likelihood
    """

    n_units = Rates.shape[1]
    Ls = np.zeros((tvec_decode.shape[0], tvec_decode.shape[0]))
    for i,t in enumerate(tvec_decode):
        L = np.zeros((tvec_decode.shape[0], n_units))
        for u in range(n_units):
            r_ind = np.argmin(np.absolute(rvec_decode-Rates[i,u])) # find closest present rate
            l = copy(Prts[:,r_ind,u])
            L[:,u] = l

        L = np.sum(np.log(L), axis=1) # sum the logs instead of prod the p
        L -= max(L) # normalize
        L = np.exp(L) # convert back to p
        Ls[i,:] = L # input time on first axis

    return Ls

def decode_trials(Rates, tvec, Prts, tvec_decode, rvec_decode):
    """
    decodes multiple trials

    arguments:
        Rates = array of time x units x trials
        tvec = the corresponding time vector
        Prts = the likelihoods of the units
        tvec_decode = the requested times at which the decoder should decode
        rvec_decode = the same with rates
    
    returns:
        Ls = np.array of shape tvec_decode[0] x tvec_decode.shape[0] x n_trials

            these are the posterior values P(t|r(t))
            with r(t) on the first axis (=input time)
            and decoded time on 2nd
            and trial on the 3rd

            TODO this should definitely not be called L, see above and refactor
    """

    n_trials = Rates.shape[2]

    # decode trial by trial
    dt = np.diff(tvec)[0]

    # reinterpolate average rates to the decodable times
    Rates_ip = interp1d(tvec, Rates, axis=0)(tvec_decode)

    # decode
    Ls = np.zeros((tvec_decode.shape[0], tvec_decode.shape[0], n_trials))
    # for i in tqdm(range(n_trials),desc='decoding trials'):
    for i in range(n_trials):
        Ls[:,:,i] = decode(Rates_ip[:,:,i], Prts, tvec_decode, rvec_decode)

    return Ls

def train_decode(Rates, tvec, train_inds, test_inds, tvec_decode, rvec_decode):
    """ TODO check if this is ever necessary, acutally makes cross-decode 
    a general case and this the special one """
    Prts = train(Rates[:,:,train_inds], tvec, tvec_decode, rvec_decode, bandwidth=0.25*2.2)
    Ls = decode_trials(Rates[:,:,test_inds], tvec, Prts, tvec_decode, rvec_decode)
    return Ls

def train_decode_xval(Rates, tvec, tvec_decode, rvec_decode, k=10):
    """ currently the only function that makes use of the new data format
    change this call signature!
    Rates: a t x n_units x n_trials array """

    n_trials = Rates.shape[2]
    inds = np.arange(n_trials)
    train_inds, test_inds = kfold_split(inds, k)

    Ls_k = np.zeros((tvec_decode.shape[0], tvec_decode.shape[0], k))
    for j in range(k):
        logger.info("xval decoding, k=%i" % k)
        Ls = train_decode(Rates, tvec, train_inds[j], test_inds[j], tvec_decode, rvec_decode)
        Ls_k[:,:,j] = np.average(Ls, axis=2)

    return np.average(Ls_k, axis=2)

def train_crossdecode_xval(Rates_vpl, Rates_vpl_da, tvec, tvec_decode, rvec_decode, k_folds=10):
    """
    Rates_vpl and Rates_vpl_da are now: a) resliced sigal, with already trials selected
    -> this should make this compatible with a meta animal
    """

    train_inds, test_inds = kfold_split(np.arange(Rates_vpl.shape[2]), k_folds)

    Ls_vpl_k = np.zeros((tvec_decode.shape[0], tvec_decode.shape[0], k_folds))
    Ls_vpl_da_k = np.zeros((tvec_decode.shape[0], tvec_decode.shape[0], k_folds))
    Ls_vpl_chance_k = np.zeros((tvec_decode.shape[0], tvec_decode.shape[0], k_folds))
    Ls_vpl_da_chance_k = np.zeros((tvec_decode.shape[0], tvec_decode.shape[0], k_folds))

    for k in range(k_folds):
        logger.info("xval x-decoding, k=%i" % k)
        Rates_vpl_train = Rates_vpl[:,:, train_inds[k]]
        Rates_vpl_test = Rates_vpl[:,:, test_inds[k]]

        Prts = train(Rates_vpl_train, tvec, tvec_decode, rvec_decode, bandwidth=0.25*2.2) # TODO oddly specific bandwith here
        Ls_vpl_k[:,:,k] = np.average(decode_trials(Rates_vpl_test, tvec, Prts, tvec_decode, rvec_decode),axis=2)
        Ls_vpl_da_k[:,:,k] = np.average(decode_trials(Rates_vpl_da, tvec, Prts, tvec_decode, rvec_decode),axis=2)

        # chance level estimation
        # shuffling Prts in time axis
        rand_inds = np.arange(tvec_decode.shape[0])
        np.random.shuffle(rand_inds)
        Prts_shuff = Prts[rand_inds,:,:]

        # decoding with shuffled Prts
        Ls_vpl_chance_k[:,:,k] = np.average(decode_trials(Rates_vpl_test, tvec, Prts_shuff, tvec_decode, rvec_decode),axis=2)
        Ls_vpl_da_chance_k[:,:,k] = np.average(decode_trials(Rates_vpl_da, tvec, Prts_shuff, tvec_decode, rvec_decode),axis=2)

    # averaging over k-folds
    Ls_vpl = np.average(Ls_vpl_k, axis=2)
    Ls_vpl_da = np.average(Ls_vpl_da_k, axis=2)
    Ls_vpl_chance = np.average(Ls_vpl_chance_k, axis=2)
    Ls_vpl_da_chance = np.average(Ls_vpl_da_chance_k, axis=2)

    return Ls_vpl, Ls_vpl_da, Ls_vpl_chance, Ls_vpl_da_chance 

"""
 
 ##     ## ######## ########    ###       ###    ##    ## #### ##     ##    ###    ##       
 ###   ### ##          ##      ## ##     ## ##   ###   ##  ##  ###   ###   ## ##   ##       
 #### #### ##          ##     ##   ##   ##   ##  ####  ##  ##  #### ####  ##   ##  ##       
 ## ### ## ######      ##    ##     ## ##     ## ## ## ##  ##  ## ### ## ##     ## ##       
 ##     ## ##          ##    ######### ######### ##  ####  ##  ##     ## ######### ##       
 ##     ## ##          ##    ##     ## ##     ## ##   ###  ##  ##     ## ##     ## ##       
 ##     ## ########    ##    ##     ## ##     ## ##    ## #### ##     ## ##     ## ######## 
 
"""

def build_metaanimal(Signals, UnitDfs, StimsDfs):
    """
    arguments: a list of Signals, UnitDfs, StimsDfs
    per animal data

    does:

    returns:
    A dict, where the stim_label index will return the truncated along trial and 
    stacked along unit axis of all animals = the meta animal

    a merged UnitDf

    TODO move this to data_structures.py

    """

    n_animals = len(Signals)
    stim_labels = np.sort(StimsDfs[0]['stim_id'].unique())

    # create resorted
    for i in range(n_animals):
        Signals[i].resort_to_labels(StimsDfs[i]['stim_id'].values)

    # Meta UnitDf
    Meta_UnitDf = pd.concat([UnitDf.groupby('selection').get_group(True) for UnitDf in UnitDfs],axis=0)

    # Meta Animal
    Meta_Animal = {}

    # cut to min and stack
    for j, label in enumerate(stim_labels):
        n_trials = [Signals[i].resorted[label].shape[2] for i in range(n_animals)]
        min_trials = np.min(n_trials)
        Meta_Animal[label] = np.concatenate([Signals[i].resorted[label][:,:,:min_trials] for i in range(n_animals)],axis=1)

    return Meta_Animal, Meta_UnitDf