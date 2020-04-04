# %%
%matplotlib qt5

import matplotlib.pyplot as plt
import matplotlib as mpl
import seaborn as sns

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
plt.style.use('default')
mpl.rcParams['figure.dpi'] = 331

"""
 
                      _ 
   _ __ ___  __ _  __| |
  | '__/ _ \/ _` |/ _` |
  | | |  __/ (_| | (_| |
  |_|  \___|\__,_|\__,_|
                        
 
"""
# %% 
folder = Path("/home/georg/data/2019-12-03_JP1355_GR_full_awake_2/stim3_g0/")

os.chdir(folder)
import analysis_params

bin_file = folder / analysis_params.bin_file
path = bin_file.with_suffix('.neo.zfr.sliced')
# path = folder / "test_data.neo.zfr.sliced"

with open(path,'rb') as fH:
    print("reading ", path)
    Segs = pickle.load(fH)
    print("... done")

# read stim related info
stim_path = folder / analysis_params.stim_file
stim_map_path = folder / analysis_params.stim_map_file

StimMap = pd.read_csv(stim_map_path, delimiter=',')
StimsDf = pd.read_csv(stim_path, delimiter=',')

# %% # reformatting from new data format 
nUnits = len(Segs[0].spiketrains)
nTrials = len(Segs)

Rates = sp.zeros((nUnits,nTrials),dtype='object')
for j in range(nTrials):
    Rates[:,j] = Segs[j].analogsignals

# %% selection

stim_k = 1 # the stim to analyze 
stim_inds = StimsDf.groupby(['stim_id','opto']).get_group((stim_k,'red')).index
Rates_ = Rates[:,stim_inds]

opto_inds = StimsDf.groupby(['stim_id','opto']).get_group((stim_k,'both')).index
Rates_opto_ = Rates[:,opto_inds]

nTrials = stim_inds.shape[0]
nUnits = Rates_.shape[0]

# keep an epoch
vpl_stim, = select(Segs[stim_inds[0]].epochs,'VPL_stims')

# %% inspect this
# plot average rates

# vpl data
nUnits = Rates_.shape[0]
r_avgs = sp.zeros((Rates_[0,0].shape[0],nUnits))
for u in range(nUnits):
    r = average_asigs(Rates_[u,:])
    r_avgs[:,u] = r.magnitude.flatten()

inds = sp.argsort(sp.argmax(r_avgs,0))[::-1]
ext = (r.times[0],r.times[-1],0,nUnits)
plt.matshow(r_avgs.T[inds,:],origin='bottom',extent=ext,vmin=-1,vmax=3)
plt.gca().set_aspect('auto')
add_stim(plt.gca(),vpl_stim)

# opto data
nUnits = Rates_opto_.shape[0]
r_avgs_opto = sp.zeros((Rates_opto_[0,0].shape[0],nUnits))
for u in range(nUnits):
    r = average_asigs(Rates_opto_[u,:])
    r_avgs_opto[:,u] = r.magnitude.flatten()

# inds = sp.argsort(sp.argmax(r_avgs,0))
ext = (r.times[0],r.times[-1],0,nUnits)
plt.matshow(r_avgs_opto.T[inds,:],origin='bottom',extent=ext,vmin=-1,vmax=3)
plt.gca().set_aspect('auto')
add_stim(plt.gca(),vpl_stim)

D = r_avgs_opto - r_avgs
v = 2
plt.matshow(D.T[inds,:],origin='bottom',extent=ext,vmin=-v,vmax=v,cmap='PiYG')
plt.gca().set_aspect('auto')
add_stim(plt.gca(),vpl_stim)

# %% 
fig, axes = plt.subplots()
colors = sns.color_palette('viridis',n_colors=nUnits)
ysep = 0.2
tvec = Rates_[0,0].times.magnitude.flatten()
for u in range(nUnits):
    axes.plot(r_avgs[:,inds[u]]+u*ysep,color=colors[u],zorder=-1*u)
    # axes.fill_between(tvec,r_avgs[:,inds[u]]+u*ysep,u*ysep,color=colors[u],zorder=-1*u)

"""
 
   ____  _____ ____ ___  ____  _____ ____  
  |  _ \| ____/ ___/ _ \|  _ \| ____|  _ \ 
  | | | |  _|| |  | | | | | | |  _| | |_) |
  | |_| | |__| |__| |_| | |_| | |___|  _ < 
  |____/|_____\____\___/|____/|_____|_| \_\
                                           
 
"""
# %%

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

def calc_Prt_scipy(Rates_train, tt_dc, bandwidth=None):
    """ calculates a KDE for rates from all trail in Rates_train
    for each timepoint in tt_dc (in unit seconds) """

    nUnits = Rates_train.shape[0]
    nTrials = Rates_train.shape[1]
    tvec = Rates_train[0,0].times.rescale('s').magnitude.flatten()

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
            
            pdf = sp.stats.gaussian_kde(samples,bw_method=bandwidth).evaluate(rr)
            Prt[i,:,u] = pdf * dr # scaling?

    return Prt

def calc_Prt_kde_sklearn(Rates_train, tt_dc, bandwidth=None):
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
    
    if estimator == "scipy_kde":
        Prt = calc_Prt_scipy(Rates_train, tt_dc, bandwidth=bandwidth)

    if estimator == "sklearn_kde":
        Prt = calc_Prt_kde_sklearn(Rates_train, tt_dc, bandwidth=bandwidth)
    
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

# %% spitting data into train and test
k = 5
nTrials = Rates_.shape[1]
rand_inds = sp.arange(nTrials)
sp.random.shuffle(rand_inds)

nUnits_lim = nUnits
Rates_ = Rates_[:,rand_inds]

# another working copy: subset in units
Rates__ = Rates_[:nUnits_lim,:]
Rates_opto__ = Rates_opto_[:nUnits_lim,:]

# discard excess data
ex = nTrials % k
if ex != 0:
    Rates_cut = Rates__[:,:-ex]
    print("discarded %2d trials" % (ex))
else:
    Rates_cut = Rates__

Rates_split = sp.split(Rates_cut,k,1)

# decoding times
dt = 0.01
tt_dc = sp.arange(0.4,2.95,dt)

# firing rate vector and binning
dr = 0.1
rr = sp.arange(-4,4,dr)

Ls_avgs = sp.zeros((tt_dc.shape[0],tt_dc.shape[0],k))
Ls_opto_avgs = sp.zeros((tt_dc.shape[0],tt_dc.shape[0],k))

tstart = time()
tloops = []
for ki in range(k):
    ti1 = time()
    Rates_test = Rates_split[ki]
    train_inds = list(range(k))
    train_inds.remove(ki)
    Rates_train = sp.concatenate([Rates_split[i] for i in train_inds],axis=1)

    # train and self validate
    Prt = calc_Prt(Rates_train, tt_dc, rr, bandwidth=0.25)
    Ls = decode_trials(Prt, Rates_test, tt_dc, rr)

    # decode all opto trials with this
    Ls_opto = decode_trials(Prt, Rates_opto__, tt_dc, rr)

    # store
    Ls_avgs[:,:,ki] = sp.average(Ls,axis=2)
    Ls_opto_avgs[:,:,ki] = sp.average(Ls_opto,axis=2)

    # error quant
    tt_decoded = tt_dc[sp.argmax(sp.average(Ls,axis=2),axis=1)]
    rss = sp.sum((tt_decoded - tt_dc)**2)
    rss_avg = rss / tt_dc.shape[0]
    t_err = sp.sqrt(rss_avg)

    # timeit
    ti2 = time()
    tloop = ti2-ti1
    tloops.append(tloop)
    print("loop time: %5.1f, left: %5.1f, rss/n: %8.4f, t_err: %8.4f" % (tloop, (k-ki-1)*sp.average(tloops), rss_avg, t_err))

tfin = time()
print("total runtime: ", tfin-tstart)

Ls_avg = sp.average(Ls_avgs,axis=2)
tt_decoded = tt_dc[sp.argmax(Ls_avg,axis=1)]
rss = sp.sum((tt_decoded - tt_dc)**2)
print("error from avg: ", rss/tt_dc.shape[0])
# sp.save('decoder_out_kde_20units_001.npy',Ls_avgs)

"""
 
         _     
  __   _(_)___ 
  \ \ / / / __|
   \ V /| \__ \
    \_/ |_|___/
               
 
"""
# %%
 
Ls_avg = sp.average(Ls_avgs,axis=2)
Ls_opto_avg = sp.average(Ls_opto_avgs,axis=2)

Ls_D = Ls_opto_avg - Ls_avg

gkw = dict(height_ratios=(1,0.05))
fig , axes = plt.subplots(nrows=2, ncols=3, figsize=[10,4.6], gridspec_kw=gkw)

ext = (tt_dc[0],tt_dc[-1],tt_dc[0],tt_dc[-1])
v = 0.1
kw = dict(vmin=0, vmax=0.25, cmap='magma', extent=ext, origin="bottom")
kw_D = dict(vmin=-v,vmax=v,cmap='PiYG', extent=ext, origin="bottom")
kw_bar = dict(orientation="horizontal",label="normed p", shrink=0.8)

im = axes[0,0].matshow(Ls_avg.T,**kw)
fig.colorbar(im,cax=axes[1,0],**kw_bar)

im = axes[0,1].matshow(Ls_opto_avg.T,**kw)
fig.colorbar(im,cax=axes[1,1],**kw_bar)

im = axes[0,2].matshow(Ls_D.T,**kw_D)
fig.colorbar(im,cax=axes[1,2],orientation="horizontal",label="stim - no stim", shrink=0.8)

axes[0,0].set_title('decoding VPL only')
axes[0,1].set_title('decoding VPL+SNc stim')
axes[0,2].set_title('difference')

for ax in axes[0,:]:
    add_stim(ax,vpl_stim,axis='xy',DA=False)

    ax.set_aspect('equal')
    ax.set_xlabel('real time (s)')
    ax.set_ylabel('decoded time (s)')
    ax.xaxis.set_ticks_position('bottom')

    ax.set_xlim(tt_dc[0],tt_dc[-1])
    ax.set_ylim(tt_dc[0],tt_dc[-1])

fig.tight_layout()

# %%
colors = sns.color_palette('viridis',n_colors=tt_dc.shape[0])
fig, axes = plt.subplots()

ysep = 0.0
for i in range(tt_dc.shape[0]):
    axes.plot(tt_dc,Ls_avg[i,:]+i*ysep,lw=1,alpha=0.8,color=colors[i])
    idx = sp.argmax(Ls_avg[i,:])
    axes.plot(tt_dc[idx],Ls_avg[i,idx]+i*ysep,'.',alpha=0.5,color=colors[i])


# %% plot real time vs decoded time
fig, axes = plt.subplots()
tt_decoded = tt_dc[sp.argmax(Ls_avg,axis=1)]
tt_decoded_opto = tt_dc[sp.argmax(Ls_opto_avg,axis=1)]

axes.plot(tt_dc,tt_decoded,'.',color='firebrick',alpha=0.8)
axes.plot(tt_dc,tt_decoded_opto,'.',color='darkcyan',alpha=0.8)
axes.plot(tt_dc,tt_dc,':',alpha=0.5,color='k')
axes.set_ylabel('decoded time (s)')
axes.set_xlabel('real time (s)')

add_stim(axes,vpl_stim,axis='x',DA=False)

"""
 
       _                  
    __| | ___  _ __   ___ 
   / _` |/ _ \| '_ \ / _ \
  | (_| | (_) | | | |  __/
   \__,_|\___/|_| |_|\___|
                          
 
"""
# %% Prt inspect
u = 10
fig , axes = plt.subplots()
ext = (tt_dc[0],tt_dc[-1],rr[0],rr[-1])
axes.matshow(Prt[:,:,u].T,origin='bottom',extent=ext)
axes.set_aspect('auto')
axes.set_xlabel('time')
axes.set_ylabel('firing rate')


# %% comparing sklearn and scipy kde bandwiths
N = int(200)
samples = sp.concatenate([sp.randn(N)-2,sp.randn(N)+2],axis=0)
samples.shape
v = sp.linspace(-6,6,50)
dv = sp.diff(v)[0]
# plt.hist(samples,bins=v,density=True)

bw = 0.1
pdf = sp.stats.gaussian_kde(samples,bw_method=bw).evaluate(v)
plt.plot(v,pdf*dv)

from sklearn.neighbors import KernelDensity
kde_sk = KernelDensity(bandwidth=bw*2.2) # factor 2.2 makes them almost equal
kde_sk.fit(samples[:,sp.newaxis])
pdf_sk = sp.exp(kde_sk.score_samples(v[:,sp.newaxis]))
plt.plot(v,pdf_sk*dv)
