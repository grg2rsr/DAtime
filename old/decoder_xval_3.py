# %%
%matplotlib qt5

import matplotlib.pyplot as plt
import matplotlib as mpl
import seaborn as sns

import sys,os
from time import time
import pickle
# from tqdm import tqdm_notebook as tqdm
from tqdm import tqdm
from copy import copy
import scipy as sp
import numpy as np
import pandas as pd
import neo
import elephant as ele
import quantities as pq

from scipy.interpolate import interp1d
plt.style.use('default')
mpl.rcParams['figure.dpi'] = 331

"""
 
                  _       _       _        
   _ __ ___  __ _| |   __| | __ _| |_ __ _ 
  | '__/ _ \/ _` | |  / _` |/ _` | __/ _` |
  | | |  __/ (_| | | | (_| | (_| | || (_| |
  |_|  \___|\__,_|_|  \__,_|\__,_|\__\__,_|
                                           
 
"""
# %%
# folder = "/home/georg/data/2020-03-04_GR_JP2111_full/stim1_g0"
folder = "/home/georg/data/2019-12-03_JP1355_GR_full_awake_2/stim3_g0"

os.chdir(folder)

import importlib
import params
importlib.reload(params)

# bin and ks2
bin_path = os.path.join(folder,params.dat_path)

# frates_sliced_path = os.path.splitext(bin_path)[0] + '.frates_sliced.npy'
frates_z_sliced_path = os.path.splitext(bin_path)[0] + '.frates_z_sliced.npy'

# print("loading resliced firing rates:", os.path.basename(frates_sliced_path))
# with open(frates_sliced_path,'rb') as fH:
#     Rates = pickle.load(fH)

print("loading resliced z-scored firing rates:", os.path.basename(frates_z_sliced_path))
with open(frates_z_sliced_path,'rb') as fH:
    Rates_z = pickle.load(fH)

# stimulus related
for fname in os.listdir():
    if 'stims.csv' in fname and not 'map' in fname:
        stim_fpath = os.path.join(folder,fname)
    if 'stim_map' in fname:
        stim_map_fpath = os.path.join(folder,fname)

# read stims
""" note that there is here a reversal in the nomenclature !!! 
the StimMap resolves stim id to stim parameters, the StimsDf
actually has which stim id was presented when ... 
"""
StimMap = pd.read_csv(stim_fpath, delimiter=',')
StimsDf = pd.read_csv(stim_map_fpath, delimiter=',')

# selection
k = 2 # the stim to analyze 
stim_inds = StimsDf.groupby(['stim_id','opto']).get_group((k,'red')).index
Rates_ = Rates_z[:,stim_inds]

opto_inds = StimsDf.groupby(['stim_id','opto']).get_group((k,'both')).index
Rates_opto = Rates_z[:,opto_inds]

nTrials = stim_inds.shape[0]
nUnits = Rates_.shape[0]

# setting the starting point to -1s
# FIXME hardcode - this could be inferred if event is present in neo obj
# - is analogsignal so it would need to be a segment ... 

for i in range(Rates_.shape[0]):
    for j in range(Rates_.shape[1]):
        Rates_[i,j].t_start = -1*pq.s

for i in range(Rates_opto.shape[0]):
    for j in range(Rates_opto.shape[1]):
        Rates_opto[i,j].t_start = -1*pq.s

"""
 
    __       _              _       _        
   / _| __ _| | _____    __| | __ _| |_ __ _ 
  | |_ / _` | |/ / _ \  / _` |/ _` | __/ _` |
  |  _| (_| |   <  __/ | (_| | (_| | || (_| |
  |_|  \__,_|_|\_\___|  \__,_|\__,_|\__\__,_|
                                             
 
"""
# %%
dt = 0.01
tt = sp.arange(-1,3,dt)

nUnits = 50
nTrials = 50

Rates = sp.zeros((nUnits,nTrials),dtype='object')
SpikeTrains = sp.zeros((nUnits,nTrials),dtype='object')

# generating spike trains
fr_opts = dict(sampling_period=dt*pq.s, kernel=ele.kernels.GaussianKernel(sigma=50*pq.ms))

for i in tqdm(range(nUnits)):
    # peak_time = tt[sp.random.randint(tt.shape[0])]
    peak_time = sp.rand() * tt[-1]
    rate_gen = sp.stats.distributions.norm(peak_time,0.5).pdf(tt) * 10
    asig = neo.core.AnalogSignal(rate_gen,t_start=tt[0]*pq.s,units=pq.Hz,sampling_period=dt*pq.s)
    for j in range(nTrials):
        st = ele.spike_train_generation.inhomogeneous_poisson_process(asig)
        SpikeTrains[i,j] = st

        r = ele.statistics.instantaneous_rate(st,**fr_opts)
        rz = ele.signal_processing.zscore(r)
        Rates[i,j] = rz

Rates_ = Rates

# %% inspect this
# plot average rates

def average_asigs(asigs):
    S = sp.stack([asig.magnitude.flatten() for asig in asigs],axis=1)
    asig = neo.core.AnalogSignal(sp.average(S,axis=1),units=asigs[0].units,t_start=asigs[0].t_start,sampling_rate=asigs[0].sampling_rate)
    return asig

nUnits = Rates_.shape[0]
r_avgs = sp.zeros((Rates_[0,0].shape[0],nUnits))
for u in range(nUnits):
    r = average_asigs(Rates_[u,:])
    r_avgs[:,u] = r.magnitude.flatten()


inds = sp.argsort(sp.argmax(r_avgs,0))
ext = (r.times[0],r.times[-1],0,nUnits)
plt.matshow(r_avgs.T[inds,:],origin='bottom',extent=ext,vmin=-1,vmax=3)
plt.gca().set_aspect('auto')

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

def decode(R, Prt, tt_dc, rr, output='p'):
    """ decodes rates matrix of shape unit x time """
    nUnits = R.shape[1]
    Ls = sp.zeros((tt_dc.shape[0],tt_dc.shape[0]))
    for i,t in enumerate(tt_dc):
        L = sp.zeros((tt_dc.shape[0],nUnits))
        for u in range(nUnits):
            # find closest present rate
            r_ind = sp.argmin(sp.absolute(rr-R[i,u]))
            l = copy(Prt[:,r_ind,u])
            # l[l==0] = 1e-1 # to avoid multiplication to 0
            L[:,u] = l

        if output == 'p':
            # L = L*2
            L = sp.prod(L,axis=1)
            if not sp.all(L==0):
                L = L / L.max()

        if output == 'log':
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

def decode_trials(Prt, Rates_test, tt_dc, rr, output='p'):
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
        Ls[:,:,i] = decode(Rsi[:,:,i], Prt, tt_dc, rr, output=output)

    return Ls

# %% spitting data into train and test
k = 5
nTrials = Rates_.shape[1]
rand_inds = sp.arange(nTrials)
sp.random.shuffle(rand_inds)

nUnits_lim = nUnits
Rates_ = Rates_[:,rand_inds]

Rates__ = Rates_[:nUnits_lim,:]
Rates_opto_ = Rates_opto[:nUnits_lim,:]

# discard excess data
ex = nTrials % k
if ex != 0:
    Rates_cut = Rates__[:,:-ex]
else:
    Rates_cut = Rates__

Rates_split = sp.split(Rates_cut,k,1)

# decoding times
dt = 0.1
tt_dc = sp.arange(0.0,2.5,dt)

# firing rate vector and binning
dr = 0.1
rr = sp.arange(-4,4,dr)

Ls_avgs = sp.zeros((tt_dc.shape[0],tt_dc.shape[0],k))
Ls_opto_avgs = sp.zeros((tt_dc.shape[0],tt_dc.shape[0],k))

tstart = time()

for ki in range(k):
    ti1 = time()
    Rates_test = Rates_split[ki]
    train_inds = list(range(k))
    train_inds.remove(ki)
    Rates_train = sp.concatenate([Rates_split[i] for i in train_inds],axis=1)

    # parameters
    estimator = "sklearn_kde"
    output = "log"

    # train and self validate
    Prt = calc_Prt(Rates_train, tt_dc, rr, estimator=estimator, bandwidth=0.25)
    Ls = decode_trials(Prt, Rates_test, tt_dc, rr, output=output)

    # decode all opto trials with this
    Ls_opto = decode_trials(Prt, Rates_opto_, tt_dc, rr, output=output)

    # store
    Ls_avgs[:,:,ki] = sp.average(Ls,axis=2)
    Ls_opto_avgs[:,:,ki] = sp.average(Ls_opto,axis=2)

    # error quant
    tt_decoded = tt_dc[sp.argmax(sp.average(Ls,axis=2),axis=1)]
    rss = sp.sum((tt_decoded - tt_dc)**2)
    t_err = sp.sqrt(rss / tt_dc.shape[0])

    # timeit
    ti2 = time()
    tloop = ti2-ti1
    print("loop time: %5.1f, left: %5.1f, rss/n: %8.4f, t_err: %8.4f" % (tloop, (k-ki)*tloop, rss/tt_dc.shape[0], t_err))

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

def label_stim_artifact(axes,k,StimMap,offset,axis='x'):
    dur,n,f = StimMap.iloc[k][['dur','n','f']]
    for j in range(int(n)):
        # start stop of each pulse in s
        start = j * 1/f + offset.rescale('s').magnitude
        stop = start + dur
        if axis is 'x':
            axes.axvspan(start,stop,color='firebrick',alpha=0.25,linewidth=0)
        if axis is 'y':
            axes.axhspan(start,stop,color='firebrick',alpha=0.25,linewidth=0)

# %%
 
Ls_avg = sp.average(Ls_avgs,axis=2)
Ls_opto_avg = sp.average(Ls_opto_avgs,axis=2)

Ls_D = Ls_opto_avg - Ls_avg

fig , axes = plt.subplots(ncols=3,figsize=[11,4])

ext = (tt_dc[0],tt_dc[-1],tt_dc[0],tt_dc[-1])
v = 0.1
im = axes[0].matshow(Ls_avg.T,origin='bottom',extent=ext, cmap='viridis')
im = axes[1].matshow(Ls_opto_avg.T,origin='bottom',extent=ext, cmap='viridis')
im = axes[2].matshow(Ls_D.T,origin='bottom',extent=ext, cmap='PiYG',vmin=-v,vmax=v)

plt.colorbar(im,label='stim - no stim',shrink=0.75)

axes[0].set_title('decoding VPL only')
axes[1].set_title('decoding VPL+SNc stim')
axes[2].set_title('difference')

for ax in axes:
    label_stim_artifact(ax,2,StimMap,0.25*pq.s,axis='x')
    label_stim_artifact(ax,2,StimMap,0.25*pq.s,axis='y')

    ax.set_aspect('equal')
    ax.set_xlabel('real time (s)')
    ax.set_ylabel('decoded time (s)')

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

for i,v in enumerate(tt_decoded):
    axes.plot(tt_dc[i],v,'o',color='firebrick',alpha=0.8)

for i,v in enumerate(tt_decoded_opto):
    axes.plot(tt_dc[i],v,'o',color='darkcyan',alpha=0.8)

axes.plot(tt_dc,tt_dc,':',alpha=0.5,color='k')
label_stim_artifact(axes,2,StimMap,0.25*pq.s,axis='x')
axes.set_xlabel('real time (s)')
axes.set_ylabel('decoded time (s)')

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

# %%
"""
ki = 1
i = 16
"""