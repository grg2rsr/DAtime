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
 
                      _ 
   _ __ ___  __ _  __| |
  | '__/ _ \/ _` |/ _` |
  | | |  __/ (_| | (_| |
  |_|  \___|\__,_|\__,_|
                        
 
"""
# %%
folder = "/home/georg/data/2020-03-04_GR_JP2111_full/stim1_g0"
os.chdir(folder)

import importlib
import params
importlib.reload(params)

# bin and ks2
bin_path = os.path.join(folder,params.dat_path)

frates_sliced_path = os.path.splitext(bin_path)[0] + '.frates_sliced.npy'
frates_z_sliced_path = os.path.splitext(bin_path)[0] + '.frates_z_sliced.npy'

print("loading resliced firing rates:", os.path.basename(frates_sliced_path))
with open(frates_sliced_path,'rb') as fH:
    Rates = pickle.load(fH)
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


"""
 
                  _       _       _        
   _ __ ___  __ _| |   __| | __ _| |_ __ _ 
  | '__/ _ \/ _` | |  / _` |/ _` | __/ _` |
  | | |  __/ (_| | | | (_| | (_| | || (_| |
  |_|  \___|\__,_|_|  \__,_|\__,_|\__\__,_|
                                           
 
"""

# %% select data
k = 2 # the stim to analyze - this depends on what is stored!

stim_inds = StimsDf.groupby(['stim_id','opto']).get_group((k,'red')).index
Rates_ = Rates_z[:,stim_inds]
nTrials = stim_inds.shape[0]
nUnits = Rates_.shape[0]

# setting the starting point to -1s
# FIXME hardcode - this could be inferred if event is present in neo obj
# - is analogsignal so it would need to be a segment ... 

for i in range(Rates_.shape[0]):
    for j in range(Rates_.shape[1]):
        Rates_[i,j].t_start = -1*pq.s


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
    r = average_asigs(Rates[u,:])
    r_avgs[:,u] = r.magnitude.flatten()


inds = sp.argsort(sp.argmax(r_avgs,0))
ext = (r.times[0],r.times[-1],0,nUnits)
plt.matshow(r_avgs.T[inds,:],origin='bottom')
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


def calc_kdes(Rates_train, tt_dc):
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

    all_kdes = []
    for u in range(nUnits):
        kdes = []
        for i,t in enumerate(tt_dc):
            samples = []
            # the sp way
            start_ind = sp.argmin(sp.absolute(tvec - t))
            stop_ind = sp.argmin(sp.absolute(tvec - (t + dt)))
            samples = [sp.average(R_train[start_ind:stop_ind,u,j],axis=0) for j in range(nTrials)]
            # the old way
                # for j in range(Rates_train.shape[1]):
                    # times = (t,t+dt) * pq.s
                    # samples.append(sp.average(Rates_train[u,j].time_slice(*times).magnitude))
            kdes.append(sp.stats.gaussian_kde(samples))
        all_kdes.append(kdes)

    return all_kdes

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

def calc_Prt(all_kdes, tt_dc, rr):
    """ P(r|t) is of shape t x r x unit """

    nUnits = len(all_kdes)
    Prt = sp.zeros((tt_dc.shape[0],rr.shape[0],nUnits))
    for u in range(nUnits):
        kdes = all_kdes[u]
        for i,t in enumerate(tt_dc):
            Prt[i,:,u] = kdes[i].evaluate(rr) * dr
    
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
            l = Prt[:,r_ind,u]
            l[l==0] = 1e-5 # to avoid multiplication w 0
            L[:,u] = l

        if output == 'p':
            L = sp.prod(L,axis=1)
            L = L / L.max()

        if output == 'log':
            L = sp.sum(sp.log(L),axis=1)

        Ls[i,:] = L # input time on first axis
    return Ls


def decode_trials(Rates_train, Rates_test, tt_dc, rr, estimator=None, output='p', bandwidth=None):
    """ returns trial by trial decoding tt_dc x tt_dc x nTrials matrix """
    if estimator is None:
        Prt = calc_Prt_direct(Rates_train, tt_dc, rr)
    
    if estimator == "scipy_kde":
        Prt = calc_Prt_scipy(Rates_train, tt_dc, bandwidth=bandwidth)

    if estimator == "sklearn_kde":
        Prt = calc_Prt_kde_sklearn(Rates_train, tt_dc, bandwidth=bandwidth) 
    
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
rand_inds = sp.arange(nTrials)
sp.random.shuffle(rand_inds)
Rates_ = Rates_[:,rand_inds]

Rates_split = sp.split(Rates_,k,1)

# decoding times
dt = 0.01
tt_dc = sp.arange(0.5,2.5,dt)

# firing rate vector and binning
dr = 0.1
rr = sp.arange(-4,4,dr)

Ls_avgs = sp.zeros((tt_dc.shape[0],tt_dc.shape[0],k))

tstart = time()

for ki in range(k):
    ti1 = time()
    Rates_test = Rates_split[ki]
    train_inds = list(range(k))
    train_inds.remove(ki)
    Rates_train = sp.concatenate([Rates_split[i] for i in train_inds],axis=1)

    Ls = decode_trials(Rates_train, Rates_test, tt_dc, rr, 
                       estimator="sklearn_kde", output='p', bandwidth=0.15)

    # store
    Ls_avgs[:,:,ki] = sp.average(Ls,axis=2)
    
    # error quant
    tt_decoded = tt_dc[sp.argmax(sp.average(Ls,axis=2),axis=1)]
    rss = sp.sum((tt_decoded - tt_dc)**2)

    # timeit
    ti2 = time()
    tloop = ti2-ti1
    print("loop time: ",tloop," left: ",(k-ki)*tloop, "rss: ", rss)

tfin = time()
print("total runtime: ", tfin-tstart)

Ls_avg = sp.average(Ls_avgs,axis=2)
tt_decoded = tt_dc[sp.argmax(Ls_avg,axis=1)]
rss = sp.sum((tt_decoded - tt_dc)**2)
print("error from avg: ", rss)
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
# Ls_avg = Ls_avgs[:,:,1]
colors = sns.color_palette('viridis',n_colors=tt_dc.shape[0])
fig, axes = plt.subplots()

ysep = 0.0
for i in range(tt_dc.shape[0]):
    axes.plot(tt_dc,Ls_avg[i,:]+i*ysep,lw=1,alpha=0.8,color=colors[i])
    idx = sp.argmax(Ls_avg[i,:])
    axes.plot(tt_dc[idx],Ls_avg[i,idx]+i*ysep,'.',alpha=0.5,color=colors[i])

fig , axes = plt.subplots()
ext = (tt_dc[0],tt_dc[-1],tt_dc[0],tt_dc[-1])
axes.matshow(Ls_avg.T,origin='bottom',extent=ext)
axes.set_aspect('auto')

# %% plot real time vs decoded time
fig, axes = plt.subplots()
tt_decoded = tt_dc[sp.argmax(Ls_avg,axis=1)]
for i,v in enumerate(tt_decoded):
    axes.plot(tt_dc[i],v,'o',color=colors[i])
axes.plot(tt_dc,tt_dc,':',alpha=0.5,color='k')


# %% Prt inspect
u = 20
fig , axes = plt.subplots()
ext = (tt_dc[0],tt_dc[-1],rr[0],rr[-1])
axes.matshow(Prt[:,:,u].T,origin='bottom',extent=ext)
axes.set_aspect('auto')
axes.set_xlabel('time')
axes.set_ylabel('firing rate')

# %% vis kdes
u = 0
kdes = all_kdes[u]
K = sp.zeros((tt_dc.shape[0],rr.shape[0]))
for i,t in enumerate(tqdm(tt_dc)):
    K[i,:] = [kdes[i].integrate_box_1d(r,r+dr) for r in rr]
    # K[i,:] = kdes[i].evaluate(rr)

plt.matshow(K.T)

# %%
r = 2
K = sp.zeros((tt_dc.shape[0],nUnits))

for u in range(nUnits):
    for i,t in enumerate(tt_dc):
        kdes = all_kdes[inds[u]]
        K[i,u] = kdes[i].integrate_box_1d(r,r+dr)

plt.matshow(K.T,origin='bottom')

""" notes
with dt = 0.01

sscipy estimator makes weird lobes 
whereas sklearn estimator doesn't (more granular though)

"""

# %% analysis of errors
rss = sp.sum((tt_decoded - tt_dc)**2)
rss

# %% comparing bandwiths
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
kde_sk = KernelDensity(bandwidth=bw*2.2)
kde_sk.fit(samples[:,sp.newaxis])
pdf_sk = sp.exp(kde_sk.score_samples(v[:,sp.newaxis]))
plt.plot(v,pdf_sk*dv)
# %%


# %%
