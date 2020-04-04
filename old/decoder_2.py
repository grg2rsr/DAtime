# %%
%matplotlib qt5

import matplotlib.pyplot as plt
import matplotlib as mpl
import seaborn as sns

import sys,os
import pickle
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
dt = 0.05
tt = sp.arange(-1,3,dt)

nUnits = 10
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

"""
 
   ____  _____ ____ ___  ____  _____ ____  
  |  _ \| ____/ ___/ _ \|  _ \| ____|  _ \ 
  | | | |  _|| |  | | | | | | |  _| | |_) |
  | |_| | |__| |__| |_| | |_| | |___|  _ < 
  |____/|_____\____\___/|____/|_____|_| \_\
                                           
 
"""

# %% spitting data into train and test
k = 5
split_ind = int(nTrials/k*(k-1))
rand_inds = sp.arange(nTrials)
sp.random.shuffle(rand_inds)
Rates_ = Rates_[:,rand_inds]
Rates_train = Rates_[:,:split_ind]
Rates_test = Rates_[:,split_ind:]

# %% P(r|t) generation
# requested decoding time vector
dt = 0.05
tt_dc = sp.arange(0.5,2.5,dt)

# making the kdes
all_kdes = []
for u in tqdm(range(nUnits)):
    kdes = []
    for i,t in enumerate(tt_dc):
        samples = []
        for n in range(Rates_train.shape[1]):
            times = (t,t+dt) * pq.s
            samples.append(sp.average(Rates_train[u,n].time_slice(*times).magnitude))
        kdes.append(sp.stats.gaussian_kde(samples))
    all_kdes.append(kdes)


# # making the kdes
# force_recalc = False
# kdes_path = os.path.splitext(bin_path)[0] + 'kdes.npy'

# if os.path.exists(kdes_path) and not force_recalc:
#     print("loading kdes")
#     with open(kdes_path,'rb') as fH:
#         all_kdes = pickle.load(fH)
# else:
#     # making the estimates
#     all_kdes = []
#     for u in tqdm(range(nUnits)):
#         kdes = []
#         for i,t in enumerate(tt_dc):
#             samples = []
#             for n in range(nTrials):
#                 times = (t,t+dt) * pq.s
#                 samples.append(sp.average(Rates_[u,n].time_slice(*times).magnitude))
#             kdes.append(sp.stats.gaussian_kde(samples))
#         all_kdes.append(kdes)

#     with open(kdes_path,'wb') as fH:
#         pickle.dump(all_kdes,fH)

# %% filling P(r|t)

# firing rate vector and binning
dr = 0.2
rr = sp.arange(-4,4,dr)

# P(r|t) is of shape t x r x unit
Prt = sp.zeros((tt_dc.shape[0],rr.shape[0],nUnits))

for u in tqdm(range(nUnits)):
    kdes = all_kdes[u]
    for i,t in enumerate(tt_dc):
        Prt[i,:,u] = [kdes[i].integrate_box_1d(r,r+dr) for r in rr]

# %%
fig , axes = plt.subplots()
ext = (tt_dc[0],tt_dc[-1],rr[0],rr[-1])
axes.matshow(Prt[:,:,2].T,origin='bottom',vmin=0,vmax=1e-1,extent=ext)
axes.set_aspect('auto')
axes.set_xlabel('time')
axes.set_ylabel('firing rate')

# %% first simple test: decoding the average trajectory
# OF THE TEST SET
# time vector of data
R = Rates_test[0,0]
dt = R.sampling_period.rescale('s').magnitude
tt = sp.arange(R.t_start.rescale('s').magnitude,R.t_stop.rescale('s').magnitude,dt)

# get avg firing rates
r_avgs = sp.zeros((tt.shape[0],nUnits))
for u in range(nUnits):
    sigs = Rates_test[u,:]
    S = sp.stack([sig.magnitude for sig in sigs], axis=1)[:,:,0]
    r_avgs[:,u] = sp.average(S,axis=1)

inds = sp.argsort(sp.argmax(r_avgs,0))
ext = (-1,3,0,nUnits)
plt.matshow(r_avgs.T[inds,:],origin='bottom',extent=ext)
plt.gca().set_aspect('auto')

# reinterpolate average rates to the decodable times
r_avgs_ = interp1d(tt,r_avgs,axis=0)(tt_dc)

# %% decode it

# declared above
# dt = 0.05
# tt_dc = sp.arange(0.2,2.5,dt)

def decode(R):
    """ decodes rates matrix of shape unit x time """
    Ls = sp.zeros((tt_dc.shape[0],tt_dc.shape[0]))
    for i,t in enumerate(tt_dc):
        L = sp.zeros((tt_dc.shape[0],nUnits))
        for u in range(nUnits):
            # find closest present rate
            r_ind = sp.argmin(sp.absolute(rr-R[i,u]))
            l = Prt[:,r_ind,u]
            l[l==0] = 1e-100 # to avoid multiplication w 0
            L[:,u] = l

        L = sp.prod(L,axis=1)
        L = L / L.max()

        # L = sp.sum(sp.log(L),axis=1)

        Ls[i,:] = L # input time on first axis
    return Ls

Ls =  decode(r_avgs_)
# inspect
fig, axes = plt.subplots()
axes.matshow(Ls.T,origin='bottom')

# %% 
colors = sns.color_palette('viridis',n_colors=tt_dc.shape[0])
fig, axes = plt.subplots()

ysep = 0.2
for i in range(tt_dc.shape[0]):
    axes.plot(tt_dc,Ls[i,:]+i*ysep,lw=1,alpha=0.8,color=colors[i])
    idx = sp.argmax(Ls[i,:])
    axes.plot(tt_dc[idx],Ls[i,idx]+i*ysep,'.',alpha=0.5,color=colors[i])

# %% plot real time vs decoded time
fig, axes = plt.subplots()
tt_decoded = tt_dc[sp.argmax(Ls,axis=1)]
for i,v in enumerate(tt_decoded):
    axes.plot(tt_dc[i],v,'o',color=colors[i])
axes.plot(tt_dc,tt_dc,':',alpha=0.5,color='k')

# %% decode each trial and average

nTrials_test = Rates_test.shape[1]
# get the matrix Rs
# time vector of data
R = Rates_test[0,0]
dt = R.sampling_period.rescale('s').magnitude
tt = sp.arange(R.t_start.rescale('s').magnitude,R.t_stop.rescale('s').magnitude,dt)

# get avg firing rates
Rs = sp.zeros((tt.shape[0],nUnits,nTrials_test))
for u in range(nUnits):
    for j in range(nTrials_test):
        Rs[:,u,j] = Rates_test[u,j].magnitude.flatten()

# reinterpolate average rates to the decodable times
Rsi = interp1d(tt,Rs,axis=0)(tt_dc)

Ls = sp.zeros((tt_dc.shape[0],tt_dc.shape[0], nTrials_test))
for i in range(nTrials_test):
    Ls[:,:,i] = decode(Rsi[:,:,i])

# %%

fig , axes = plt.subplots()
ext = (tt_dc[0],tt_dc[-1],tt_dc[0],tt_dc[-1])
# axes.matshow(Ls[:,:,3].T,origin='bottom',vmin=0,vmax=1,extent=ext)
axes.matshow(sp.average(Ls,axis=2).T,origin='bottom',extent=ext)
axes.set_aspect('auto')

# %% 
Ls_avg = sp.average(Ls,axis=2)
colors = sns.color_palette('viridis',n_colors=tt_dc.shape[0])
fig, axes = plt.subplots()

ysep = 0.2
for i in range(tt_dc.shape[0]):
    axes.plot(tt_dc,Ls_avg[i,:]+i*ysep,lw=1,alpha=0.8,color=colors[i])
    idx = sp.argmax(Ls_avg[i,:])
    axes.plot(tt_dc[idx],Ls_avg[i,idx]+i*ysep,'.',alpha=0.5,color=colors[i])
