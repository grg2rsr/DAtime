"""
 
                                         
   _ __  _   _ _ __ _ __   ___  ___  ___ 
  | '_ \| | | | '__| '_ \ / _ \/ __|/ _ \
  | |_) | |_| | |  | |_) | (_) \__ \  __/
  | .__/ \__,_|_|  | .__/ \___/|___/\___|
  |_|              |_|                   
 

reads the neo object
estimates firing rates for the entire recording
zscores

slices
"""


# %%
%matplotlib qt5
%load_ext autoreload
%autoreload 2

import matplotlib.pyplot as plt
import matplotlib as mpl
# import seaborn as sns

import sys,os
import pickle
from copy import copy
from tqdm import tqdm
from pathlib import Path

import scipy as sp
import numpy as np
import pandas as pd
import neo
import elephant as ele
# import spikeglx as glx
import quantities as pq

# import npxlib

plt.style.use('default')
mpl.rcParams['figure.dpi'] = 166

# %% file dialog
bin_path = utils.get_file_dialog()

# %% previous
bin_path = Path("/media/georg/htcondor/shared-paton/georg/DAtime/data/batch_of_10/2020-06-20_1a_JJP-00875_wt/stim1_g0/stim1_g0_t0.imec.ap.bin")
bin_path = Path("/media/georg/htcondor/shared-paton/georg/DAtime/data/batch_of_10/2020-06-22_3b_JJP-00869_wt/stim1_g0/stim1_g0_t0.imec.ap.bin")
bin_path = Path("/media/georg/htcondor/shared-paton/georg/DAtime/data/batch_of_10/2020-06-22_4a_JJP-00871_wt/stim1_g0/stim1_g0_t0.imec.ap.bin")
bin_path = Path("/media/georg/htcondor/shared-paton/georg/DAtime/data/batch_of_10/2020-06-23_4b_JJP-00871_wt/stim1_g0/stim1_g0_t0.imec.ap.bin")
bin_path = Path("/media/georg/htcondor/shared-paton/georg/DAtime/data/batch_of_10/2020-06-25_7a_JJP-00874_dh/stim1_g0/stim1_g0_t0.imec.ap.bin")
bin_path = Path("/media/georg/htcondor/shared-paton/georg/DAtime/data/batch_of_10/2020-06-26_7b_JJP-00874_dh/stim1_g0/stim1_g0_t0.imec.ap.bin")
bin_path = Path("/media/georg/htcondor/shared-paton/georg/DAtime/data/batch_of_10/2020-06-26_8a_JJP-00876_dh/stim1_g0/stim1_g0_t0.imec.ap.bin")

# from here 1.0
bin_path = Path("/media/georg/htcondor/shared-paton/georg/DAtime/data/batch_of_10/2020-06-27_8b_JJP-00876_dh/stim1_g0/stim1_g0_imec0/stim1_g0_t0.imec0.ap.bin")
# bin_path = Path("/media/georg/htcondor/shared-paton/georg/DAtime/data/batch_of_10/2020-06-29_10a_JJP-00870_dh/stim2_g0/stim2_g0_imec0/stim2_g0_t0.imec0.ap.bin")

folder = bin_path.parent

# %%
os.chdir(folder)

import analysis_params
bin_file = folder / analysis_params.bin_file
neo_path = bin_file.with_suffix('.neo')

# get data
with open(neo_path,'rb') as fH:
    Seg = pickle.load(fH)


# %% estimate firing rates
"""
estimate firing rates for the entire recording
-> all units in SpikeTrains, which are all accepted units after KS
"""
sigma = analysis_params.k_sigma * pq.ms
kernel = ele.kernels.GaussianKernel(sigma=sigma)
from CausalAlphaKernel import CausalAlphaKernel
kernel = CausalAlphaKernel(sigma)

for St in tqdm(Seg.spiketrains):
    frate = ele.statistics.instantaneous_rate(St, sampling_period=5*pq.ms, kernel=kernel)
    frate_z = ele.signal_processing.zscore(frate)
    frate_z.annotate(id=St.annotations['id'])
    frate_z.t_start = frate_z.t_start.rescale('s')
    Seg.analogsignals.append(frate_z)

with open(neo_path.with_suffix('.neo.zfr'), 'wb') as fH:
    print("writing Seg with firing rates ... ")
    pickle.dump(Seg,fH)
print("...done")

"""
 
       _ _      _             
   ___| (_) ___(_)_ __   __ _ 
  / __| | |/ __| | '_ \ / _` |
  \__ \ | | (__| | | | | (_| |
  |___/_|_|\___|_|_| |_|\__, |
                        |___/ 
 
"""

# %% write a sliced variant
os.chdir(neo_path.parent)
import analysis_params

pre = analysis_params.trial_pre * pq.s
post = analysis_params.trial_post * pq.s

nUnits = len(Seg.analogsignals)
nTrials = len(Seg.events[0])

Segs = []
for i,t in enumerate(tqdm(Seg.events[0].times, desc='slicing')):
    seg = Seg.time_slice(t+pre,t+post)
    seg.annotate(trial_index=i)

    # take care of the time shift
    for asig in seg.analogsignals:
        asig.t_start = pre

    spiketrains = []
    for st in seg.spiketrains:
        s = st.time_shift(-t)
        s.t_start, s.t_stop = pre, post
        spiketrains.append(s)
    seg.spiketrains = spiketrains
    
    events = []
    for event in seg.events:
        events.append(event.time_shift(-t))
    seg.events = events

    epochs = []
    for epoch in seg.epochs:
        epochs.append(epoch.time_shift(-t))
    seg.epochs = epochs

    Segs.append(seg)

with open(neo_path.with_suffix('.neo.zfr.sliced'), 'wb') as fH:
    print("writing sliced Segs ... ")
    pickle.dump(Segs,fH)
    print("... done")



"""
 
       _                  
    __| | ___  _ __   ___ 
   / _` |/ _ \| '_ \ / _ \
  | (_| | (_) | | | |  __/
   \__,_|\___/|_| |_|\___|
                          
 
"""


# %% loading previous
with open(neo_path.with_suffix('.neo.zfr'), 'rb') as fH:
    Seg = pickle.load(fH)


# %% loading sliced frates
with open(neo_path.with_suffix('.neo.zfr.sliced'), 'rb') as fH:
    Segs = pickle.load(fH)


"""
 
   _                           _   
  (_)_ __  ___ _ __   ___  ___| |_ 
  | | '_ \/ __| '_ \ / _ \/ __| __|
  | | | | \__ \ |_) |  __/ (__| |_ 
  |_|_| |_|___/ .__/ \___|\___|\__|
              |_|                  
 
"""
# %% inspect a segment
i = 0
window = (-1,3) * pq.s
t = Seg.events[0].times[i]
t_slice = window + t
s = Seg.time_slice(*t_slice)

fig, axes = plt.subplots()
axes.axvline(t)
axes.axvline(t-0.978*pq.s+0.25*pq.s)

ysep = 0.5*pq.dimensionless
for i,asig in enumerate(s.analogsignals):
    axes.plot(asig.times.rescale('s'),asig+ysep*i,color='k',alpha=0.5,lw=0.75)

for epoch in s.epochs:
    if epoch.annotations['label'] == 'VPL_stims':
        color = 'firebrick'
    else:
        color = 'darkcyan'

    for i in range(epoch.times.shape[0]):
        t = epoch.times[i]
        dur = epoch.durations[i]
        axes.axvspan(t,t+dur,color=color,alpha=0.5)

# %% check kernel asymmetry
# seems to work nicely!
i = 1
window = (-1,3) * pq.s
t = Seg.events[0].times[i]
t_slice = window + t
s = Seg.time_slice(*t_slice)

j = 2 # unit
asig = s.analogsignals[j]
st = s.spiketrains[j]

fig, axes = plt.subplots()
axes.plot(asig.times,asig)
for t in st.times:
    axes.axvline(t,color='k',alpha=0.5,lw=1)

# %% inspection

nUnits = len(Segs[0].spiketrains)
nTrials = len(Segs)

Rates = sp.zeros((nUnits,nTrials),dtype='object')
Sts = sp.zeros((nUnits,nTrials),dtype='object')
for j in range(nTrials):
    Rates[:,j] = Segs[j].analogsignals
    Sts[:,j] = Segs[j].spiketrains

# %%
stim_inds = sp.where(sp.array(Segs[0].events[0].annotations['stim_ids']) == 1)[0]

asigs = [Rates[0,i] for i in stim_inds]
sts = [Sts[0,i] for i in stim_inds]

for asig in asigs:
    plt.plot(asig.times,asig,color='k',lw=1)

for st in sts:
    for t in st.times:
        plt.gca().axvline(t)


# %% to check - go into this via inspection
# StatsDf

TrialInfo = pd.read_csv(folder / "TrialInfo.csv",index_col=0)
UnitInfo = pd.read_csv(folder / "UnitInfo.csv")

unit_id = 1536
stim_id = 0

stim_inds = TrialInfo.groupby('stim_id').get_group(stim_id).index

from helpers import select
Sts = []
for i in stim_inds:
    Sts.append(select(Segs[i].spiketrains, stim_id, key='id')[0])

# plot raster
fig, axes = plt.subplots()
ysep = 0.1
for i,st in enumerate(Sts):
    axes.plot(st.times.magnitude, sp.ones(st.times.shape[0]) + i*ysep,'.',color='k')


# %% averaging all firing rates to check clock alignment
from helpers import average_asigs
fr_avg = average_asigs(Seg.analogsignals)
ds = 10
fig, axes = plt.subplots()
axes.plot(fr_avg.times[::ds],fr_avg[::ds,0])
for t in Seg.events[0].times:
    axes.axvline(t,color='k',alpha=0.5)
# %%
