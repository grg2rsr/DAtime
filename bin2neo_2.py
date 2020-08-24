"""
 
                                         
   _ __  _   _ _ __ _ __   ___  ___  ___ 
  | '_ \| | | | '__| '_ \ / _ \/ __|/ _ \
  | |_) | |_| | |  | |_) | (_) \__ \  __/
  | .__/ \__,_|_|  | .__/ \___/|___/\___|
  |_|              |_|                   
 
this script reads the bin files and the kilosort2 output and writes a 
neo segment with all spiketrains

Kilosort information could be in a dataframe and best also each row attached
as each units annotations (redundant though)

"""



# %%
%matplotlib qt5

import matplotlib.pyplot as plt
import matplotlib as mpl
import seaborn as sns

import sys,os
from copy import copy

import pickle
from tqdm import tqdm
from pathlib import Path

import scipy as sp
import numpy as np
import pandas as pd
import neo
import spikeglx as glx
import quantities as pq

import npxlib

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
folder = Path("/home/georg/data/2019-12-03_JP1355_GR_full_awake_2/stim3_g0")
# folder = Path("/home/georg/data/2020-03-04_GR_JP2111_full/stim1_g0")

os.chdir(folder)
import importlib
import params
importlib.reload(params)

# glx
bin_path = folder.joinpath(params.dat_path)
Reader = glx.Reader(bin_path)

# ks2
kilosort_folder = folder
ks2 = npxlib.read_kilosort2(kilosort_folder)
phy = npxlib.read_phy(kilosort_folder)
CInfo = phy['cluster_info']

meta_path = Path(bin_path).with_suffix('.meta')
meta_data = glx.read_meta_data(meta_path)

fs = meta_data['imSampRate'] * pq.Hz
t_stop = meta_data['fileTimeSecs'] * pq.s

# get all spiketrains
SpikeTrains = npxlib.read_spiketrains_2(ks2,fs=fs,t_stop=t_stop)
print("initial number of detected cells by Ks2: ", len(SpikeTrains))

# attach ks2 IDs to each unit for redundancy
for i,St in enumerate(SpikeTrains):
    St.annotations = dict(CInfo.loc[i])

# %%
"""
 
       _                 
   ___| |_ ___  _ __ ___ 
  / __| __/ _ \| '__/ _ \
  \__ \ || (_) | | |  __/
  |___/\__\___/|_|  \___|
                         
 
"""
# is never needed?
# for i,St in enumerate(tqdm(SpikeTrains,desc="dumping spiketrains")):
#     unit_folder = folder/"neo_data"/str(St.annotations['id'])
#     os.makedirs(unit_folder,exist_ok=True)
#     with open(unit_folder/"spiketrain.neo",'wb') as fH:
#         pickle.dump(St,fH)


"""
 
   _   _       _ _   ___        __       
  | | | |_ __ (_) |_|_ _|_ __  / _| ___  
  | | | | '_ \| | __|| || '_ \| |_ / _ \ 
  | |_| | | | | | |_ | || | | |  _| (_) |
   \___/|_| |_|_|\__|___|_| |_|_|  \___/ 
                                         
 
"""
# %%

UnitInfo = copy(CInfo)

# firing rate 
UnitInfo.firing_rate = sp.array([float(fr.split(' ')[0]) for fr in CInfo.firing_rate])

# manual selection
UnitInfo['Label'] = UnitInfo['KSLabel']
UnitInfo.loc[UnitInfo['group'] == 'noise','Label'] = 'noise'

# depth
# kilosort stores inverse
UnitInfo['depth'] = UnitInfo['depth'] - 4000 

# determine sep
import analysis_params
importlib.reload(analysis_params)
sep = analysis_params.depth_sep

# add it - FIXME this is recording specific
# this could be replaced with a more automated function that relates
# penetration coordinates and depth with label -> IBL
UnitInfo['area'] = 'CX'
UnitInfo.loc[UnitInfo['depth'] < sep,'area'] = 'STR'

UnitInfo.to_csv(folder/'UnitInfo.csv')


"""
 
   _____     _       _ ___        __       
  |_   _| __(_) __ _| |_ _|_ __  / _| ___  
    | || '__| |/ _` | || || '_ \| |_ / _ \ 
    | || |  | | (_| | || || | | |  _| (_) |
    |_||_|  |_|\__,_|_|___|_| |_|_|  \___/ 
                                           
 
"""
# %% 
# read stims
""" nomenclature fix here: a map maps stim id to stim params """

stim_path = folder / analysis_params.stim_file
stim_map_path = folder / analysis_params.stim_map_file

StimMap = pd.read_csv(stim_map_path, delimiter=',',index_col=0)
TrialInfo = pd.read_csv(stim_path, delimiter=',',index_col=0)

# expand on logical
TrialInfo['blue'] = False
TrialInfo['red'] = False
TrialInfo.loc[TrialInfo['opto'] == 'blue','blue'] = True
TrialInfo.loc[TrialInfo['opto'] == 'red','red'] = True
TrialInfo.loc[TrialInfo['opto'] == 'both','blue'] = True
TrialInfo.loc[TrialInfo['opto'] == 'both','red'] = True

# stim triggers
ttl_path = bin_path.with_suffix('.ttl.npy')
if os.path.exists(ttl_path):
    trig_times = sp.load(ttl_path)
else:
    ttl_channel = analysis_params.ttl_channel
    trig_times = npxlib.get_TTL_onsets(bin_path,ttl_channel)
    
    trig_fname = bin_path.with_suffix('.ttl.npy')
    sp.save(trig_fname, trig_times)


# adding timepoints to TrialInfo
TrialInfo['t'] = trig_times

# store
TrialInfo.to_csv(folder/"TrialInfo.csv")


# %%
"""
 
                                                        
    ___  _ __   ___   _ __   ___  ___    ___  ___  __ _ 
   / _ \| '_ \ / _ \ | '_ \ / _ \/ _ \  / __|/ _ \/ _` |
  | (_) | | | |  __/ | | | |  __/ (_) | \__ \  __/ (_| |
   \___/|_| |_|\___| |_| |_|\___|\___/  |___/\___|\__, |
                                                  |___/ 
 
"""

Seg = neo.core.Segment()

# pack all spiketrains
for s in SpikeTrains:
    Seg.spiketrains.append(s)

# pack the trigger
trig = neo.core.Event(TrialInfo['t'].values*pq.s)
trig.annotate(label='trig', stim_ids=list(TrialInfo.stim_id.values), opto=list(TrialInfo.opto.values))
Seg.events.append(trig)

# adding DA stim spans
da_times = trig.times[TrialInfo['blue'] == True]
times = da_times - analysis_params.DA_onset * pq.s
dur = analysis_params.DA_duration * pq.s
DA_stims = neo.core.Epoch(times=times,durations=dur)
DA_stims.annotate(label='DA_stims')

# VPL
VPL_stims = neo.core.Epoch()

for i, t in enumerate(trig.times):
    k = TrialInfo.loc[i,'stim_id']
    dur, n, f = StimMap.iloc[k][['dur','n','f']]
    times = sp.arange(n) * 1/f * pq.s + analysis_params.VPL_onset * pq.s
    epoch = neo.core.Epoch(times=times + t ,durations=dur*pq.s)
    VPL_stims = VPL_stims.merge(epoch)

VPL_stims.annotate(label='VPL_stims')

Seg.epochs.append(VPL_stims)
Seg.epochs.append(DA_stims)


import pickle
with open(bin_path.with_suffix('.neo'), 'wb') as fH:
    print("writing neo segment ... ")
    pickle.dump(Seg,fH)
    print("...done")


"""
 
       _                  
    __| | ___  _ __   ___ 
   / _` |/ _ \| '_ \ / _ \
  | (_| | (_) | | | |  __/
   \__,_|\___/|_| |_|\___|
                          
 
"""



















"""
 
   _                 _ 
  | | ___   __ _  __| |
  | |/ _ \ / _` |/ _` |
  | | (_) | (_| | (_| |
  |_|\___/ \__,_|\__,_|
                       
 
"""
# %%
with open(bin_path.with_suffix('.neo'), 'rb') as fH:
    Seg = pickle.load(fH)

"""
 
       _            _   _       _     _     _        
    __| | ___ _ __ | |_| |__   | |__ (_)___| |_ ___  
   / _` |/ _ \ '_ \| __| '_ \  | '_ \| / __| __/ _ \ 
  | (_| |  __/ |_) | |_| | | | | | | | \__ \ || (_) |
   \__,_|\___| .__/ \__|_| |_| |_| |_|_|___/\__\___/ 
             |_|                                     
 
"""

# %% two colored histogram
fig, axes = plt.subplots(figsize=[2.915, 4.325])
bins = sp.arange(-4000,0,100)
axes.hist(Df['depth'],bins=bins,orientation='horizontal',edgecolor='k',linewidth=2)
axes.hist(Df.groupby('area').get_group('CX')['depth'],bins=bins,orientation='horizontal',color='gray')
axes.hist(Df.groupby('area').get_group('STR')['depth'],bins=bins,orientation='horizontal',color='#1f77b4')

axes.set_ylabel("depth on probe (µm)")
axes.set_xlabel("count")
axes.set_title("unit count by depth")
sns.despine(fig)

axes.axhline(sep,linestyle=':', color='k')
fig.tight_layout()
fig.savefig('/home/georg/Desktop/ciss/depth_hist_colored.png',dpi=331)

# %%
"""
.______          ___   ____    __    ____    ____    ____  __   ___________    __    ____
|   _  \        /   \  \   \  /  \  /   /    \   \  /   / |  | |   ____\   \  /  \  /   /
|  |_)  |      /  ^  \  \   \/    \/   /      \   \/   /  |  | |  |__   \   \/    \/   /
|      /      /  /_\  \  \            /        \      /   |  | |   __|   \            /
|  |\  \----./  _____  \  \    /\    /          \    /    |  | |  |____   \    /\    /
| _| `._____/__/     \__\  \__/  \__/            \__/     |__| |_______|   \__/  \__/

"""
fig, axes = plt.subplots(figsize=[6.4 , 9.23])
t_start = 123*pq.s
t_stop = 123.5*pq.s
Asig = npxlib.glx2asig(bin_path, t_start, t_stop)
Asig = npxlib.global_mean(Asig,ks2)
# Asig = npxlib.preprocess_ks2based(Asig,ks2)
kw=dict(lw=0.5,alpha=0.8,color='k')
npxlib.plot_npx_asig(Asig,ds=1, ysep=0.0002,kwargs=kw,ax=axes)

import seaborn as sns
sns.despine(ax=axes,left=True)
axes.set_xlabel('time (s)')
axes.set_ylabel('channel')
axes.set_yticks([])
fig.tight_layout()

"""
 
                               _ 
  __  _____ ___  _ __ _ __ ___| |
  \ \/ / __/ _ \| '__| '__/ _ \ |
   >  < (_| (_) | |  | | |  __/ |
  /_/\_\___\___/|_|  |_|  \___|_|
                                 
 
"""
# %% 
import elephant as ele
us = [0,1,21,3,4,5,6]
fig, axes = plt.subplots(nrows=len(us),sharex=True,figsize=[3,5])
for i,u in enumerate(us):
    St = Seg.spiketrains[u]
    bst = ele.conversion.BinnedSpikeTrain(St,binsize=1*pq.ms)

    asig, inds = ele.spike_train_correlation.cross_correlation_histogram(bst,bst,window=(-50,50))
    asig[sp.where(inds==0)[0]] = 0
    x = inds
    y = asig.magnitude.flatten()
    axes[i].fill_between(x,y, step='mid')
    axes[i].axvspan(-1.5,1.5,color='gray',lw=0,alpha=0.5)

fig.suptitle('spiketrain autocorrelations')
sns.despine(fig=fig)
axes[-1].set_xlabel('lag (ms)')
for ax in axes:
    ax.set_yticks([])

fig.savefig('autocorr.png',dpi=331)

# %%

"""
 
             _ _        _             _           
   ___ _ __ (_) | _____| |_ _ __ __ _(_)_ __  ___ 
  / __| '_ \| | |/ / _ \ __| '__/ _` | | '_ \/ __|
  \__ \ |_) | |   <  __/ |_| | | (_| | | | | \__ \
  |___/ .__/|_|_|\_\___|\__|_|  \__,_|_|_| |_|___/
      |_|                                         
 
"""
from helpers import add_epoch

fig, axes = plt.subplots(figsize=[7.5,3.8])
t_start = 123*pq.s
t_stop = t_start + 35*pq.s

seg = Seg.time_slice(t_start,t_stop)

for St in seg.spiketrains:
    depth = St.annotations['depth']
    if St.annotations['area'] == 'STR':
        color = '#1f77b4'
    else:
        color = 'gray'
    axes.plot(St.times,sp.ones(St.times.shape)*depth,'o',markersize=.5,color=color,alpha=0.75)
    
    hmin, hmax = 1.01, 1.05

    for time in seg.events[0].times:
        axes.axvspan(time+0.25*pq.s, time+0.5*pq.s,hmin,hmax,clip_on=False,alpha=0.5,color='firebrick',linewidth=0)
    # for epoch in seg.epochs:
    #     if epoch.annotations['label'] == 'VPL_stims':
    #         for time in epoch.times:
            # add_epoch(axes,epoch,color='firebrick', alpha=0.25, linewidth=2,above=True)   

sns.despine(fig=fig)
axes.set_ylabel('depth (µm)')
axes.set_xlabel('time (s)')
fig.tight_layout()
fig.savefig('spiketrains_depth.png',dpi=331)