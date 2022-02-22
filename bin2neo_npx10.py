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
%load_ext autoreload
%autoreload 2
import importlib

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
import utils

from readSGLX import readMeta

plt.style.use('default')
# mpl.rcParams['figure.dpi'] = 331
mpl.rcParams['figure.dpi'] = 166

"""
 
 ########  ########    ###    ########  
 ##     ## ##         ## ##   ##     ## 
 ##     ## ##        ##   ##  ##     ## 
 ########  ######   ##     ## ##     ## 
 ##   ##   ##       ######### ##     ## 
 ##    ##  ##       ##     ## ##     ## 
 ##     ## ######## ##     ## ########  
 
"""

# %% file dialog
bin_path = utils.get_file_dialog()

# %% previous
# all npx 3A
# bin_path = Path("/media/georg/htcondor/shared-paton/georg/DAtime/data/batch_of_10/2020-06-20_1a_JJP-00875_wt/stim1_g0/stim1_g0_t0.imec.ap.bin")
# bin_path = Path("/media/georg/htcondor/shared-paton/georg/DAtime/data/batch_of_10/2020-06-22_3b_JJP-00869_wt/stim1_g0/stim1_g0_t0.imec.ap.bin")
# bin_path = Path("/media/georg/htcondor/shared-paton/georg/DAtime/data/batch_of_10/2020-06-22_4a_JJP-00871_wt/stim1_g0/stim1_g0_t0.imec.ap.bin")
# bin_path = Path("/media/georg/htcondor/shared-paton/georg/DAtime/data/batch_of_10/2020-06-23_4b_JJP-00871_wt/stim1_g0/stim1_g0_t0.imec.ap.bin")
# bin_path = Path("/media/georg/htcondor/shared-paton/georg/DAtime/data/batch_of_10/2020-06-25_7a_JJP-00874_dh/stim1_g0/stim1_g0_t0.imec.ap.bin")
# bin_path = Path("/media/georg/htcondor/shared-paton/georg/DAtime/data/batch_of_10/2020-06-26_7b_JJP-00874_dh/stim1_g0/stim1_g0_t0.imec.ap.bin")
# bin_path = Path("/media/georg/htcondor/shared-paton/georg/DAtime/data/batch_of_10/2020-06-26_8a_JJP-00876_dh/stim1_g0/stim1_g0_t0.imec.ap.bin")

# from here 1.0
bin_path = Path("/media/georg/htcondor/shared-paton/georg/DAtime/data/batch_of_10/2020-06-27_8b_JJP-00876_dh/stim1_g0/stim1_g0_imec0/stim1_g0_t0.imec0.ap.bin")
# bin_path = Path("/media/georg/htcondor/shared-paton/georg/DAtime/data/batch_of_10/2020-06-29_10a_JJP-00870_dh/stim2_g0/stim2_g0_imec0/stim2_g0_t0.imec0.ap.bin")

# %% take care of the correct paths and read phy / kilosort output

# get phy params file
folder = bin_path.parent
os.chdir(folder / "phyfiles")
import params
importlib.reload(params)
os.chdir(folder)

# read phy and ks2 output
kilosort_folder = folder / "phyfiles"
ks2 = npxlib.read_kilosort2(kilosort_folder, keys=['spike_clusters','spike_times'])
phy = npxlib.read_phy(kilosort_folder)

# %%
"""
 
 ##     ## ##    ## #### ######## #### ##    ## ########  #######  
 ##     ## ###   ##  ##     ##     ##  ###   ## ##       ##     ## 
 ##     ## ####  ##  ##     ##     ##  ####  ## ##       ##     ## 
 ##     ## ## ## ##  ##     ##     ##  ## ## ## ######   ##     ## 
 ##     ## ##  ####  ##     ##     ##  ##  #### ##       ##     ## 
 ##     ## ##   ###  ##     ##     ##  ##   ### ##       ##     ## 
  #######  ##    ## ####    ##    #### ##    ## ##        #######  
 

UnitInfo hold information regarding the sorted units
"""

UnitInfo = phy['cluster_info']

# manual selection
UnitInfo['Label'] = UnitInfo['KSLabel']

# transfer noise label - overwrites good KSLabel
UnitInfo.loc[UnitInfo['group'] == 'noise','Label'] = 'noise'

# ulitmately discard all non-good and noise
UnitInfo = UnitInfo.groupby('Label').get_group('good')

# %% depth related
# KS stores depth as 0 is lowest site, 4000 is highest.
# goal is to have an intuitive DV representation, with 0 being at brain surface
UnitInfo['depth'] = UnitInfo['depth'] - 4000

# %% plot
fig, axes = plt.subplots(figsize=[3,5])
values = UnitInfo.depth.values

# for v in values:
#     axes.axhline(v,alpha=0.25,color='k')

axes.hist(values, bins=75,orientation='horizontal')
axes.set_ylabel('DV depth [um]')
axes.set_xlabel('count')
fig.tight_layout()

# %% assign areas
os.chdir(folder)
import analysis_params

upper_sep = analysis_params.depth_sep

# add to UnitInfo
UnitInfo['area'] = 'CX'
UnitInfo.loc[UnitInfo['depth'] < upper_sep,'area'] = 'STR'

# if lower sep is set, set those to sth else
if hasattr(analysis_params,'depth_sep_2'):
    lower_sep = analysis_params.depth_sep_2
    UnitInfo.loc[UnitInfo['depth'] < lower_sep,'area'] = '?'
else:
    lower_sep = None

# %% colored histogram
fig, axes = plt.subplots(figsize=[2.915, 4.325])
bins = sp.linspace(-4000,0,100)
# axes.hist(UnitInfo['depth'],bins=bins,orientation='horizontal',edgecolor='k',linewidth=1)
axes.hist(UnitInfo.groupby('area').get_group('CX')['depth'],bins=bins,orientation='horizontal',color='gray',alpha=0.75)
axes.hist(UnitInfo.groupby('area').get_group('STR')['depth'],bins=bins,orientation='horizontal',color='#1f77b4',alpha=0.75)

axes.set_ylabel("DV depth (µm)")
axes.set_xlabel("count")
axes.set_title("unit count by depth")
sns.despine(fig)

axes.axhline(upper_sep,linestyle=':', color='k')
if lower_sep is not None:
    axes.axhline(lower_sep,linestyle=':', color='k')

fig.tight_layout()
os.makedirs(folder / "plots", exist_ok=True)
fig.savefig(folder / "plots" / 'depth_hist_colored.png',dpi=331)

# %% read spiketrains

# meta_path = Path(bin_path).with_suffix('.meta')
# meta_data = glx.read_meta_data(meta_path)
meta_data = readMeta(bin_path)

fs = float(meta_data['imSampRate']) * pq.Hz
t_stop = float(meta_data['fileTimeSecs']) * pq.s

# this returns spiketrains of all cells present in the recording
SpikeTrains = npxlib.read_spiketrains_2(ks2,fs=fs,t_stop=t_stop)
print("initial number of detected cells by Ks2: ", len(SpikeTrains))

# filter to only those in UnitInfo
ids = [St.annotations['id'] for St in SpikeTrains]
good_ids = UnitInfo['id']
ix = [ids.index(id) for id in good_ids]
SpikeTrains = [SpikeTrains[i] for i in ix]

# add UnitInfo to each SpikeTrain (for redundancy)
# attach ks2 IDs to each unit
for i,St in enumerate(SpikeTrains):
    unit_id = St.annotations['id']
    info = UnitInfo.loc[UnitInfo['id'] == unit_id].iloc[0].to_dict()
    St.annotations = info

UnitInfo.to_csv(folder/'UnitInfo.csv')

"""
 
 ######## ########  ####    ###    ##       #### ##    ## ########  #######  
    ##    ##     ##  ##    ## ##   ##        ##  ###   ## ##       ##     ## 
    ##    ##     ##  ##   ##   ##  ##        ##  ####  ## ##       ##     ## 
    ##    ########   ##  ##     ## ##        ##  ## ## ## ######   ##     ## 
    ##    ##   ##    ##  ######### ##        ##  ##  #### ##       ##     ## 
    ##    ##    ##   ##  ##     ## ##        ##  ##   ### ##       ##     ## 
    ##    ##     ## #### ##     ## ######## #### ##    ## ##        #######  
 
"""

# %% 
import analysis_params
force_recalc = True

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
stim_name = bin_path.parent.parent.stem
ttl_bin_path = bin_path.parent.parent / Path(stim_name+'_t0.nidq.bin')

ttl_path = bin_path.with_suffix('.ttl.npy')
if os.path.exists(ttl_path) and not force_recalc:
    trig_times = sp.load(ttl_path)
else:
    ttl_channel = analysis_params.ttl_channel
    trig_times = npxlib.get_TTL_onsets10b(ttl_bin_path,ttl_channel)

    trig_fname = bin_path.with_suffix('.ttl.npy')
    sp.save(trig_fname, trig_times)

# %% find clock offset and drift
"""
 
 ##     ##    ###    ##    ## ##     ##    ###    ##           ######  ##        #######   ######  ##    ##    ######## #### ##     ## 
 ###   ###   ## ##   ###   ## ##     ##   ## ##   ##          ##    ## ##       ##     ## ##    ## ##   ##     ##        ##   ##   ##  
 #### ####  ##   ##  ####  ## ##     ##  ##   ##  ##          ##       ##       ##     ## ##       ##  ##      ##        ##    ## ##   
 ## ### ## ##     ## ## ## ## ##     ## ##     ## ##          ##       ##       ##     ## ##       #####       ######    ##     ###    
 ##     ## ######### ##  #### ##     ## ######### ##          ##       ##       ##     ## ##       ##  ##      ##        ##    ## ##   
 ##     ## ##     ## ##   ### ##     ## ##     ## ##          ##    ## ##       ##     ## ##    ## ##   ##     ##        ##   ##   ##  
 ##     ## ##     ## ##    ##  #######  ##     ## ########     ######  ########  #######   ######  ##    ##    ##       #### ##     ## 
 
"""

# helper
def plt_raster_w_psth(SpikeTrains, t, pre=-0.5*pq.s, post=0.5*pq.s):
    fig, axes = plt.subplots(nrows=2, sharex=True)

    ysep=0.05
    Sts = []
    for i,St in enumerate(SpikeTrains):
        st = St.time_slice(t+pre,t+post)
        axes[0].plot(st.times,sp.ones(st.times.shape)*i*ysep,'o',markersize=.5,color='k',alpha=0.75)
        Sts.append(st)

    # psth
    sts = sp.concatenate([st.times.magnitude for st in Sts])
    count, bins, _ = axes[1].hist(sts,bins=sp.linspace(sts.min(),sts.max(),2000))
    
    for ax in axes:
        ax.axvline(t,color='k',alpha=0.25)

    return axes

# %% find clock offset on first trigger
pre = -2 * pq.s
post = 2 * pq.s

t1_ni = trig_times[0]
axes = plt_raster_w_psth(SpikeTrains, t1_ni, pre, post)

# fill in this value manually
t1_npx = 21.177633 * pq.s

for ax in axes:
    ax.axvline(t1_npx,color='r',alpha=0.25)

# %% 
pre = -2 * pq.s
post = 2 * pq.s

t2_ni = trig_times[-1]
axes = plt_raster_w_psth(SpikeTrains, t2_ni, pre, post)

# fill in this value manually
t2_npx = 4424.59901 * pq.s

for ax in axes:
    ax.axvline(t2_npx,color='r',alpha=0.25)

# %% 
from scipy.stats import linregress
slope, offset = linregress([t1_ni,t2_ni],[t1_npx,t2_npx])[0:2]
print(slope)
print(offset)

# %% correct 
import analysis_params

# ILLEGAL HACK to manually correct for sampling problem on 8b - 10b recordings
# just for inspection - remove afterwards
if hasattr(analysis_params,'ni_offset'):
    print('WARNING - manual clock async fix - WARNING')
    slope = analysis_params.ni_slope
    offset = analysis_params.ni_offset
    trig_times_corr = trig_times * slope + offset * pq.s
    trig_times_corr = trig_times_corr - analysis_params.VPL_onset * pq.s

    # adding timepoints to TrialInfo
    TrialInfo['t'] = trig_times_corr
else:
    TrialInfo['t'] = trig_times

# store
TrialInfo.to_csv(folder/"TrialInfo.csv")

# %% check for correction
axes = plt_raster_w_psth(SpikeTrains, TrialInfo.iloc[0]['t']*pq.s, pre-20*pq.s, post+10*pq.s)

# %%
axes = plt_raster_w_psth(SpikeTrains, TrialInfo.iloc[-1]['t']*pq.s, pre-10*pq.s, post+10*pq.s)


# %%
"""
 
                                                        
    ___  _ __   ___   _ __   ___  ___    ___  ___  __ _ 
   / _ \| '_ \ / _ \ | '_ \ / _ \/ _ \  / __|/ _ \/ _` |
  | (_) | | | |  __/ | | | |  __/ (_) | \__ \  __/ (_| |
   \___/|_| |_|\___| |_| |_|\___|\___/  |___/\___|\__, |
                                                  |___/ 
 
"""

import analysis_params

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
# %%
"""
 
 ##      ## ########  #### ######## #### ##    ##  ######       ######  ########     #######  ##    ## ##       ##    ## 
 ##  ##  ## ##     ##  ##     ##     ##  ###   ## ##    ##     ##    ##    ##       ##     ## ###   ## ##        ##  ##  
 ##  ##  ## ##     ##  ##     ##     ##  ####  ## ##           ##          ##       ##     ## ####  ## ##         ####   
 ##  ##  ## ########   ##     ##     ##  ## ## ## ##   ####     ######     ##       ##     ## ## ## ## ##          ##    
 ##  ##  ## ##   ##    ##     ##     ##  ##  #### ##    ##           ##    ##       ##     ## ##  #### ##          ##    
 ##  ##  ## ##    ##   ##     ##     ##  ##   ### ##    ##     ##    ##    ##       ##     ## ##   ### ##          ##    
  ###  ###  ##     ## ####    ##    #### ##    ##  ######       ######     ##        #######  ##    ## ########    ##    
 
"""


# %%
# some temp debug stuff

times = SpikeTrains[10].times.flatten()
fig, axes = plt.subplots()
axes.vlines(times,0,0.5,color='b')
stim_times = TrialInfo['t'].values
axes.vlines(stim_times,0.5,1,color='r')

# %%
# depth resolved raster
fig, axes = plt.subplots()

t_start = 53*pq.s
t_stop = 55*pq.s

seg = Seg.time_slice(t_start, t_stop)
ysep = 0.1
for i,st in enumerate(seg.spiketrains):
    try:
        axes.plot(st.times.flatten(), sp.ones(st.times.shape[0]) + i * ysep ,'.', color='k')
    except:
        pass

from helpers import add_epoch
for epoch in seg.epochs:
    if epoch.annotations['label'] == 'DA_stims':
        color='darkcyan'
    else:
        color='firebrick'
    epoch = epoch.time_shift(-.932253*pq.s)
    add_epoch(axes,epoch,color=color)





# %%






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
axes.hist(UnitInfo['depth'],bins=bins,orientation='horizontal',edgecolor='k',linewidth=2)
axes.hist(UnitInfo.groupby('area').get_group('CX')['depth'],bins=bins,orientation='horizontal',color='gray')
axes.hist(UnitInfo.groupby('area').get_group('STR')['depth'],bins=bins,orientation='horizontal',color='#1f77b4')

axes.set_ylabel("depth on probe (µm)")
axes.set_xlabel("count")
axes.set_title("unit count by depth")
sns.despine(fig)

# axes.axhline(sep,linestyle=':', color='k')
fig.tight_layout()
# fig.savefig('/home/georg/Desktop/ciss/depth_hist_colored.png',dpi=331)

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
# fig.savefig('spiketrains_depth.png',dpi=331)
# %%
