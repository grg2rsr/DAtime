"""
 
                                         
   _ __  _   _ _ __ _ __   ___  ___  ___ 
  | '_ \| | | | '__| '_ \ / _ \/ __|/ _ \
  | |_) | |_| | |  | |_) | (_) \__ \  __/
  | .__/ \__,_|_|  | .__/ \___/|___/\___|
  |_|              |_|                   
 
"""


# %%
%matplotlib qt5

import matplotlib.pyplot as plt
import matplotlib as mpl
mpl.rcParams['figure.dpi'] = 331
import seaborn as sns

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

from helpers import *

# %% 

folder = Path("/home/georg/data/2019-12-03_JP1355_GR_full_awake_2/stim3_g0/")

os.chdir(folder)
import analysis_params

bin_file = folder / analysis_params.bin_file
path = bin_file.with_suffix('.neo.zfr.sliced')

with open(path,'rb') as fH:
    print("reading ", path)
    Segs = pickle.load(fH)
    print("... done")

# read stim related info
stim_path = folder / analysis_params.stim_file
stim_map_path = folder / analysis_params.stim_map_file

StimMap = pd.read_csv(stim_map_path, delimiter=',')
StimsDf = pd.read_csv(stim_path, index_col=0, delimiter=',')

# %% rearrange
nUnits = len(Segs[0].spiketrains)
nTrials = len(Segs)

Rates = sp.zeros((nUnits,nTrials),dtype='object')
SpikeTrains = sp.zeros((nUnits,nTrials),dtype='object')
ix = sp.arange(nTrials)

for i in range(nTrials):
    Rates[:,i] = Segs[i].analogsignals
    SpikeTrains[:,i] = Segs[i].spiketrains


# %% load stats
StatsDf = pd.read_csv(bin_file.with_suffix('.stim_stats.csv'))

"""
 
         _       _   
   _ __ | | ___ | |_ 
  | '_ \| |/ _ \| __|
  | |_) | | (_) | |_ 
  | .__/|_|\___/ \__|
  |_|                
 
"""
# %% 3 axes example plot for a neuron
# top: rates
# middle: raster with VPL
# bottom: raster with DA
# unit_id = 142

# %% 
stim_id = 1
sig_unit_ids = StatsDf.groupby(('stim_id','area','sig')).get_group((stim_id,'STR',True))['unit_id'].unique()

# get corresponding indices to unit_ids
all_ids = [st.annotations['id'] for st in Segs[0].spiketrains]
sig_unit_ix = [all_ids.index(id) for id in sig_unit_ids]

# %%
for unit_id,unit_ix in tqdm(zip(sig_unit_ids,sig_unit_ix)):
        
    stim_inds = StimsDf.groupby(('opto','stim_id')).get_group(('red',stim_id)).index
    stim_inds_opto = StimsDf.groupby(('opto','stim_id')).get_group(('both',stim_id)).index
   
    kw = dict(height_ratios=(1,1,len(stim_inds_opto)/len(stim_inds)))
    fig, axes = plt.subplots(nrows=3,sharex=True,gridspec_kw=kw)

    # average rates
    R = Rates[unit_ix,stim_inds]
    R_avg = average_asigs(R)
    axes[0].plot(R_avg.times,R_avg,color='firebrick',alpha=0.75,label='VPL stim')

    R = Rates[unit_ix,stim_inds_opto]
    R_avg = average_asigs(R)
    axes[0].plot(R_avg.times,R_avg,color='darkcyan',alpha=0.75,label='SNc/VTA+VPL stim')

    # spiketrain rasters
    Sts = SpikeTrains[unit_ix,stim_inds]
    for i, st in enumerate(Sts):
        spike_times = st.times
        axes[1].plot(spike_times, [i]*len(spike_times),'.',color='firebrick',markersize=2,alpha=0.5)

    Sts = SpikeTrains[unit_ix,stim_inds_opto]
    for i, st in enumerate(Sts):
        spike_times = st.times
        axes[2].plot(spike_times, [i]*len(spike_times),'.',color='darkcyan',markersize=2,alpha=0.5)

    # deco
    sns.despine(fig)
    axes[-1].set_xlabel('time (s)')
    [ax.set_ylabel('trial') for ax in axes[1:]]
    axes[0].set_ylabel('rate (z)')
    axes[0].legend(fontsize='8',loc='upper left')

    vpl_stim, = select(Segs[stim_inds[0]].epochs,'VPL_stims')
    da_stim, = select(Segs[stim_inds_opto[0]].epochs,'DA_stims')

    for ax in axes:
        add_epoch(ax,vpl_stim,color='firebrick')

    add_epoch(axes[0],da_stim,color='darkcyan',above=True)
    fig.tight_layout()

    fig.savefig('plots/unit_'+str(unit_id)+'.png',dpi=300)
    plt.close(fig)



"""
example units for talk

163 more spikes with DA
168 complex change in temp profile
170 da leads to increas in excitability
209 da smears out to later
273 mass shifts to later
341 complex change
719 more spikes with DA
720 DA responding and more complex changes
745 shifts to later
750 shifts to later
760 the weird mountain
858 shifts to later

"""
