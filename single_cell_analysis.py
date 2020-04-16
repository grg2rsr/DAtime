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
# folder = Path("/home/georg/data/2020-03-04_GR_JP2111_full/stim1_g0")

os.chdir(folder)
import importlib
import analysis_params
importlib.reload(analysis_params)

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
    fig, axes = plt.subplots(nrows=3,sharex=True,gridspec_kw=kw,figsize=[5,4])

    # average rates
    R = Rates[unit_ix,stim_inds]
    R_avg = average_asigs(R)
    axes[0].plot(R_avg.times,R_avg,color='firebrick',alpha=0.75,label='VPL')

    R = Rates[unit_ix,stim_inds_opto]
    R_avg = average_asigs(R)
    axes[0].plot(R_avg.times,R_avg,color='darkcyan',alpha=0.75,label='VPL+SNc')

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
        add_epoch(ax,vpl_stim,color='firebrick',linewidth=0.5)

    add_epoch(axes[0],da_stim,color='darkcyan',above=True)

    kw = dict(top=0.953,
            bottom=0.12,
            left=0.135,
            right=0.977,
            hspace=0.165,
            wspace=0.2)
    fig.subplots_adjust(**kw)
    fig.savefig('plots/unit_'+str(unit_id)+'_stim%1d.png'%stim_id,dpi=300)
    plt.close(fig)

    # fig.tight_layout()



"""
examples for talk
for stim 1
133 inh by da
140 peak destroy  more spikes 
149 more spikes
156 more spikes
161 desync and more late
168 late more
202 more
686 less inh
710 da resp but shut down by vpl

252 

from stim 2 
50 more spikes
shift early
103 more spikes
115 shift late
149 shift late
156 more spikes
161 desync and late
163 later and more
209 more and later
252 more and later
298 less
321 less
712 less
720 less
727 more
737 blurred
760 more
777 less

"""


"""
THESE ARE THE OLD AND BAD ONES
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
