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
# import seaborn as sns

import sys,os
import pickle
from tqdm import tqdm
from pathlib import Path

import scipy as sp
import numpy as np
import pandas as pd
import neo
# import elephant as ele
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
# folder = Path("/home/georg/data/2019-12-03_JP1355_GR_full_awake_2/stim3_g0")
folder = Path("/home/georg/data/2020-03-04_GR_JP2111_full/stim1_g0")

os.chdir(folder)
import importlib
import params
importlib.reload(params)

# bin and ks2
bin_path = folder.joinpath(params.dat_path)
Reader = glx.Reader(bin_path)

kilosort_folder = folder
ks2 = npxlib.read_kilosort2(kilosort_folder)
phy = npxlib.read_phy(kilosort_folder)
CInfo = phy['cluster_info']

meta_path = Path(bin_path).with_suffix('.meta')
meta_data = glx.read_meta_data(meta_path)

fs = meta_data['imSampRate'] * pq.Hz
t_stop = meta_data['fileTimeSecs'] * pq.s

# this returns spiketrains of all cells present in the recording
SpikeTrains = npxlib.read_spiketrains(ks2,fs=fs,t_stop=t_stop)
print("initial number of detected cells by Ks2: ", len(SpikeTrains))

# attach ks2 IDs to each unit
for i,St in enumerate(SpikeTrains):
    St.annotations = dict(CInfo.loc[i])

"""
 
            _           _   
   ___  ___| | ___  ___| |_ 
  / __|/ _ \ |/ _ \/ __| __|
  \__ \  __/ |  __/ (__| |_ 
  |___/\___|_|\___|\___|\__|
                            
 
"""

# select: those with label good, and not put to noise
# the rest is definitely not interesting (for first pass data analysis)
# Df will have all the info

Df = CInfo.loc[CInfo.group != 'noise'].groupby('KSLabel').get_group('good')
good_ids = Df['id']

# labels = [St.annotations['label'] for St in SpikeTrains]

# subset
SpikeTrains = [St for St in SpikeTrains if St.annotations['id'] in good_ids.values]
print("units left after good and not noise", len(SpikeTrains))

### minimum firing rate filtering
Df.firing_rate = sp.array([float(fr.split(' ')[0]) for fr in Df.firing_rate])
fr_thresh = 0.25
good_ids = Df.loc[Df.firing_rate > fr_thresh]['id']

SpikeTrains = [St for St in SpikeTrains if St.annotations['id'] in good_ids.values]
Df = Df.loc[good_ids.index]
print("units left after 0.25Hz firing rate threshold", len(SpikeTrains))

"""
 
       _            _   _     
    __| | ___ _ __ | |_| |__  
   / _` |/ _ \ '_ \| __| '_ \ 
  | (_| |  __/ |_) | |_| | | |
   \__,_|\___| .__/ \__|_| |_|
             |_|              
 
"""

# depth
# kilosort stores inverse
Df['depth'] = Df['depth'] - 4000 

# determine sep
import analysis_params
importlib.reload(analysis_params)
sep = analysis_params.depth_sep

# %% inspect
if 1:
    fig, axes = plt.subplots(figsize=[2.915, 4.325])
    bins = sp.arange(-4000,0,100)
    axes.hist(Df.depth,bins=bins,color='cornflowerblue',orientation='horizontal')
    axes.set_ylabel("depth (µm)")
    axes.set_xlabel("count")
    axes.set_title("cells by depth")

    
    axes.axhline(sep,linestyle=':', color='k')
    fig.tight_layout()

# label
Df['area'] = 'CX'
Df.loc[Df['depth'] < sep,'area'] = 'STR'

# %% write this to the spiketrains
for i,St in enumerate(SpikeTrains):
    St.annotate(area=Df.iloc[i]['area'])

# %%
"""
     _______.___________. __  .___  ___.
    /       |           ||  | |   \/   |
   |   (----`---|  |----`|  | |  \  /  |
    \   \       |  |     |  | |  |\/|  |
.----)   |      |  |     |  | |  |  |  |
|_______/       |__|     |__| |__|  |__|

"""
# backup a copy
# 2DO

# read stims
""" nomenclature fix here: a map maps stim id to stim params """

stim_path = folder / analysis_params.stim_file
stim_map_path = folder / analysis_params.stim_map_file

StimMap = pd.read_csv(stim_map_path, delimiter=',',index_col=0)
StimsDf = pd.read_csv(stim_path, delimiter=',',index_col=0)

# expand on logical
StimsDf['blue'] = False
StimsDf['red'] = False
StimsDf.loc[StimsDf['opto'] == 'blue','blue'] = True
StimsDf.loc[StimsDf['opto'] == 'red','red'] = True
StimsDf.loc[StimsDf['opto'] == 'both','blue'] = True
StimsDf.loc[StimsDf['opto'] == 'both','red'] = True

def ttl2event(ttl_path, glx_reader, ttl_channel):
    """  """
    # get ttl - extract if not done already
    if os.path.exists(ttl_path):
        onset_inds = sp.load(ttl_path)
    else:
        npxlib.get_TTL_onsets(bin_path,ttl_channel)
        onset_inds = sp.load(ttl_path)

    fs = glx_reader.meta['imSampRate'] * pq.Hz
    trig = neo.core.Event((onset_inds / fs).rescale(pq.s))

    return trig

# stim triggers
ttl_path = bin_path.with_suffix('.ttl.npy')

ttl_channel = analysis_params.ttl_channel
trig = ttl2event(ttl_path, Reader,ttl_channel)
trig.annotate(label='trig', stim_ids=list(StimsDf.stim_id.values), opto=list(StimsDf.opto.values))

# adding timepoints to StimsDf
StimsDf['t'] = trig.times

# store
StimsDf.to_csv(stim_path)



# %% adding DA stim spans
da_times = trig.times[StimsDf['blue'] == True]
times = da_times - analysis_params.DA_onset * pq.s
dur = analysis_params.DA_duration * pq.s
DA_stims = neo.core.Epoch(times=times,durations=dur)
DA_stims.annotate(label='DA_stims')

# VPL
VPL_stims = neo.core.Epoch()

for i, t in enumerate(trig.times):
    k = StimsDf.loc[i,'stim_id']
    dur, n, f = StimMap.iloc[k][['dur','n','f']]
    times = sp.arange(n) * 1/f * pq.s + analysis_params.VPL_onset * pq.s
    epoch = neo.core.Epoch(times=times + t ,durations=dur*pq.s)
    VPL_stims = VPL_stims.merge(epoch)

VPL_stims.annotate(label='VPL_stims')
"""
 
       _                             
   ___| |_ ___  _ __ __ _  __ _  ___ 
  / __| __/ _ \| '__/ _` |/ _` |/ _ \
  \__ \ || (_) | | | (_| | (_| |  __/
  |___/\__\___/|_|  \__,_|\__, |\___|
                          |___/      
 
"""
# %% pack it all into a neo segment for cohesive storage
# all info is in here

Seg = neo.core.Segment()
for s in SpikeTrains:
    Seg.spiketrains.append(s)
Seg.events.append(trig)
Seg.epochs.append(DA_stims)
Seg.epochs.append(VPL_stims)

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
# with open(bin_path.with_suffix('.neo'), 'rb') as fH:
    # seg = pickle.load(fH)



















# %%
"""
.______          ___   ____    __    ____    ____    ____  __   ___________    __    ____
|   _  \        /   \  \   \  /  \  /   /    \   \  /   / |  | |   ____\   \  /  \  /   /
|  |_)  |      /  ^  \  \   \/    \/   /      \   \/   /  |  | |  |__   \   \/    \/   /
|      /      /  /_\  \  \            /        \      /   |  | |   __|   \            /
|  |\  \----./  _____  \  \    /\    /          \    /    |  | |  |____   \    /\    /
| _| `._____/__/     \__\  \__/  \__/            \__/     |__| |_______|   \__/  \__/

"""
# fig, axes = plt.subplots(figsize=[6.4 , 9.23])
# t_start = 123*pq.s
# t_stop = 123.5*pq.s
# Asig = npxlib.glx2asig(bin_path, t_start, t_stop)
# Asig = npxlib.global_mean(Asig,ks2)
# # Asig = npxlib.preprocess_ks2based(Asig,ks2)
# kw=dict(lw=0.5,alpha=0.8,color='k')
# npxlib.plot_npx_asig(Asig,ds=1, ysep=0.0002,kwargs=kw,ax=axes)

# import seaborn as sns
# sns.despine(ax=axes,left=True)
# axes.set_xlabel('time (s)')
# axes.set_ylabel('channel')
# axes.set_yticks([])
# fig.tight_layout()
