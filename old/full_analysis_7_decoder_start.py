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
import spikeglx as glx
import quantities as pq

import npxlib

plt.style.use('default')
mpl.rcParams['figure.dpi'] = 331
"""
 __    __   _______  __      .______    _______ .______          _______.
|  |  |  | |   ____||  |     |   _  \  |   ____||   _  \        /       |
|  |__|  | |  |__   |  |     |  |_)  | |  |__   |  |_)  |      |   (----`
|   __   | |   __|  |  |     |   ___/  |   __|  |      /        \   \
|  |  |  | |  |____ |  `----.|  |      |  |____ |  |\  \----.----)   |
|__|  |__| |_______||_______|| _|      |_______|| _| `._____|_______/

"""

def label_stim_artifact(axes,k,StimMap,offset):
    dur,n,f = StimMap.iloc[k][['dur','n','f']]
    for j in range(int(n)):
        # start stop of each pulse in s
        start = j * 1/f + offset.rescale('s').magnitude
        stop = start + dur
        axes.axvspan(start,stop,color='firebrick',alpha=0.5,linewidth=0)

def z(vec):
    Z = (vec-sp.average(vec)) / sp.std(vec)
    if sp.any(Z == sp.nan):
        import pdb
        pdb.set_trace()

    return Z


"""
.______      ___   .___________. __    __
|   _  \    /   \  |           ||  |  |  |
|  |_)  |  /  ^  \ `---|  |----`|  |__|  |
|   ___/  /  /_\  \    |  |     |   __   |
|  |     /  _____  \   |  |     |  |  |  |
| _|    /__/     \__\  |__|     |__|  |__|

"""
# %% 
# folder = "/media/georg/data/2020-03-04_GR_JP2111_full/stim1_g0" # the good looking one
# folder = "/media/georg/the_trunk/2020-03-04_GR_JP2111_full/stim1_g0"
folder = "/home/georg/data/2020-03-04_GR_JP2111_full/stim1_g0"

os.chdir(folder)

import importlib
import params
importlib.reload(params)

# bin and ks2
bin_path = os.path.join(folder,params.dat_path)
Reader = glx.Reader(bin_path) # FIXME make bin path optional
kilosort_folder = folder

ks2 = npxlib.read_kilosort2(kilosort_folder)
phy = npxlib.read_phy(kilosort_folder)
CInfo = phy['cluster_info']

from pathlib import Path
meta_path = Path(bin_path).with_suffix('.meta')
meta_data = glx.read_meta_data(meta_path)
fs = meta_data['imSampRate'] * pq.Hz
t_stop = meta_data['fileTimeSecs'] * pq.s

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

# %%
"""
     _______. _______  __       _______   ______ .___________. __    ______   .__   __.
    /       ||   ____||  |     |   ____| /      ||           ||  |  /  __  \  |  \ |  |
   |   (----`|  |__   |  |     |  |__   |  ,----'`---|  |----`|  | |  |  |  | |   \|  |
    \   \    |   __|  |  |     |   __|  |  |         |  |     |  | |  |  |  | |  . `  |
.----)   |   |  |____ |  `----.|  |____ |  `----.    |  |     |  | |  `--'  | |  |\   |
|_______/    |_______||_______||_______| \______|    |__|     |__|  \______/  |__| \__|

"""

# this returns spiketrains of all cells present in the recording
SpikeTrains = npxlib.read_spiketrains_nobin(ks2,fs=fs,t_stop=t_stop)
print("initial number of detected cells by Ks2: ", len(SpikeTrains))
# select: those with label good, and not put to noise
# the rest is definitely not interesting (for first pass data analysis)
# Df will have all the info
Df = CInfo.loc[CInfo.group != 'noise'].groupby('KSLabel').get_group('good')
labels = [St.annotations['label'] for St in SpikeTrains]

good_ids = Df['id']
SpikeTrains = [St for St in SpikeTrains if St.annotations['label'] in good_ids.values]
print("units left after good and not noise", len(SpikeTrains))

### minimum firing rate filtering
Df.firing_rate = sp.array([float(fr.split(' ')[0]) for fr in Df.firing_rate])
fr_thresh = 0.5
good_ids = Df.loc[Df.firing_rate > fr_thresh]['id']

SpikeTrains = [St for St in SpikeTrains if St.annotations['label'] in good_ids.values]
Df = Df.loc[good_ids.index]
print("units left after 0.5Hz firing rate threshold", len(SpikeTrains))

### depth related
# postprocessing Df: fix firing rate colum

# depth
Df['depth'] = Df['depth'] - 4000 
CInfo['depth'] = CInfo['depth'] - 4000 

# inspect
fig, axes = plt.subplots(figsize=[2.915, 4.325])
bins = sp.arange(-4000,0,100)
axes.hist(CInfo.depth,bins=bins,alpha=0.5,color='cornflowerblue',label='all KS2',orientation='horizontal')
axes.hist(Df.depth,bins=bins,color='cornflowerblue',label='after selection',orientation='horizontal')
axes.set_ylabel("depth (µm)")
axes.set_xlabel("count")
axes.set_title("cells by depth")

# determine sep
sep = -1800

axes.axhline(sep,linestyle=':', color='k')
axes.legend()
fig.tight_layout()

# label
Df['area'] = 'CX'
Df.loc[Df['depth'] < sep,'area'] = 'STR'

str_ids = Df.groupby('area').get_group('STR')['id']

### sort spiketrains according to depth
# inds = sp.argsort(Df['depth']).values
# SpikeTrains = [SpikeTrains[i] for i in inds]

# %%
"""
     _______.___________. __  .___  ___.
    /       |           ||  | |   \/   |
   |   (----`---|  |----`|  | |  \  /  |
    \   \       |  |     |  | |  |\/|  |
.----)   |      |  |     |  | |  |  |  |
|_______/       |__|     |__| |__|  |__|

"""

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


def ttl2event(ttl_path, glx_reader):
    """  """
    # get ttl - extract if not done already
    if os.path.exists(ttl_path):
        onset_inds = sp.load(ttl_path)
    else:
        npxlib.get_TTL_onsets(bin_path,3) # FIXME HARDCODE ttl channel id
        onset_inds = sp.load(ttl_path)

    fs = glx_reader.meta['imSampRate'] * pq.Hz
    trig = neo.core.Event((onset_inds / fs).rescale(pq.s))

    return trig

# stim triggers
ttl_path = bin_path + '.ttl.npy' # HARDCODED
trig = ttl2event(ttl_path, Reader)

# pack it all into a neo segment - why?
Seg = neo.core.Segment()
for s in SpikeTrains:
    Seg.spiketrains.append(s)
Seg.events.append(trig)

nStims = len(Seg.events[0])
nUnits = len(Seg.spiketrains)

# adding timepoints to StimsDf
StimsDf['t'] = trig.times

# %%
"""
     ___      .__   __.      ___       __      ____    ____  _______. __       _______.
    /   \     |  \ |  |     /   \     |  |     \   \  /   / /       ||  |     /       |
   /  ^  \    |   \|  |    /  ^  \    |  |      \   \/   / |   (----`|  |    |   (----`
  /  /_\  \   |  . `  |   /  /_\  \   |  |       \_    _/   \   \    |  |     \   \
 /  _____  \  |  |\   |  /  _____  \  |  `----.    |  | .----)   |   |  | .----)   |
/__/     \__\ |__| \__| /__/     \__\ |_______|    |__| |_______/    |__| |_______/

"""

# %% depth / raster plot
import seaborn as sns
colors = sns.color_palette('deep',2)

t_slice = (120,150) * pq.s

fig, axes = plt.subplots(figsize=[5.51 , 4.015])
for i,St in enumerate(tqdm(SpikeTrains)):
    times = St.time_slice(*t_slice).times

    area = Df[Df.id == St.annotations['label']]['area'].values
    depth = Df[Df.id == St.annotations['label']]['depth'].values

    if area  == 'STR':
        color = colors[0]
    else:
        color = colors[1]
    plt.plot(times, sp.ones(times.shape) * depth, '.', color=color, alpha=0.3, markersize=1)

axes.set_xlabel('time (s)')
axes.set_ylabel('depth (µm)')

times = Seg.events[0].time_slice(*t_slice).times
axes.vlines(times,0,100,lw=3,color='maroon')

sns.despine(ax=axes)
fig.tight_layout()
# fig.suptitle('depth resolved raster')

# %%

"""
estimate firing rates for the entire recording
-> all units in SpikeTrains, which are all accepted units after KS
"""
force_recalc = False
frates_path = os.path.splitext(bin_path)[0] + '.frates.npy'
fr_opts = dict(sampling_period=5*pq.ms, kernel=ele.kernels.GaussianKernel(sigma=50*pq.ms))

if os.path.exists(frates_path) and not force_recalc:
    print("loading firing rates:", os.path.basename(frates_path))

    with open(frates_path,'rb') as fH:
        frates = pickle.load(fH)
else:
    frates = []

    for u in tqdm(range(nUnits),desc="calculating firing rates"): # all units
        frate = ele.statistics.instantaneous_rate(Seg.spiketrains[u],**fr_opts)
        frate.annotate(index=u)
        frates.append(frate)

    with open(frates_path,'wb') as fH:
        print("writing firing rates to disc: ", os.path.basename(frates_path))
        pickle.dump(frates,fH)

# z-scored version
frates_z_path = os.path.splitext(bin_path)[0] + '.frates_z.npy'

if os.path.exists(frates_z_path) and not force_recalc:
    print("loading z-scored firing rates:", os.path.basename(frates_path))

    with open(frates_z_path,'rb') as fH:
        frates_z = pickle.load(fH)

else:
    frates_z = []
    import copy
    for i, rate in enumerate(tqdm(frates,desc="z-scoring frates")):
        r = copy.copy(rate)
        frate_z = ele.signal_processing.zscore(r)
        frates_z.append(frate_z)
    
    with open(frates_z_path,'wb') as fH:
        print("writing z-scoresd firing rates to disc: ", os.path.basename(frates_z_path))
        pickle.dump(frates_z,fH)

#  %%
"""
compute pre / post
"""

def calc_stim_end(i, StimMap):
    """ calculate stimulus stop """
    dur,n,f = StimMap.iloc[i][['dur','n','f']]
    for q in range(int(n)):
        # start stop of each pulse in s
        start = q * 1/f
        stop = start + dur

    stim_end = stop * pq.s

    return stim_end 

# calc_stim_end(2,StimMap)

force_recalc = False

### pre
# window definitions: relative to stim offset (onset!?!)
t_start = -1*pq.s

pre_mat_path = os.path.splitext(bin_path)[0] + '.pre.npy'

if os.path.exists(pre_mat_path) and not force_recalc:
    print("loading pre post data")
    N_spikes_pre = sp.load(pre_mat_path)
else:
    N_spikes_pre = sp.zeros((nUnits,nStims))

    for j in tqdm(range(nStims),desc="pre stim avg firing rate calc"):
        t = Seg.events[0][j]
        for i,st in enumerate(Seg.spiketrains):
            N_spikes_pre[i,j] = len(st.time_slice(t+t_start,t)) # divide by 1
    sp.save(pre_mat_path,N_spikes_pre)

### post
# window definitions: relative to stim offset
t_start = 0.1*pq.s + 0.25 * pq.s # HARDCODED stim delay
t_stop = 2.9*pq.s + 0.25 * pq.s

post_mat_path = os.path.splitext(bin_path)[0] + '.post.npy'

if os.path.exists(post_mat_path) and not force_recalc:
    N_spikes_post = sp.load(post_mat_path)
else:
    N_spikes_post = sp.zeros((nUnits,nStims))

    for j in tqdm(range(nStims),desc="post stim avg firing rate calc"):
        t = Seg.events[0][j]

        stim_end = calc_stim_end(StimsDf.iloc[j]['stim_id'],StimMap)
        
        for i,st in enumerate(Seg.spiketrains):
            N_spikes_post[i,j] = len(st.time_slice(t+stim_end+t_start,t+t_stop))
            N_spikes_post[i,j] = N_spikes_post[i,j] / (t_stop - t_start - stim_end).magnitude

    # N_spikes_post = N_spikes_post/(t_stop-t_start).magnitude
    sp.save(post_mat_path,N_spikes_post)

N_spikes_diff = N_spikes_post - N_spikes_pre

# %% plot this 

nStimClasses = len(StimsDf.groupby('stim_id'))

fig, axes = plt.subplots(ncols=nStimClasses, figsize=[5.285, 4.775], sharey=True)
for k in range(nStimClasses):
    inds = StimsDf.groupby(['stim_id','opto']).get_group((k,'red')).index
    im = axes[k].matshow(N_spikes_diff[:,inds],vmin=-15,vmax=15,cmap='PiYG')
    
axes[0].set_ylabel('cell id')
axes[1].set_xlabel('stim #')
fig.suptitle('∆spikes in 1s, post - pre')

cbar = fig.colorbar(im, ax=axes.ravel().tolist(), shrink=0.75,label='∆spikes')

for ax in axes:
    ax.set_xticklabels([])

# fig.tight_layout()

# %%
"""
get significant upmodulated units
"""
def get_modulated_units(k,N_spikes_diff):
    """ per stim class k
    returns indices to the modulated units """
    inds = StimsDf.groupby(['stim_id','opto']).get_group((k,'red')).index

    # t test for changed activity after stim
    sig_modulated = sp.stats.ttest_1samp(N_spikes_diff[:,inds],0,1)[1] < 0.05
    
    return sp.where(sig_modulated)[0]

# get number of mod units
sig_n = []
nStimClasses = len(StimsDf.groupby('stim_id'))

for k in range(nStimClasses):
    sig_n.append(get_modulated_units(k,N_spikes_diff))


# visualize this result
M = sp.zeros((N_spikes_diff.shape[0],nStimClasses))
for k in range(nStimClasses):
    M[sig_n[k],k] = 1

print([a.shape[0] for a in sig_n])

# %% plot as bars
# sort according to average mod
# sort_inds = sp.argsort(sp.average(N_spikes_diff,1))
# cells sorted according to their average response height - not
# N_spikes_diff_s = N_spikes_diff[sort_inds,:]
N_spikes_diff_s = N_spikes_diff

fig, axes = plt.subplots(ncols=nStimClasses,sharey=True,figsize=[12,3])

for k in range(nStimClasses):
    inds = StimsDf.groupby(['stim_id','opto']).get_group((k,'red')).index
    M = N_spikes_diff_s[:,inds]
    pos = sp.arange(M.shape[0])

    axes[k].plot(pos,sp.average(M,axis=1), '.', color='black', alpha=0.7)
    axes[k].errorbar(pos,sp.average(M,axis=1), yerr=sp.stats.sem(M,axis=1), color='gray', lw=1, alpha=0.5, fmt='None')

    # plot sig n on top
    # sig = sort_inds[sig_n[k]] # take care of the sorting!
    axes[k].plot(pos[sig_n[k]],sp.average(M,axis=1)[sig_n[k]], '.', color='red', alpha=0.7)
    
for ax in axes:
    ax.axhline(0,linestyle=':',color='k',lw=1, alpha=0.5)
    sns.despine(bottom=True,ax=ax)
    ax.set_xticks([])

axes[0].set_ylabel("∆spikes in 1s, post - pre")
fig.suptitle("identifying modulated cells")
axes[1].set_xlabel('cell id')

# %%
"""
slice frates into stim groups
think about flexible representation

Rates is an array of nUnits x nStims that contains the AnalogSignal
"""
force_recalc = True
frates_sliced_path = os.path.splitext(bin_path)[0] + '.frates_sliced.npy'
frates_z_sliced_path = os.path.splitext(bin_path)[0] + '.frates_z_sliced.npy'

if os.path.exists(frates_sliced_path) and not force_recalc:
    print("loading resliced firing rates:", os.path.basename(frates_sliced_path))
    with open(frates_sliced_path,'rb') as fH:
        Rates = pickle.load(fH)
    print("loading resliced z-scored firing rates:", os.path.basename(frates_z_sliced_path))
    with open(frates_a_sliced_path,'rb') as fH:
        Rates_z = pickle.load(fH)

else:
    window = 3*pq.s
    offset = -1*pq.s

    events, = Seg.events

    Rates = sp.zeros((nUnits,nStims),dtype='object')
    Rates_z = sp.zeros((nUnits,nStims),dtype='object')

    for u in tqdm(range(nUnits),desc="gathering rates at trials"):
        for i in range(nStims): # this iterates over stims
            t = events[i]

            r = frates[u].time_slice(t+offset,t+window)
            Rates[u,i] = r

            r = frates_z[u].time_slice(t+offset,t+window)
            Rates_z[u,i] = r

    print("writing resliced firing rates to disc: ", os.path.basename(frates_sliced_path))
    with open(frates_sliced_path,'wb') as fH:
        pickle.dump(Rates,fH)

    print("writing resliced z-scored firing rates to disc: ", os.path.basename(frates_z_sliced_path))
    with open(frates_z_sliced_path,'wb') as fH:
        pickle.dump(Rates_z,fH)

# for inspection - a slice
# if 0:
#     F = sp.stack([rate.magnitude.flatten() for rate in Rates[0,:]]).T
#     fig, axes = plt.subplots(nrows=2)
#     tvec = Rates[0,0].times.rescale('s') - Seg.events[0][inds[0]]
#     axes[0].plot(tvec,F,alpha=0.8,lw=1)
#     axes[0].plot(tvec,sp.average(F,axis=1),lw=2,color='k')
#     axes[1].matshow(F.T)
#     axes[1].set_aspect('auto')

# TODO pickle result



# %% firing rate plot

stim_inds = StimsDf.groupby(['stim_id','opto']).get_group((k,'red')).index
Rates_nostim = Rates[sig_n[k],:][:,stim_inds]

stim_inds = StimsDf.groupby(['stim_id','opto']).get_group((k,'blue')).index
Rates_stim = Rates[sig_n[k],:][:,stim_inds]

i = 10
fig, axes = plt.subplots()
for j in range(Rates_nostim.shape[1]):
    axes.plot(Rates_nostim[i,j], color='maroon', lw=1,alpha=0.5)

for j in range(Rates_stim.shape[1]):
    axes.plot(Rates_stim[i,j], color='cornflowerblue', lw=1,alpha=0.5)










# %% 
"""
center of mass sorting
in: list of rates
out: argsorted inds
"""

k = 2 # first subsetting here!
stim_inds_stim = StimsDf.groupby(['stim_id','opto']).get_group((k,'both')).index
stim_inds_nostim = StimsDf.groupby(['stim_id','opto']).get_group((k,'red')).index

sig_unit_ids = Df.iloc[sig_n[k]]['id']

# average rates helper
def average_rates(rates,offset):
    """ iterates over list/array rates and returns an asig with the average
    tvec will be offsetted by offset
    add other stats here later, this will break compatibility
    """
    if type(rates) == sp.ndarray:
        N = rates.shape[0]
    if type(rates) == list:
        N = len(rates)

    r = sp.stack([r.magnitude.flatten() for r in rates]).T
    r_avg = sp.average(r,axis=1)
    asig = neo.core.AnalogSignal(r_avg,units=rates[0].units,t_start=offset,sampling_rate=rates[0].sampling_rate)
    return asig

# for each unit, average all trials
rates_avg_nostim = []
rates_avg_stim = []

# sig_mod_unit_ids = sig_n[k]
# for i,unit_id in enumerate(sig_n[k]):
for i,unit_id in enumerate(sig_n[k]):
    # make average rate for this unit in the nostim case
    R = Rates_z[unit_id,stim_inds_nostim]
    r_avg = average_rates(R,offset)
    rates_avg_nostim.append(r_avg)

    # and the stim case
    R = Rates_z[unit_id,stim_inds_stim]
    r_avg = average_rates(R,offset)
    rates_avg_stim.append(r_avg)

# # kicking zeros
# for i in range(len(rates_avg_nostim)):
#     if sp.all(rates_avg_nostim_z[i] < 0.01):
#         asig = rates_avg_nostim[i]
#         rates_avg_nostim[i] = neo.core.AnalogSignal(sp.zeros(asig.shape)*asig.units,t_start=asig.t_start,sampling_period=asig.sampling_period)

# for i in range(len(rates_avg_stim)):
#     if sp.all(rates_avg_stim_z[i] < 0.01):
#         asig = rates_avg_stim[i]
#         rates_avg_stim[i] = neo.core.AnalogSignal(sp.zeros(asig.shape)*asig.units,t_start=asig.t_start,sampling_period=asig.sampling_period)

# center of mass sorting on no-stim
cmass = []
for i in range(len(rates_avg_nostim)):
    vals = rates_avg_nostim[i].magnitude.flatten()
    vals = vals - vals.min()
    cs = sp.cumsum(vals)
    mid = (cs[-1] - cs[0])/2
    cmass.append(sp.argmin(sp.absolute(cs - mid)))
order = sp.argsort(cmass)

if 0:
    # zscore averages
    rates_avg_nostim_z = []
    rates_avg_stim_z = []

    for r in rates_avg_nostim:
        if sp.all(r<0.001):
            r = neo.core.AnalogSignal(sp.zeros(r.shape)*r.units,t_start=r.t_start,sampling_period=r.sampling_period)
        rates_avg_nostim_z.append(ele.signal_processing.zscore(r))

    for r in rates_avg_stim:
        if sp.all(r<0.001):
            r = neo.core.AnalogSignal(sp.zeros(r.shape)*r.units,t_start=r.t_start,sampling_period=r.sampling_period)
        rates_avg_stim_z.append(ele.signal_processing.zscore(r))



"""
.______    __        ______   .___________.
|   _  \  |  |      /  __  \  |           |
|  |_)  | |  |     |  |  |  | `---|  |----`
|   ___/  |  |     |  |  |  |     |  |
|  |      |  `----.|  `--'  |     |  |
| _|      |_______| \______/      |__|

"""
# %% plot summary figure: all rates stim (blue) and no-stim (red)
# note: change this color code in the future to red blue purple?

# line plot
fig, axes = plt.subplots(ncols=3,figsize=[6.435, 3.97 ])

tvec = (Rates[0][0].times - Rates[0][0].times[0] + offset).rescale('s').magnitude

ysep = 0.5 # for z scored

for i in range(len(rates_avg_stim)):
    ind = order[i]
    # ind = u
    # ysep = 0.5*pq.Hz # for rates
    # sf = np.max(sp.average(R_s[ind],axis=1))

    axes[0].plot(tvec,rates_avg_nostim[ind]+i*ysep,lw=1,color='firebrick',alpha=0.50)
    axes[0].plot(tvec[cmass[ind]],i*ysep,'.',color='k',alpha=0.5,markersize=1)

    axes[1].plot(tvec,rates_avg_stim[ind]+i*ysep,lw=1,color='darkcyan',alpha=0.50)
    axes[1].plot(tvec[cmass[ind]],i*ysep,'.',color='k',alpha=0.5,markersize=1)

    axes[2].plot(tvec,rates_avg_stim[ind]+i*ysep,lw=1,color='darkcyan',alpha=0.50)
    axes[2].plot(tvec,rates_avg_nostim[ind]+i*ysep,lw=1,color='firebrick',alpha=0.50)
    axes[2].plot(tvec[cmass[ind]],i*ysep,'.',color='k',alpha=0.5,markersize=1)
    
    # axes.axhline(i*ysep,linestyle=':',color='k',alpha=0.5)

t_offset = 0.25*pq.s
for ax in axes:
    label_stim_artifact(ax,k,StimMap,offset=t_offset)
    sns.despine(ax=ax,left=True)
    ax.set_yticks([])
    # ax.set_xticks([0,1,2])

for ax in axes[1:]:
    ax.axvline(0,lw=1)

axes[1].set_xlabel('time (s)')
fig.tight_layout()

# %% difference image

nUnits = len(rates_avg_stim)

D = []
for i in range(len(rates_avg_stim)):
    rate_diff = rates_avg_stim[i] - rates_avg_nostim[i]
    D.append(rate_diff.magnitude.flatten())

D = sp.stack(D).T
D = D[:,order]

fig, axes = plt.subplots(figsize=[6.245, 3.475])
im = axes.matshow(D.T,vmin=-1,vmax=1,cmap='PiYG',origin='bottom',extent=(-1,3,0,nUnits))
axes.set_aspect('auto')

label_stim_artifact(axes,k,StimMap,offset=t_offset)
axes.axvline(0)
axes.set_xlabel('time (s)')
axes.set_ylabel('units')
plt.colorbar(im,label='firing rate ∆z',shrink=0.75)

#%% single cell example plot

SpikeTrains_sig = [SpikeTrains[i] for i in sig_n[k]]
os.makedirs('plots',exist_ok=True)

for i in range(len(rates_avg_stim)):
    # i = 26
    ind = order[i]
    fig, axes = plt.subplots(nrows=3,sharex=True,figsize=[3.505, 4.44 ])
    axes[0].plot(tvec,rates_avg_stim[ind],color='darkcyan',alpha=0.75,label='VTA+VPL stim')
    axes[0].plot(tvec,rates_avg_nostim[ind],color='firebrick',alpha=0.75,label='VPL stim only')
    axes[0].legend(fontsize='5',loc='upper right')

    # get rasters
    St = SpikeTrains_sig[ind]

    St_all = []
    for t in events:
        st = St.time_slice(t+offset,t+window)
        St_all.append(st)

    St_stim = [St_all[i] for i in stim_inds_stim]
    St_nostim = [St_all[i] for i in stim_inds_nostim]

    # raster
    for i,St_trial in enumerate(St_nostim):
        spike_times = St_trial.times - St_trial.t_start + offset
        axes[1].plot(spike_times, [i]*len(spike_times),'.',color='firebrick',markersize=2,alpha=0.5)

    for i,St_trial in enumerate(St_stim):
        spike_times = St_trial.times - St_trial.t_start + offset
        axes[2].plot(spike_times, [i]*len(spike_times),'.',color='darkcyan',markersize=2,alpha=0.5)

    axes[1].set_ylim(0,stim_inds_nostim.shape[0])
    axes[2].set_ylim(0,stim_inds_stim.shape[0])
    for ax in axes[1:]:
        ax.set_xlim(offset,window)

    for ax in axes:
        ax.axvline(0,color='darkcyan')
        label_stim_artifact(ax,k,StimMap,offset=t_offset)

    sns.despine(fig)
    axes[-1].set_xlabel('time (s)')
    [ax.set_ylabel('trial') for ax in axes[1:]]
    axes[0].set_ylabel('rate (z)')

    # get the area
    unit_id = sig_unit_ids.iloc[ind]
    area = str(Df.loc[Df.id==unit_id]['area'].values[0])

    fig.suptitle('unit id: '+str(unit_id) + ' - '+area, fontsize=8)
    fig.tight_layout()
    fig.subplots_adjust(top=0.926)

    fig.savefig('plots/unit_'+str(unit_id)+'.png',dpi=300)
    plt.close(fig)


# for the future ... 
# """
#  _______   _______   ______   ______    _______   _______ .______
# |       \ |   ____| /      | /  __  \  |       \ |   ____||   _  \
# |  .--.  ||  |__   |  ,----'|  |  |  | |  .--.  ||  |__   |  |_)  |
# |  |  |  ||   __|  |  |     |  |  |  | |  |  |  ||   __|  |      /
# |  '--'  ||  |____ |  `----.|  `--'  | |  '--'  ||  |____ |  |\  \----.
# |_______/ |_______| \______| \______/  |_______/ |_______|| _| `._____|

# """

# %% 
"""
this has to be completely redone and rethought

spike prob dist: per neuron, along time

what to calculate:
given a vector of rates
r what is the probability of timepoint t?
p(t|r) = p(r|t) * p(t) / p(r)

decoded time per ML
sweep across time with given r, and compute likelihood function
"""

# to get P(r|t)
# t needs to be binned
# assess firing rates at each t_i
# make a KDE on the firing rate prob for this slice
# pack together, list of KDEs?
# scaling: like a dist, sums to 1

# note - in the Rates, from above is offset -1 and window 3 s 
# and offset is to start of +DA
stim_inds = StimsDf.groupby(['stim_id','opto']).get_group((k,'red')).index
R = Rates_z[4,stim_inds]

unit_id = 4
t_samp = sp.arange(0.2,2.5,0.01) * pq.s
dt = sp.diff(t_samp)[0]
unit_ids = range(10)

kdes_all = []
for unit_id in tqdm(unit_ids):
    # heavy fixme: the 1s is because of the offset
    kdes = []
    for t in tqdm(t_samp):
        samples = []
        for stim_ind in stim_inds:
            R = Rates_z[unit_id,stim_ind]
            t_start = R.t_start
            values = R.time_slice(t+1*pq.s+t_start,t_start+1*pq.s+t+dt).magnitude
            samples.append(sp.average(values))
        kde = sp.stats.gaussian_kde(samples)
        kdes.append(kde)

    kdes_all.append(kdes)

kdes_path = os.path.splitext(bin_path)[0] + 'kdes.npy'
with open(kdes_path,'wb') as fH:
    pickle.dump(kdes_all,fH)

# testing with decoding the average trajectory

# get firing rates
r_vecs = sp.zeros((t_samp.shape[0],len(unit_ids)))

for i,t in enumerate(tqdm(t_samp)):
    for u,unit_id in enumerate(unit_ids):
        # get the average rate
        D = sp.concatenate([Rates_z[unit_id,stim_ind].magnitude for stim_ind in stim_inds],axis=1)
        r_avg = sp.average(D,axis=1)
        r_avg = neo.core.AnalogSignal(r_avg,units=pq.Hz,t_start=-1*pq.s,sampling_period=5*pq.ms)

        # put a time slice of the average rate in the matrix
        r_vecs[i,u] = sp.average(r_avg.time_slice(t,t+dt).magnitude)
        
# plt.matshow(r_vecs.T,origin='bottom')

df = 0.01

# evaluating the kdes
n = t_samp.shape[0]
Ls = sp.zeros((n,n,len(unit_ids)))
for u,unit_id in enumerate(tqdm(unit_ids)):
    for i,t in enumerate(t_samp):
        f = r_vecs[i,u]

        kdes = kdes_all[u]
        Ls[i,:,u] = [kde.integrate_box_1d(f,f+df) for kde in kdes]


L = Ls[:,:,6]
plt.matshow(L.T,origin='bottom')


# for checking: given a list of kdes, make the matrix image
kdes = kdes_all[6]
f_samp = sp.linspace(-6,6,100)
df = sp.diff(f_samp)[0]

K = sp.zeros((len(kdes),f_samp.shape[0]))
for k,kde in enumerate(tqdm(kdes)):
    for i,f in enumerate(f_samp):
        K[k,i] = kde.integrate_box_1d(f,f+df)

plt.matshow(K.T > 0.005,origin='bottom')

#
plt.plot(r_vecs[:,6])

# %%
# testing with it's own average rate
# avg rate of this unit

D = sp.concatenate([Rates_z[unit_id,stim_ind].magnitude for stim_ind in stim_inds],axis=1)
rs = sp.average(D,axis=1)
rs = neo.core.AnalogSignal(rs,units=pq.Hz,t_start=-1*pq.s,sampling_period=5*pq.ms)

# 
#
fs = []
for i,t in enumerate(t_samp):
    f = sp.average(rs.time_slice(t,t+dt).magnitude) # f will be the r vector with all rates at this timepoint
    fs.append(f)
fs = sp.array(fs)

L = sp.zeros((t_samp.shape[0],t_samp.shape[0]))
for i,f in enumerate(fs):
    L[i,:] = [kde.evaluate(f) for kde in kdes] # the horizontal slice

# L = sp.concatenate(ps,axis=1) # axis one is over time samples, axis 0 is from the neuron

plt.plot(L[0,:]) # this is P(t=t0|r=r0)
# high everywhere except when there is activity => dip. makes sense
# unclear where during baseline but unlikely to be during stim

# now: high rate
plt.plot(L[50,:])

plt.matshow(L.T[:80,:80],origin='bottom')
plt.colorbar()

# 
#                          _ _       
#   _ __ _____      ___ __(_) |_ ___ 
#  | '__/ _ \ \ /\ / / '__| | __/ _ \
#  | | |  __/\ V  V /| |  | | ||  __/
#  |_|  \___| \_/\_/ |_|  |_|\__\___|
#                                    
# 

# %% fake data generation

dt = 0.05
tt = sp.arange(0,5,dt)

nUnits = 10
nTrials = 50

Rates = sp.zeros((nUnits,nTrials),dtype='object')
SpikeTrains = sp.zeros((nUnits,nTrials),dtype='object')

# generating spike trains
fr_opts = dict(sampling_period=dt*pq.s, kernel=ele.kernels.GaussianKernel(sigma=50*pq.ms))

for i in tqdm(range(nUnits)):
    peak_time = tt[sp.random.randint(tt.shape[0])]
    rate_gen = sp.stats.distributions.norm(peak_time,0.5).pdf(tt) * 10
    asig = neo.core.AnalogSignal(rate_gen,t_start=tt[0]*pq.s,units=pq.Hz,sampling_period=dt*pq.s)
    for j in range(nTrials):
        st = ele.spike_train_generation.inhomogeneous_poisson_process(asig)
        SpikeTrains[i,j] = st

        r = ele.statistics.instantaneous_rate(st,**fr_opts)
        rz = ele.signal_processing.zscore(r)
        Rates[i,j] = rz

# %% spike train inspection
i = 9
Sts = SpikeTrains[i,:]
ysep=0.1
for j,St in enumerate(Sts):
    plt.plot(St.times,sp.ones((len(St.times))) * j * ysep,'.',color='k')

# %% inspect
i = 5
sigs = Rates[i,:]
S = sp.stack([sig.magnitude for sig in sigs], axis=1)[:,:,0]

for sig in sigs:
    plt.plot(sig.times,sig,alpha=0.5,lw=1)
plt.plot(sig.times,sp.average(S,axis=1),lw=2,color='k')

# %% P(r|t) generation

dr = 0.05
rr = sp.arange(-4,4,dr)

dt = 0.05
tt = sp.arange(0.2,2.5,dt)

nUnits = 10
nTrials = 50

# making the estimates
all_kdes = []
for u in tqdm(range(nUnits)):
    kdes = []
    for i,t in enumerate(tt):
        samples = []
        for n in range(nTrials):
            times = (t,t+dt) * pq.s
            samples.append(sp.average(Rates[u,n].time_slice(*times).magnitude))
        kdes.append(sp.stats.gaussian_kde(samples))
    all_kdes.append(kdes)

# %% filling P(r|t)
Prt = sp.zeros((tt.shape[0],rr.shape[0],nUnits))
for u in tqdm(range(nUnits)):
    kdes = all_kdes[u]
    for i,t in enumerate(tt):
        Prt[i,:,u] = [kdes[i].integrate_box_1d(r,r+dr) for r in rr]

# %%
plt.matshow(Prt[:,:,0].T,origin='bottom',vmin=0,vmax=1e-2)
plt.gca().set_aspect('auto')

# %% decoding with average: get avg

dt = 0.05
tt = sp.arange(0,5,dt)

r_avgs = sp.zeros((tt.shape[0],nUnits))
for u in range(nUnits):
    sigs = Rates[u,:]
    S = sp.stack([sig.magnitude for sig in sigs], axis=1)[:,:,0]
    r_avgs[:,u] = sp.average(S,axis=1)

plt.matshow(r_avgs.T,origin='bottom')

# inds = sp.argsort(sp.argmax(r_avgs,0))
# plt.matshow(r_avgs.T[inds,:],origin='bottom')
# %% 

dr = 0.05
rr = sp.arange(-4,4,dr)

from scipy.interpolate import interp1d
dt = 0.05
tt_dc = sp.arange(0.2,2.5,dt)
# reinterpolate average rates
r_avgs_ = interp1d(tt,r_avgs,axis=0)(tt_dc)

Ls = sp.zeros((tt_dc.shape[0],tt_dc.shape[0]))
for i,t in enumerate(tt_dc):
    L = sp.zeros((tt_dc.shape[0],nUnits))
    for u in range(nUnits):
        # closest present rate
        r_ind = sp.argmin(sp.absolute(rr-r_avgs_[i,u]))
        L[:,u] = (Prt[:,r_ind,u])
    L = sp.prod(L,axis=1)
    # L = sp.sum(sp.log(L),axis=1)
    L = L / L.max()
    Ls[i,:] = L # input time on first axis

plt.matshow(Ls.T,origin='bottom')

# WORKS