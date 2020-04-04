%matplotlib qt5

import matplotlib.pyplot as plt
import matplotlib as mpl
# import seaborn as sns

mpl.rcParams['figure.dpi']=331
plt.style.use('default')

import sys,os
import pickle, dill
from tqdm import tqdm

import scipy as sp
import numpy as np
import pandas as pd
import neo
import elephant as ele
import spikeglx as glx
import quantities as pq

import npxlib

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
# folder = "/media/georg/data/2019-08-08_JP1202_second_full/stim1"
# folder = "/media/georg/data/2019-08-22_JP1259_third_full/stim1"
# folder = "/media/georg/data/2019-08-23_JP1259_third_full_2/stim2"
# folder = "/media/georg/data/2019-08-23_JP1258_fourth_full/stim1/"
# folder = "/media/georg/data/2019-11-07_JP1184_GR_JC_dual_fiber/stim1"
# folder = "/media/georg/data/2019-12-03_JP1355_GR_full_awake_2/stim3_g0"
folder = "/home/georg/data"
os.chdir(folder)

import importlib
import params
importlib.reload(params)

# bin and ks2
bin_path = os.path.join(folder,params.dat_path)
# Reader = glx.Reader(bin_path) # FIXME make bin path optional
kilosort_folder = folder

ks2 = npxlib.read_kilosort2(kilosort_folder)
phy = npxlib.read_phy(kilosort_folder)
CInfo = phy['cluster_info']

# TODO work on this data subselection

# select: those with label good, and not put to noise
# the rest is definitely not interesting (for first pass data analysis)
good_inds = CInfo.loc[CInfo.group != 'noise'].groupby('KSLabel').get_group('good')['id']

from pathlib import Path
meta_path = Path(bin_path).with_suffix('.meta')
meta_data = glx.read_meta_data(meta_path)
fs = meta_data['imSampRate'] * pq.Hz
t_stop = meta_data['fileTimeSecs'] * pq.s
SpikeTrains = npxlib.read_spiketrains_nobin(ks2,fs=fs,t_stop=t_stop)
# SpikeTrains = npxlib.read_spiketrains(ks2,Reader)

# subselecting spiketrains keeping only good
SpikeTrains = [St for St in SpikeTrains if St.annotations['label'] in good_inds]

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


def ttl2event(ttl_path, fs):
    """  """
    # get ttl - extract if not done already
    if os.path.exists(ttl_path):
        onset_inds = sp.load(ttl_path)
    else:
        npxlib.get_TTL_onsets(bin_path,4) # FIXME HARDCODE ttl channel id
        onset_inds = sp.load(ttl_path)

    trig = neo.core.Event((onset_inds / fs).rescale(pq.s))

    return trig

# stim triggers
ttl_path = bin_path + '.ttl.npy' # HARDCODED
trig = ttl2event(ttl_path, fs)

# pack it all into a neo segment
Seg = neo.core.Segment()
for s in SpikeTrains:
    Seg.spiketrains.append(s)
Seg.events.append(trig)

nStims = len(Seg.events[0])
nUnits = len(Seg.spiketrains)

"""
 _______   _______ .______   .___________. __    __
|       \ |   ____||   _  \  |           ||  |  |  |
|  .--.  ||  |__   |  |_)  | `---|  |----`|  |__|  |
|  |  |  ||   __|  |   ___/      |  |     |   __   |
|  '--'  ||  |____ |  |          |  |     |  |  |  |
|_______/ |_______|| _|          |__|     |__|  |__|

"""

"""
analysis fixme here: 
make Df to contain the sectioning of the data as columns (STR true/false col)

"""

# postprocessing Df: fix firing rate colum
Df = CInfo.loc[CInfo.group != 'noise'].groupby('KSLabel').get_group('good')
Df.firing_rate = sp.array([float(fr.split(' ')[0]) for fr in Df.firing_rate])

Df = Df.drop('ContamPct',axis=1)
Df = Df.drop('KSLabel',axis=1)
Df = Df.drop('group',axis=1)
Df = Df.drop('n_spikes',axis=1)

depth_thresh = 2200 # this will have to be set in each recording

Df['region'] = ['STR' if v else 'CX' for v in Df.depth < depth_thresh]

# from this Df, inds are extracted for the analysis

"""
     ___      .__   __.      ___       __      ____    ____  _______. __       _______.
    /   \     |  \ |  |     /   \     |  |     \   \  /   / /       ||  |     /       |
   /  ^  \    |   \|  |    /  ^  \    |  |      \   \/   / |   (----`|  |    |   (----`
  /  /_\  \   |  . `  |   /  /_\  \   |  |       \_    _/   \   \    |  |     \   \
 /  _____  \  |  |\   |  /  _____  \  |  `----.    |  | .----)   |   |  | .----)   |
/__/     \__\ |__| \__| /__/     \__\ |_______|    |__| |_______/    |__| |_______/

"""
# %%
"""
estimate firing rates for the entire recording
-> all units in SpikeTrains, which are all accepted units after KS
"""


force_recalc = True
frates_path = Path(bin_path).with_suffix('.frates.dill')

if os.path.exists(frates_path) and not force_recalc:
    print("loading firing rates:", frates_path)

    with open(frates_path,'rb') as fH:
        frates = dill.load(fH)
else:
    frates = []
    binsize = 1*pq.ms

    for u in tqdm(range(nUnits),desc="calculating firing rates"): # all units (all good)
        frate = ele.statistics.instantaneous_rate(Seg.spiketrains[u],sampling_period=binsize,kernel=ele.kernels.GaussianKernel(sigma=50*pq.ms))
        frate.annotate(index=u)
        frates.append(frate)

    with open(frates_path,'wb') as fH:
        print("writing firing rates to disc: ", frates_path)
        dill.dump(frates,fH)

# %%
# z scored version is never stored to avoid cluttering
frates_z = []
import copy
for i, rate in enumerate(tqdm(frates,desc="z scoring frates")):
    r = copy.copy(rate)
    frate_z = ele.signal_processing.zscore(r)
    frates_z.append(frate_z)



#  %%
"""
compute pre / post
"""

def calc_stim_end(i, StimsDf):
    """ calculate stimulus stop """
    dur,n,f = StimsDf.iloc[i][['dur','n','f']]
    for q in range(int(n)):
        # start stop of each pulse in s
        start = q * 1/f
        stop = start + dur

    stim_end = stop * pq.s

    return stim_end 

# calc_stim_end(2,StimMap)

force_recalc = True

### pre
# window definitions: relative to stim offset
t_start = -1*pq.s

pre_mat_path = os.path.splitext(bin_path)[0] + '.pre.npy'

if os.path.exists(pre_mat_path) and not force_recalc:
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

# %%
"""
get significant upmodulated units
"""
def get_modulated_units(k,N_spikes_diff):
    """ per stim class k
    returns indices to the modulated units """
    inds = StimsDf.groupby(('stim_id','opto')).get_group((k,False)).index

    # t test for changed activity after stim
    sig_modulated = sp.stats.ttest_1samp(N_spikes_diff[:,inds],0,1)[1] < 0.05
    
    return sp.where(sig_modulated)[0]

# get number of mod units
sig_n = []
nStimClasses = len(StimsDf.groupby('stim_id'))

for k in range(nStimClasses):
    sig_n.append(get_modulated_units(k,N_spikes_diff))


# %%
"""
slice frates into stim groups
think about flexible representation

Rates is an array of nUnits x nStims that contains the AnalogSignal

"""

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

# %% 
"""
center of mass sorting
in: list of rates
out: argsorted inds
"""

k = 0 # first subsetting here!
stim_inds_stim = StimsDf.groupby(('stim_id','opto')).get_group((k,True)).index
stim_inds_nostim = StimsDf.groupby(('stim_id','opto')).get_group((k,False)).index

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
for i,unit_id in enumerate(range(nUnits)):
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
fig, axes = plt.subplots()

tvec = (Rates[0][0].times - Rates[0][0].times[0] + offset).rescale('s').magnitude

for i in range(len(rates_avg_stim)):
    ind = order[i]
    # ind = u
    # scaling 
    ysep = 2 # for z scored
    # ysep = 0.5*pq.Hz # for rates
    # sf = np.max(sp.average(R_s[ind],axis=1))
    axes.plot(tvec,rates_avg_stim[ind]+i*ysep,color='darkcyan',alpha=0.50)
    axes.plot(tvec,rates_avg_nostim[ind]+i*ysep,color='firebrick',alpha=0.50)
    axes.plot(tvec[cmass[ind]],i*ysep,'.',color='k',alpha=0.5)
    # axes.axhline(i*ysep,linestyle=':',color='k',alpha=0.5)

t_offset = 0.25*pq.s
label_stim_artifact(axes,k,StimMap,offset=t_offset)
axes.axvline(0)


# %% difference image
D = []
for i in range(len(rates_avg_stim)):
    rate_diff = rates_avg_stim[i] - rates_avg_nostim[i]
    D.append(rate_diff.magnitude.flatten())

D = sp.stack(D).T
D = D[:,order]

fig, axes = plt.subplots()
im = axes.matshow(D.T,vmin=-1,vmax=1,cmap='PiYG',origin='bottom',extent=(-1,3,0,nUnits))
axes.set_aspect('auto')

label_stim_artifact(axes,k,StimMap,offset=t_offset)
axes.axvline(0)
axes.set_xlabel('t (s)')
plt.colorbar(im,label='∆z')

out_path = os.path.join(folder,'diff_plot_'+str(k)+'.png')
# plt.savefig(out_path,dpi=300)

#%% single cell example plot

os.makedirs('plots',exist_ok=True)

for i in range(len(rates_avg_stim)):

    # i = 75
    ind = order[i]
    fig, axes = plt.subplots(nrows=3,sharex=True,figsize=[6.5,9])
    axes[0].plot(tvec,rates_avg_stim[ind],color='darkcyan',alpha=0.50,label='VTA+VPL stim')
    axes[0].plot(tvec,rates_avg_nostim[ind],color='firebrick',alpha=0.50,label='VPL stim only')
    axes[0].legend()

    # get rasters
    St = SpikeTrains[ind]

    St_all = []
    for t in events:
        st = St.time_slice(t+offset,t+window)
        St_all.append(st)

    St_stim = [St_all[i] for i in stim_inds_stim]
    St_nostim = [St_all[i] for i in stim_inds_nostim]

    # raster
    # fig, axes = plt.subplots(nrows=2,sharex=True,sharey=True)
    import seaborn as sns
    # sns.despine(axes[1])

    for i,St_trial in enumerate(St_nostim):
        spike_times = St_trial.times - St_trial.t_start + offset
        axes[1].plot(spike_times, [i]*len(spike_times),'.',color='firebrick')

    for i,St_trial in enumerate(St_stim):
        spike_times = St_trial.times - St_trial.t_start + offset
        axes[2].plot(spike_times, [i]*len(spike_times),'.',color='darkcyan')

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

    fig.suptitle('unit id: '+str(ind))
    fig.tight_layout()
    fig.savefig('plots/unit_'+str(ind)+'.png',dpi=300)
    plt.close(fig)








