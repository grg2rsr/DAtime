# %%
%matplotlib qt5

import matplotlib.pyplot as plt
import matplotlib as mpl
import seaborn as sns

import sys,os
from time import time
import pickle
from tqdm import tqdm
from copy import copy
import scipy as sp
import numpy as np
import pandas as pd
import neo
import elephant as ele
import quantities as pq
from pathlib import Path

from decoder import *
from helpers import *

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
folder = Path("/home/georg/data/2019-12-03_JP1355_GR_full_awake_2/stim3_g0/")
# folder = Path("/home/georg/data/2020-03-04_GR_JP2111_full/stim1_g0")

os.chdir(folder)
import analysis_params
from importlib import reload
reload(analysis_params)

bin_file = folder / analysis_params.bin_file
path = bin_file.with_suffix('.neo.zfr.sliced')

# path = folder / "test_data.neo.zfr.sliced"

with open(path,'rb') as fH:
    print("reading ", path)
    Segs = pickle.load(fH)
    print("... done")

# read stim related info
stim_path = folder / analysis_params.stim_file
stim_map_path = folder / analysis_params.stim_map_file

StimMap = pd.read_csv(stim_map_path, delimiter=',')
StimsDf = pd.read_csv(stim_path, delimiter=',')

# reformatting from new data format 
nUnits = len(Segs[0].spiketrains)
nTrials = len(Segs)

Rates = sp.zeros((nUnits,nTrials),dtype='object')
for j in range(nTrials):
    Rates[:,j] = Segs[j].analogsignals


# %% UNIT SELECTION
area = 'STR'

StatsDf = pd.read_csv(bin_file.with_suffix('.stim_stats.csv'))
unit_ids = StatsDf.groupby(('area','stim_id','sig')).get_group((area,stim_k,True))['unit_id'].unique()

# get corresponding indices to unit_ids
all_ids = [st.annotations['id'] for st in Segs[0].spiketrains]
unit_ix = [all_ids.index(id) for id in unit_ids]

# subset
Rates_ = Rates[unit_ix,:]
# Rates_opto_ = Rates_opto_[unit_ix,:]

nUnits = Rates_.shape[0]


# %% xval with label permutation each run
stim_k = 1 # the stim to analyze
StimsDf_sub = StimsDf.groupby('stim_id').get_group(stim_k)

nLabelperm = 50

for l in range(nLabelperm):
    opto_labels = StimsDf_sub['opto'].values
    sp.random.shuffle(opto_labels)
    StimsDf_sub['opto'] = opto_labels
    stim_inds = StimsDf_sub.groupby('opto').get_group('red').index
    opto_inds = StimsDf_sub.groupby('opto').get_group('both').index

    # actual data slicing
    Rates__ = Rates_[:,stim_inds]
    Rates_opto_ = Rates_[:,opto_inds]

    nTrials = stim_inds.shape[0]
    nUnits = Rates_.shape[0]

    # keep an epoch
    vpl_stim, = select(Segs[stim_inds[0]].epochs,'VPL_stims')


    # XVAL STARTS HERE
    k = 5

    # randomly reorder trials
    rand_inds = sp.arange(nTrials)
    sp.random.shuffle(rand_inds)
    Rates__ = Rates__[:,rand_inds]

    # discard excess data
    ex = nTrials % k
    if ex != 0:
        Rates_cut = Rates__[:,:-ex]
        print("discarded %2d trials" % (ex))
    else:
        Rates_cut = Rates__

    Rates_split = sp.split(Rates_cut,k,1)

    # decoding times
    dt = 0.01
    tt_dc = sp.arange(-0.25,2.75,dt)

    # firing rate vector and binning
    dr = 0.2
    rr = sp.arange(-4,4,dr)

    # mem alloc
    Ls_avgs = sp.zeros((tt_dc.shape[0],tt_dc.shape[0],k))
    Ls_opto_avgs = sp.zeros((tt_dc.shape[0],tt_dc.shape[0],k))
    Ls_chance_avgs = sp.zeros((tt_dc.shape[0],tt_dc.shape[0],k))
    Ls_chance_opto_avgs = sp.zeros((tt_dc.shape[0],tt_dc.shape[0],k))

    tstart = time()
    ts_loop = []
    for ki in range(k):
        ti1 = time()
        Rates_test = Rates_split[ki]
        train_inds = list(range(k))
        train_inds.remove(ki)
        Rates_train = sp.concatenate([Rates_split[i] for i in train_inds],axis=1)

        # train and self validate
        Prt = calc_Prt(Rates_train, tt_dc, rr, bandwidth=0.25)
        # Prt = Prt / sp.sum(Prt,axis=0)[sp.newaxis,:,:]
        Ls = decode_trials(Prt, Rates_test, tt_dc, rr)

        # chance: shuffling Prt in time axis
        rand_inds = sp.arange(tt_dc.shape[0])
        sp.random.shuffle(rand_inds)
        Prt_shuff = Prt[rand_inds,:,:]
        Ls_chance = decode_trials(Prt_shuff, Rates_test, tt_dc, rr)

        # decode all opto trials with this
        Ls_opto = decode_trials(Prt, Rates_opto_, tt_dc, rr)

        # opto chance
        Ls_chance_opto = decode_trials(Prt_shuff, Rates_opto_, tt_dc, rr)

        # store
        Ls_avgs[:,:,ki] = sp.average(Ls,axis=2)
        Ls_opto_avgs[:,:,ki] = sp.average(Ls_opto,axis=2)
        Ls_chance_avgs[:,:,ki] = sp.average(Ls_chance,axis=2)
        Ls_chance_opto_avgs[:,:,ki] = sp.average(Ls_chance_opto,axis=2)

        # error quant
        tt_decoded = tt_dc[sp.argmax(sp.average(Ls,axis=2),axis=1)]
        rss = sp.sum((tt_decoded - tt_dc)**2)
        rss_avg = rss / tt_dc.shape[0]
        t_err = sp.sqrt(rss_avg)

        # timeit
        ti2 = time()
        t_loop = ti2-ti1
        ts_loop.append(t_loop)
        t_remain = (k-ki-1)*sp.average(ts_loop)
        print("loop time: %5.1f, left: %5.1fs = %3.1fm, rss/n: %8.4f, t_err: %8.4f" % (t_loop, t_remain, t_remain/60, rss_avg, t_err))

    tfin = time()
    print("total runtime: ", tfin-tstart)

    Ls_avg = sp.average(Ls_avgs,axis=2)
    tt_decoded = tt_dc[sp.argmax(Ls_avg,axis=1)]
    rss = sp.sum((tt_decoded - tt_dc)**2)
    print("error from avg: ", rss/tt_dc.shape[0])

    out_path = bin_file.with_suffix('.decoder.labelperm%1d.vpl.'%l+area+'.k%1d.dt=%.3f.npy'%(stim_k,dt))
    sp.save(out_path,Ls_avgs)

    out_path = bin_file.with_suffix('.decoder.labelperm%1d.opto.'%l+area+'.k%1d.dt=%.3f.npy'%(stim_k,dt))
    sp.save(out_path,Ls_opto_avgs)

    out_path = bin_file.with_suffix('.decoder.labelperm%1d.shuff.'%l+area+'.k%1d.dt=%.3f.npy'%(stim_k,dt))
    sp.save(out_path,Ls_chance_avgs)

    out_path = bin_file.with_suffix('.decoder.labelperm%1d.opto.shuff.'%l+area+'.k%1d.dt=%.3f.npy'%(stim_k,dt))
    sp.save(out_path,Ls_chance_opto_avgs)












"""
 
       _                  
    __| | ___  _ __   ___ 
   / _` |/ _ \| '_ \ / _ \
  | (_| | (_) | | | |  __/
   \__,_|\___/|_| |_|\___|
                          
 
"""

# %% load
nLabelperm = 20
area = 'STR'
dt = 0.01
tt_dc = sp.arange(-0.25,2.75,dt)
stim_k = 1

Ls_avgs = []
Ls_opto_avgs = []
Ls_chance_avgs = []
Ls_chance_opto_avgs = []

for l in range(nLabelperm):
    out_path = bin_file.with_suffix('.decoder.labelperm%1d.vpl.'%l+area+'.k%1d.dt=%.3f.npy'%(stim_k,dt))
    Ls_avgs.append(sp.load(out_path))

    out_path = bin_file.with_suffix('.decoder.labelperm%1d.opto.'%l+area+'.k%1d.dt=%.3f.npy'%(stim_k,dt))
    Ls_opto_avgs.append(sp.load(out_path))

    out_path = bin_file.with_suffix('.decoder.labelperm%1d.shuff.'%l+area+'.k%1d.dt=%.3f.npy'%(stim_k,dt))
    Ls_chance_avgs.append(sp.load(out_path))

    out_path = bin_file.with_suffix('.decoder.labelperm%1d.opto.shuff.'%l+area+'.k%1d.dt=%.3f.npy'%(stim_k,dt))
    Ls_chance_opto_avgs.append(sp.load(out_path))

Ls_avgs = sp.stack(Ls_avgs,axis=3)
Ls_opto_avgs = sp.stack(Ls_opto_avgs,axis=3)
Ls_chance_avgs = sp.stack(Ls_chance_avgs,axis=3)
Ls_chance_opto_avgs = sp.stack(Ls_chance_opto_avgs,axis=3)

# stim related
stim_inds = StimsDf.groupby(['stim_id','opto']).get_group((stim_k,'red')).index
vpl_stim, = select(Segs[stim_inds[0]].epochs,'VPL_stims')

# just average for now
Ls_avgs = sp.average(Ls_avgs,axis=3)
Ls_opto_avgs = sp.average(Ls_opto_avgs,axis=3)
Ls_chance_avgs = sp.average(Ls_chance_avgs,axis=3)
Ls_chance_opto_avgs = sp.average(Ls_chance_opto_avgs,axis=3)

"""
 
       _                    _                   _     
    __| | ___  ___ ___   __| | ___ _ __  __   _(_)___ 
   / _` |/ _ \/ __/ _ \ / _` |/ _ \ '__| \ \ / / / __|
  | (_| |  __/ (_| (_) | (_| |  __/ |     \ V /| \__ \
   \__,_|\___|\___\___/ \__,_|\___|_|      \_/ |_|___/
                                                      
 
"""
# %% 
Ls_avg = sp.average(Ls_avgs,axis=2)
Ls_opto_avg = sp.average(Ls_opto_avgs,axis=2)

Ls_chance_avg = sp.average(Ls_chance_avgs,axis=2) + 1.64*sp.std(Ls_chance_avgs,axis=2)
Ls_avg = Ls_avg - Ls_chance_avg
Ls_avg = sp.clip(Ls_avg,0,1)

Ls_chance_opto_avg = sp.average(Ls_chance_opto_avgs,axis=2) + 1.64*sp.std(Ls_chance_opto_avgs,axis=2)
Ls_opto_avg = Ls_opto_avg - Ls_chance_opto_avg
Ls_opto_avg = sp.clip(Ls_opto_avg,0,1)

Ls_D = Ls_opto_avg - Ls_avg

gkw = dict(height_ratios=(1,0.05))
fig , axes = plt.subplots(nrows=2, ncols=3, figsize=[10,4.6], gridspec_kw=gkw)

# axes[0,0].plot(tt_dc,tt_dc,':',alpha=0.35,color='w')
# axes[0,1].plot(tt_dc,tt_dc,':',alpha=0.35,color='w')
# axes[0,2].plot(tt_dc,tt_dc,':',alpha=0.35,color='k')

ext = (tt_dc[0],tt_dc[-1],tt_dc[0],tt_dc[-1])
v = 0.1
import colorcet as cc
kw = dict(vmin=0, vmax=0.25, cmap='viridis', extent=ext, origin="bottom")
kw_D = dict(vmin=-v,vmax=v,cmap='PiYG', extent=ext, origin="bottom")
kw_bar = dict(orientation="horizontal",label="p above chance", shrink=0.8)

im = axes[0,0].matshow(Ls_avg.T,**kw)
fig.colorbar(im,cax=axes[1,0],**kw_bar)

im = axes[0,1].matshow(Ls_opto_avg.T,**kw)
fig.colorbar(im,cax=axes[1,1],**kw_bar)

im = axes[0,2].matshow(Ls_D.T,**kw_D)
fig.colorbar(im,cax=axes[1,2],orientation="horizontal",label="difference", shrink=0.8)

axes[0,0].set_title('VPL stim only')
axes[0,1].set_title('VPL/SNc stim x-decoded')
axes[0,2].set_title('VPL/SNc stim - VPL stim')

axes[0,0].get_shared_x_axes().join(axes[0,0], axes[0,1])
axes[0,0].get_shared_x_axes().join(axes[0,0], axes[0,2])

axes[0,0].get_shared_y_axes().join(axes[0,0], axes[0,1])
axes[0,0].get_shared_y_axes().join(axes[0,0], axes[0,2])

for ax in axes[0,:]:
    add_stim(ax,vpl_stim,axis='xy',DA=False)
    # add_epoch(ax, vpl_stim, above=True, axis='xy')

    ax.set_aspect('equal')
    ax.set_xlabel('real time (s)')
    ax.set_ylabel('decoded time (s)')
    ax.xaxis.set_ticks_position('bottom')

    ax.set_xlim(tt_dc[0],tt_dc[-1])
    ax.set_ylim(tt_dc[0],tt_dc[-1])

for ax in axes[0,:]:
    ax.set_ylim(tt_dc[0],2.25)
    ax.set_xlim(tt_dc[0],2.25)

fig.tight_layout()
fig.savefig('/home/georg/Desktop/ciss/decoding_result_labelperm.png',dpi=331)
fig.savefig('/home/georg/Desktop/ciss/decoding_result_labelperm.svg',dpi=331)

"""
 
               _                 _   _             
    __ _ _ __ (_)_ __ ___   __ _| |_(_) ___  _ __  
   / _` | '_ \| | '_ ` _ \ / _` | __| |/ _ \| '_ \ 
  | (_| | | | | | | | | | | (_| | |_| | (_) | | | |
   \__,_|_| |_|_|_| |_| |_|\__,_|\__|_|\___/|_| |_|
                                                   
 
"""
# %%
import matplotlib.animation as animation

tt_decoded = tt_dc[sp.argmax(Ls_avg,axis=1)]
tt_decoded_opto = tt_dc[sp.argmax(Ls_opto_avg,axis=1)]

fig, axes = plt.subplots(figsize=[7,3.5])

artists = []
line, = axes.plot(tt_dc, sp.zeros(tt_dc.shape[0]),color='firebrick',lw=2)
artists.append(line)
line_opto, = axes.plot(tt_dc, sp.zeros(tt_dc.shape[0]),color='darkcyan',lw=2)
artists.append(line_opto)
vline_true = axes.axvline(color='k',linestyle=':')
artists.append(vline_true)

# axes.plot(tt_dc, sp.average(Ls_chance_avg,axis=1),color='gray')
# axes.plot(tt_dc, sp.average(Ls_chance_opto_avg,axis=1),color='gray')

axes.set_xlabel('decoder input time (s)')
axes.set_ylabel('p above chance')
sns.despine(fig, offset=10)
fig.tight_layout()
n_history = 10
history_alphas = sp.linspace(0.5,0,n_history)

vlines_dc_vpl = []
vlines_dc_da = []

for h in range(n_history):
    l = axes.axvline(ymin=0.9,ymax=1,lw=2,color='firebrick',alpha=history_alphas[h],zorder=-100)
    artists.append(l)

for h in range(n_history):
    l = axes.axvline(ymin=0.9,ymax=1,lw=2,color='darkcyan',alpha=history_alphas[h],zorder=-100)
    artists.append(l)

axes.set_ylim(-0.02,0.39)

def init():  # only required for blitting to give a clean slate.
    # line.set_ydata([np.nan] * tt_dc.shape[0])
    # line_opto.set_ydata([np.nan] * tt_dc.shape[0])
    # return line, line_opto, vline_true, vlines_dc_vpl, vlines_dc_da
    # return line, line_opto, vline_true
    # return vlines_dc_vpl, vlines_dc_da
    return artists

def animate(i):
    artists[0].set_ydata(Ls_avg[i,:])
    artists[1].set_ydata(Ls_opto_avg[i,:])
    artists[2].set_xdata(tt_dc[i])

    vlines_dc_vpl = artists[3:3+n_history]
    for j,vl in enumerate(vlines_dc_vpl):
        try:
            vl.set_xdata(tt_decoded[i-j])
        except:
            pass

    vlines_dc_da = artists[3+n_history:]
    for j,vl in enumerate(vlines_dc_da):
        try:
            vl.set_xdata(tt_decoded_opto[i-j])
        except:
            pass
        
    return artists


# frames = None
frames = sp.arange(tt_dc.shape[0])

# times = (0.5,1)
# inds = [sp.argmin(sp.absolute(tt_dc - t)) for t in times]
# frames = sp.arange(*inds,1)
# axes.set_xlim(0.4,1.2)

ani = animation.FuncAnimation(fig, animate, frames=frames, init_func=init, interval=50, blit=True, repeat=True)
# ani = animation.FuncAnimation(fig, animate, frames=frames, init_func=init, interval=50, repeat=True)

# %% save animation

moviewriter = animation.FFMpegFileWriter(fps=24)
moviewriter.setup(fig, 'my_movie.mp4', dpi=331)
n = frames.shape[0]
for j in range(n):
    animate(j)
    moviewriter.grab_frame()
moviewriter.finish()


# %% Prt inspect
u = 10
fig , axes = plt.subplots()
ext = (tt_dc[0],tt_dc[-1],rr[0],rr[-1])
axes.matshow(Prt[:,:,u].T,origin='bottom',extent=ext)
axes.set_aspect('auto')
axes.set_xlabel('time')
axes.set_ylabel('firing rate')



"""
 
   _                           _   
  (_)_ __  ___ _ __   ___  ___| |_ 
  | | '_ \/ __| '_ \ / _ \/ __| __|
  | | | | \__ \ |_) |  __/ (__| |_ 
  |_|_| |_|___/ .__/ \___|\___|\__|
              |_|                  
 
"""
# %% comparing sklearn and scipy kde bandwiths
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
kde_sk = KernelDensity(bandwidth=bw*2.2) # factor 2.2 makes them almost equal
kde_sk.fit(samples[:,sp.newaxis])
pdf_sk = sp.exp(kde_sk.score_samples(v[:,sp.newaxis]))
plt.plot(v,pdf_sk*dv)
















"""
 
       _             _                         
   ___(_)_ __   __ _| | ___   _ __ _   _ _ __  
  / __| | '_ \ / _` | |/ _ \ | '__| | | | '_ \ 
  \__ \ | | | | (_| | |  __/ | |  | |_| | | | |
  |___/_|_| |_|\__, |_|\___| |_|   \__,_|_| |_|
               |___/                           
 
"""
# %%

nTrials = Rates_.shape[1]

# another working copy: subset in units
Rates__ = copy(Rates_)
Rates_opto__ = copy(Rates_opto_)

# decoding times
dt = 0.005
tt_dc = sp.arange(-0.25,3.25,dt)

# firing rate vector and binning
dr = 0.1
rr = sp.arange(-4,4,dr)

# Ls_avgs_single = sp.zeros((tt_dc.shape[0],tt_dc.shape[0]))
# Ls_opto_avgs_single = sp.zeros((tt_dc.shape[0],tt_dc.shape[0]))
# Ls_chance_avgs_single = sp.zeros((tt_dc.shape[0],tt_dc.shape[0]))
# Ls_chance_opto_avgs_single = sp.zeros((tt_dc.shape[0],tt_dc.shape[0]))

Rates_test = Rates__
Rates_train = Rates__

# train and self validate
Prt = calc_Prt(Rates_train, tt_dc, rr, bandwidth=0.25)
Ls = decode_trials(Prt, Rates_test, tt_dc, rr)

# chance: shuffling Prt in time axis
rand_inds = sp.arange(tt_dc.shape[0])
sp.random.shuffle(rand_inds)
Prt_shuff = Prt[rand_inds,:,:]
Ls_chance = decode_trials(Prt_shuff, Rates_test, tt_dc, rr)

# decode all opto trials with this
Ls_opto = decode_trials(Prt, Rates_opto__, tt_dc, rr)

# opto chance
Ls_chance_opto = decode_trials(Prt_shuff, Rates_opto__, tt_dc, rr)




# %% 
Ls_corr = sp.average(Ls,axis=2) - sp.average(Ls_chance,axis=2)
Ls_opto_corr = sp.average(Ls_opto,axis=2) - sp.average(Ls_chance_opto,axis=2)
plt.matshow(Ls_opto_corr.T,origin='bottom')


