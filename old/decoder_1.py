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


# %%                     _ 
#   _ __ ___  __ _  __| |
#  | '__/ _ \/ _` |/ _` |
#  | | |  __/ (_| | (_| |
#  |_|  \___|\__,_|\__,_|

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

# 
#    __       _              _       _        
#   / _| __ _| | _____    __| | __ _| |_ __ _ 
#  | |_ / _` | |/ / _ \  / _` |/ _` | __/ _` |
#  |  _| (_| |   <  __/ | (_| | (_| | || (_| |
#  |_|  \__,_|_|\_\___|  \__,_|\__,_|\__\__,_|
#                                             
# 


"""
 _______   _______   ______   ______    _______   _______ .______
|       \ |   ____| /      | /  __  \  |       \ |   ____||   _  \
|  .--.  ||  |__   |  ,----'|  |  |  | |  .--.  ||  |__   |  |_)  |
|  |  |  ||   __|  |  |     |  |  |  | |  |  |  ||   __|  |      /
|  '--'  ||  |____ |  `----.|  `--'  | |  '--'  ||  |____ |  |\  \----.
|_______/ |_______| \______| \______/  |_______/ |_______|| _| `._____|

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

# %% P(r|t) generation

# requested decoding time vector
dt = 0.05
tt_dc = sp.arange(0.2,2.5,dt)

# making the kdes
force_recalc = False
kdes_path = os.path.splitext(bin_path)[0] + 'kdes.npy'

if os.path.exists(kdes_path) and not force_recalc:
    print("loading kdes")
    with open(kdes_path,'rb') as fH:
        all_kdes = pickle.load(fH)
else:
    # making the estimates
    all_kdes = []
    for u in tqdm(range(nUnits)):
        kdes = []
        for i,t in enumerate(tt_dc):
            samples = []
            for n in range(nTrials):
                times = (t,t+dt) * pq.s
                samples.append(sp.average(Rates_[u,n].time_slice(*times).magnitude))
            kdes.append(sp.stats.gaussian_kde(samples))
        all_kdes.append(kdes)

    with open(kdes_path,'wb') as fH:
        pickle.dump(all_kdes,fH)

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
axes.matshow(Prt[:,:,10].T,origin='bottom',vmin=0,vmax=1e-1,extent=ext)
axes.set_aspect('auto')
axes.set_xlabel('time')
axes.set_ylabel('firing rate')

# %% first simple test: decoding the average trajectory
# time vector of data
dt = Rates_[0,0].sampling_period.rescale('s').magnitude
tt = sp.arange(Rates_[0,0].t_start.magnitude,Rates_[0,0].t_stop.magnitude,dt)

# get avg firing rates
r_avgs = sp.zeros((tt.shape[0],nUnits))
for u in range(nUnits):
    sigs = Rates_[u,:]
    S = sp.stack([sig.magnitude for sig in sigs], axis=1)[:,:,0]
    r_avgs[:,u] = sp.average(S,axis=1)

inds = sp.argsort(sp.argmax(r_avgs,0))
ext = (-1,3,0,nUnits)
plt.matshow(r_avgs.T[inds,:],origin='bottom',extent=ext)
plt.gca().set_aspect('auto')

# %% decode it

# declared above
# dt = 0.05
# tt_dc = sp.arange(0.2,2.5,dt)

# reinterpolate average rates to the decodable times
r_avgs_ = interp1d(tt,r_avgs,axis=0)(tt_dc)

Ls = sp.zeros((tt_dc.shape[0],tt_dc.shape[0]))
for i,t in enumerate(tt_dc):
    L = sp.zeros((tt_dc.shape[0],nUnits))
    for u in range(nUnits):
        # find closest present rate
        r_ind = sp.argmin(sp.absolute(rr-r_avgs_[i,u]))
        l = Prt[:,r_ind,u]
        l[l==0] = 1e-100 # to avoid multiplication w 0
        L[:,u] = l

    L = sp.prod(L,axis=1)
    L = L / L.max()

    # L = sp.sum(sp.log(L),axis=1)

    Ls[i,:] = L # input time on first axis

# inspect
fig, axes = plt.subplots()
axes.matshow(Ls.T,origin='bottom')

# %% 
colors = sns.color_palette('viridis',n_colors=tt_dc.shape[0])
fig, axes = plt.subplots()

for i in range(tt_dc.shape[0]):
    axes.plot(tt_dc,Ls[i,:],lw=1,alpha=0.8,color=colors[i])
    idx = sp.argmax(Ls[i,:])
    axes.plot(tt_dc[idx],Ls[i,idx],'o',color=colors[i])

# %% plot real time vs decoded time
fig, axes = plt.subplots()
tt_decoded = tt_dc[sp.argmax(Ls,axis=1)]
for i,v in enumerate(tt_decoded):
    axes.plot(tt_dc[i],v,'o',color=colors[i])
axes.plot(tt_dc,tt_dc,':',alpha=0.5,color='k')

# %%
"""
kmeans xvalid
decode per trial average after
"""
