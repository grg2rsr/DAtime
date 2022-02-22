# %%
import scipy as sp
import numpy as np
import matplotlib.pyplot as plt
%matplotlib qt5
import matplotlib as mpl
mpl.rcParams['figure.dpi'] = 166

from pathlib import Path

from readSGLX import readMeta, SampRate, makeMemMapRaw, GainCorrectIM, GainCorrectNI, ExtractDigital

# %% extract digital 
# bin_path = Path("/media/georg/htcondor/shared-paton/georg/sync_testing_2_g0/sync_testing_2_g0_t0.nidq.bin")
bin_path = Path("/media/georg/htcondor/shared-paton/georg/sync_testing_5_g0/sync_testing_5_g0_t0.nidq.bin")

def get_all_digital(bin_path, dw=0, dLineList=range(8)):
    """ dw and dLineList hardcoded - unlikely to ever change """
    print("reading digital from %s" % bin_path)

    meta = readMeta(bin_path)
    n_samples = float(meta['fileTimeSecs']) * float(meta['niSampRate'])
    if n_samples % 1 != 0.0:
        print("problem: number of samples is not an integer" )
    else:
        n_samples = int(n_samples)
    print("number of samples in ni bin file samples: %i" % n_samples)

    firstSamp = 0
    lastSamp = int(n_samples-1)

    rawData = makeMemMapRaw(bin_path, meta)

    # get digital data for the selected lines
    digArray = ExtractDigital(rawData, firstSamp, lastSamp, dw, range(8), meta)
    return digArray

# digArray = get_all_digital(bin_path)

# %%
# bin_path = Path("/media/georg/htcondor/shared-paton/georg/sync_testing_2_g0/sync_testing_2_g0_t0.nidq.bin")
bin_path = Path("/media/georg/htcondor/shared-paton/georg/sync_testing_5_g0/sync_testing_5_g0_t0.nidq.bin")

def get_trigger_times(digArray, meta, ch):
    # meta = readMeta(bin_path)
    # digArray = get_all_digital(bin_path)

    trig_data = digArray[ch,:]

    # get onset_inds
    inds = sp.where(sp.diff(trig_data) == 1)[0]

    # sampling rate
    # sRate = float(meta['niSampRate'])

    # TODO figure out how to deal with first sample - add to time or not?
    # t_offset = int(meta['firstSample'])/sRate

    times = inds * 1/SampRate(meta) # float(meta['niSampRate'])# - t_offset
    return inds, times

meta = readMeta(bin_path)
digArray = get_all_digital(bin_path)
# %%
sync_inds, sync_times = get_trigger_times(digArray, meta, 0)
trig_inds, trig_times = get_trigger_times(digArray, meta, 1)

# %%
bin_path = Path("/media/georg/htcondor/shared-paton/georg/sync_testing_2_g0/sync_testing_2_g0_t0.nidq.bin")
meta = readMeta(bin_path)
t_total = float(meta['fileTimeSecs'])
sRate = float(meta['niSampRate'])
firstSamp = int(meta['firstSample'])
n_samples = int(float(meta['fileTimeSecs']) * float(meta['niSampRate']))
t_start = firstSamp / sRate

tvec = sp.arange(t_start, t_total, 1/sRate)
tvec = sp.arange(0, t_total, 1/sRate)
tvec.shape
digArray.shape
# %% dig into this
fig, axes = plt.subplots()
for i, t in enumerate(times):

    ind = int(inds[i])
    data = digArray[0,ind-1000:ind+1000]
    axes.plot(data)

# %%
bin_path = Path("/media/georg/htcondor/shared-paton/georg/sync_testing_2_g0/sync_testing_2_g0_imec0/sync_testing_2_g0_t0.imec0.ap.bin")
meta = readMeta(bin_path)
t_total = float(meta['fileTimeSecs'])
sRate = float(meta['imSampRate'])
firstSamp = int(meta['firstSample'])
n_samples = int(float(meta['fileTimeSecs']) * float(meta['imSampRate']))
t_start = firstSamp / sRate


# %% analog
# bin_path = Path("/media/georg/htcondor/shared-paton/georg/sync_testing_2_g0/sync_testing_2_g0_imec0/sync_testing_2_g0_t0.imec0.ap.bin")
bin_path = Path("/media/georg/htcondor/shared-paton/georg/sync_testing_5_g0/sync_testing_5_g0_imec0/sync_testing_5_g0_t0.imec0.ap.bin")

def make_mmap(bin_path):
    meta = readMeta(bin_path)
    sRate = SampRate(meta)
    mmap = makeMemMapRaw(bin_path, meta) # returns the entire recording memmapped
    return mmap, meta

def slice_mmap(mmap, meta, t_start, t_stop, chanList=range(384)):
    # old = 30000.101117
    # new = 30000.14484679666
    sRate = float(meta['imSampRate'])
    # sRate = 30000.14484679666

    # t_offset = int(meta['firstSample'])/sRate
    # firstSamp = int(sRate*t_start - t_offset)
    # lastSamp = int(sRate*t_stop - t_offset)

    # no offset
    firstSamp = int(sRate*t_start)
    lastSamp = int(sRate*t_stop)

    mmap_slice = mmap[chanList, firstSamp:lastSamp+1]

    # hardcode: convert to uV
    data = GainCorrectIM(mmap_slice, chanList, meta) * 1e6
    return data.T

# mmap, meta = make_mmap(bin_path)
# data = slice_mmap(mmap, meta, times[0]-0.02, times[0]+0.02)

# %% pre-process
# avg subtract for now
data = data - sp.average(data,0)[sp.newaxis,:]

# %% plot data
def plot_mmap_slice(data,ds=100,ysep=10):
    fig, axes = plt.subplots()
    line_kwargs = dict(color='k',alpha=0.5,lw=0.75)
    tvec = sp.arange(data.shape[0]) / sRate
    for i in range(data.shape[1]):
        axes.plot(tvec[::ds], data[::ds,i] + i * ysep, **line_kwargs)
    return axes

plot_mmap_slice(data, ds=10)

# %% check syncing
mmap, meta = make_mmap(bin_path)
pre, post = -1.0,1.0

# N = 2000
# skip = 100

inds = sp.arange(0,2000,100)

# make colors
import seaborn as sns
colors = sns.color_palette('viridis',n_colors=inds.shape[0])
dt = sp.median(sp.diff(sync_times))

fig, axes = plt.subplots()
for i, ind in enumerate(inds):
    t = trig_times[ind] * m + b
    if t+pre > 0 and t+post < float(meta['fileTimeSecs']):
        data = slice_mmap(mmap, meta, t+pre, t+post)

        sRate = float(meta['imSampRate'])
        tvec = sp.arange(data.shape[0]) / sRate + pre

        # avg subtract
        data = data - sp.average(data,0)[sp.newaxis,:]

        # 
        axes.plot(tvec, sp.average(data,axis=1),color=colors[i])
# %% calculate clock drift
# fig, axes = plt.subplots()
# axes.plot(sync_times)

bin_path = Path("/media/georg/htcondor/shared-paton/georg/sync_testing_5_g0/sync_testing_5_g0_imec0/sync_testing_5_g0_t0.imec0.ap.bin")
mmap, meta = make_mmap(bin_path)

# %%
chunk_size = int(1e6)
n_samples = mmap.shape[1]
n_chunks = sp.floor(n_samples / chunk_size).astype('int32')
from tqdm import tqdm

all_inds = []
all_times = []

sRate = SampRate(meta)

for i in tqdm(range(n_chunks)):
    start = i * chunk_size 
    stop = start + chunk_size
    digArray = ExtractDigital(mmap, start, stop, 0, [6], meta)
    inds, times = get_trigger_times(digArray, meta, 0)
    all_inds.append(inds + start)
    all_times.append(times + start/sRate)

all_inds = sp.concatenate(all_inds)
all_times = sp.concatenate(all_times)
# %%
from scipy import stats
n = all_times.shape[0]
m,b = stats.linregress(all_times,  sync_times[:n])[:2]
m,b = stats.linregress(sync_times[:n], all_times)[:2]


# %%

