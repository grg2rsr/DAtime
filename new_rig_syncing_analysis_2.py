# %%
%matplotlib qt5
import scipy as sp
import numpy as np
import matplotlib.pyplot as plt
import matplotlib as mpl
mpl.rcParams['figure.dpi'] = 166

from pathlib import Path
from readSGLX import readMeta, SampRate, makeMemMapRaw, GainCorrectIM, GainCorrectNI, ExtractDigital
from tqdm import tqdm

imec_bin_path = Path("/media/georg/data/yass_testings/data/20210408_GR_JJP01506_octopussy_testing_g4/20210408_GR_JJP01506_octopussy_testing_g4_imec0/20210408_GR_JJP01506_octopussy_testing_g4_t0.imec0.ap.bin")
# imec_bin_path = Path("/media/georg/data/yass_testings/data/20210408_GR_JJP01506_octopussy_testing_g4/20210408_GR_JJP01506_octopussy_testing_g4_imec0/20210408_GR_JJP01506_octopussy_testing_g4_t0.imec0.ap.bin")

imec_bin_path = Path("/home/georg/data/sync_testing_5_g0/sync_testing_5_g0_imec0/sync_testing_5_g0_t0.imec0.ap.bin")


# %% get nidaq_path based on bin_path
def get_ni_bin_path(imec_bin_path):
    f = [fname for fname in os.listdir(imec_bin_path.parent.parent) if fname.endswith('.bin')]
    if len(f) > 1:
        print("multiple bin files found in the folder!")
        print(f)
    return imec_bin_path.parent.parent / f[0]

ni_bin_path = get_ni_bin_path(imec_bin_path)

# %% extract digital from nidaq
def get_digital_data(bin_path, dw=0, dLineList=range(8)):
    """ dw and dLineList hardcoded - unlikely to ever change """

    meta = readMeta(bin_path)
    sRate = SampRate(meta)

    n_samples = float(meta['fileTimeSecs']) * sRate
    if n_samples % 1 != 0.0:
        print("problem: number of samples is not an integer" )
    else:
        n_samples = int(n_samples)
    print("number of samples in ni bin file samples: %i" % n_samples)

    # firstSamp = int(sRate*tStart)
    firstSamp = 0
    lastSamp = int(n_samples-1)

    rawData = makeMemMapRaw(bin_path, meta)

    # get digital data for the selected lines
    digArray = ExtractDigital(rawData, firstSamp, lastSamp, dw, range(8), meta)
    return digArray

# %% get onset flanks
def get_trigger_times(digArray, meta, ch):
    trig_data = digArray[ch,:]

    # get onset_inds
    inds = sp.where(sp.diff(trig_data) == 1)[0]

    sRate = sRate = SampRate(meta)

    # TODO figure out how to deal with first sample - add to time or not? - think it was no
    # t_offset = int(meta['firstSample'])/float(meta['niSampRate'])
    times = inds / sRate

    return inds, times

# %% chunkified version
def extract_onset_events(bin_path, chanList, chunk_size=4000):
    """ important note - the last chunk issue """
    meta = readMeta(bin_path)
    sRate = SampRate(meta)

    n_samples = int(float(meta['fileTimeSecs']) * sRate)
    n_chunks = sp.floor(n_samples / chunk_size).astype('int32')
    print("leftover samples: %i" % (n_samples % n_chunks))

    rawData = makeMemMapRaw(bin_path, meta)

    events = []
    for ch in chanList:
        inds = []

        # get digital data for the selected lines
        for i in tqdm(range(n_chunks)):
            start = i * chunk_size
            stop = start + chunk_size

            digArray = ExtractDigital(rawData, start, stop, 0, range(8), meta)
            trig_data = digArray[ch,:]

            ix = sp.where(sp.diff(trig_data) == 1)[0]
            inds.append(ix+start)
            # if len(ix) > 0:
                # print(len(ix))

        inds = sp.concatenate(inds)
        times = inds / sRate
        events.append([inds,times])

    return events

ni_inds, ni_times = extract_onset_events(ni_bin_path, [0])[0]
imec_inds, imec_times = extract_onset_events(imec_bin_path, [6])[0]

ni_meta = readMeta(ni_bin_path)
imec_meta = readMeta(imec_bin_path)

# %%
ni_digArray = get_digital_data(ni_bin_path)
ni_meta = readMeta(ni_bin_path)
ni_inds, ni_times = get_trigger_times(ni_digArray, ni_meta, 0) # <- sync waveform recorded on dig input 0

# %% same for the sync from the probe
imec_digArray = get_digital_data(imec_bin_path)
imec_meta = readMeta(imec_bin_path)

sync_ch = 6 # why I do not know ... 
imec_inds, imec_times = get_trigger_times(imec_digArray, imec_meta, sync_ch) # <- sync waveform recorded on dig input 0

# %% integrity testing
n_imec = imec_times.shape[0]
n_ni = ni_times.shape[0]

if (n_imec != n_ni):
    print("unequal number of triggers!")
    print("ni: %i" % n_ni)
    print("imec: %i" % n_imec)

# %%
# it is more likely (but not impossible) that the last one is cut on one of them
# mismatch on the first must be very unlucky (trigger falls into init time diff)
# between the to data streams
if (n_imec == n_ni +1):
    imec_times = imec_times[:n_ni]
    imec_inds = imec_inds[:n_ni]


# %%
from scipy.stats import linregress
m,b = linregress(ni_times,imec_times)[:2]

ni_times_corr = ni_times * m + b
print("time sync lin reg")
print(m)
print(b)

# %% VERIFY


# %% mean subtract - to be applied on slices
def global_mean_sub(mmap_slice):
    mu = sp.average(mmap_slice,1)[:,sp.newaxis]
    mmap_slice = mmap_slice - mu
    return mmap_slice

def whitening(mmap_slice, bin_path):
    W = sp.load(bin_path.parent / 'kilosort3' / 'whitening_mat.npy')
    return sp.dot(mmap_slice,W)

def drop_bad_ch(mmap_slice, bin_path):
    good_ch = sp.load(bin_path.parent / 'kilosort3' / 'channel_map.npy')
    return mmap_slice[good_ch,:]

def process_mmap_slice(mmap_slice, chanList, bin_path):
    meta = readMeta(bin_path)
    mmap_slice = drop_bad_ch(mmap_slice, bin_path)
    mmap_slice = GainCorrectIM(mmap_slice, chanList, meta) * 1e6
    mmap_slice = whitening(mmap_slice, bin_path)
    mmap_slice = global_mean_sub(mmap_slice)
    return mmap_slice

def time_slice_mmap(mmap, meta, t_start, t_stop):
    sRate = SampRate(meta)
    a = int(t_start*sRate)
    b = int(t_stop*sRate)
    return mmap[:,a:b]

# %% get stim times
stim_inds, stim_times = extract_onset_events(ni_bin_path, [1])[0]
stim_times_corr = stim_times * m + b

# %% checking the clock
import seaborn as sns

N = 4000
colors = sns.color_palette('viridis',n_colors=int(N/200))
mmap = makeMemMapRaw(imec_bin_path, imec_meta) # returns the entire recording memmapped
pre, post = -0.1,0.1
chanList = range(384)
fig, axes = plt.subplots(nrows=2,sharex=True)

tvec = sp.arange(-0.1,0.1,1/SampRate(imec_meta))[:-1]

for i, t in enumerate(ni_times[10:N:200]):
    mmap_slice = time_slice_mmap(mmap, imec_meta, t+pre, t+post)
    mmap_slice = mmap_slice[:,:tvec.shape[0]]
    axes[0].plot(tvec, mmap_slice[384,:],color=colors[i])

for i, t in enumerate(ni_times_corr[10:N:200]):
    mmap_slice = time_slice_mmap(mmap, imec_meta, t+pre, t+post)
    mmap_slice = mmap_slice[:,:tvec.shape[0]]
    axes[1].plot(tvec, mmap_slice[384,:],color=colors[i])


# %% the electrical artifact
import seaborn as sns

N = 2000
colors = sns.color_palette('viridis',n_colors=int(N/200))
mmap = makeMemMapRaw(imec_bin_path, imec_meta) # returns the entire recording memmapped
pre, post = -0.1,0.1
chanList = range(384)
fig, axes = plt.subplots(nrows=2,sharex=True)

tvec = sp.arange(-0.1,0.1,1/SampRate(imec_meta))[:-1]

for i, t in enumerate(stim_times[10:N:200]):
    mmap_slice = time_slice_mmap(mmap, imec_meta, t+pre, t+post)
    mmap_slice = mmap_slice[:,:tvec.shape[0]]
    axes[0].plot(tvec, sp.average(mmap_slice[chanList,:],axis=0),color=colors[i])

for i, t in enumerate(stim_times_corr[10:N:200]):
    mmap_slice = time_slice_mmap(mmap, imec_meta, t+pre, t+post)
    mmap_slice = mmap_slice[:,:tvec.shape[0]]
    axes[1].plot(tvec, sp.average(mmap_slice[chanList,:],axis=0),color=colors[i])
    
