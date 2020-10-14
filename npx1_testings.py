# %%
%matplotlib qt5
from pathlib import Path
import scipy as sp
import matplotlib.pyplot as plt
from readSGLX import readMeta, SampRate, makeMemMapRaw, GainCorrectIM, GainCorrectNI, ExtractDigital
import neo
import quantities as pq
from tqdm import tqdm

path = '/media/georg/Elements/2020-06-27_8b_JJP-00876_dh/stim1_g0/stim1_g0_imec0/stim1_g0_t0.imec0.ap.bin'
bin_path = Path(path)

t_start = 100 * pq.s        # in seconds
t_stop = 110 * pq.s

def glx2asig10(bin_path, t_start, t_stop):
    # get metadata
    meta = readMeta(bin_path)

    # get sampling rate
    fs = float(meta['imSampRate']) * pq.Hz

    first_samp = int((t_start*fs).magnitude)
    last_samp = int((t_stop*fs).magnitude)

    # all channels
    chanList = range(385)

    Data_raw_mmap = makeMemMapRaw(bin_path, meta)
    Data_raw_sel = Data_raw_mmap[:,first_samp:last_samp]
    # corrData = 1e6*GainCorrectIM(rawData, chanList, meta)
    # selectData = rawData[:, firstSamp:lastSamp+1]

    Asig = neo.core.AnalogSignal(Data_raw_sel * pq.V, sampling_rate = fs, t_start = t_start)
    return Asig

Asig = glx2asig10(bin_path, t_start, t_stop)

# %% extract digial

# path = "/media/georg/Elements/2020-06-29_10a_JJP-00870_dh/stim2_g0/stim2_g0_t0.nidq.bin"
path = "/media/georg/Elements/2020-06-27_8b_JJP-00876_dh/stim1_g0/stim1_g0_t0.nidq.bin"


bin_path = Path(path)
def get_TTL_onsets10(bin_path, channel_id, chunk_size=6000):

    # get metadata 
    meta = readMeta(bin_path)

    # get sampling rate
    fs = float(meta['niSampRate']) * pq.Hz

    dw = 0
    dLineList = [0] # the channel 

    Data_raw_mmap = makeMemMapRaw(bin_path, meta)

    nSamples = float(meta['fileTimeSecs']) * fs

    onset_inds = []

    for i in tqdm(range(int(nSamples/chunk_size))):
        start = i * chunk_size
        stop = start + chunk_size 

        chunk = ExtractDigital(Data_raw_mmap, start, stop, dw, dLineList, meta)
        trig_ch = chunk.flatten()

        inds = sp.where(sp.diff(trig_ch) == 1)[0]
        if len(inds) > 0:
            onset_inds.append(inds+start)
    onset_inds = sp.array(onset_inds).flatten()

    print(" - " + str(len(onset_inds)) + " events detected")

    onset_times = (onset_inds / fs).rescale('s')
    return onset_times

