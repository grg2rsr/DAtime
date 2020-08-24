# %matplotlib qt5

import matplotlib.pyplot as plt
import matplotlib as mpl
# import seaborn as sns

# plt.style.use('default')
# mpl.rcParams['figure.dpi']=166

import sys,os
from tqdm import tqdm

import scipy as sp
import pandas as pd
import neo
import elephant as ele
import spikeglx as glx
import quantities as pq
from pathlib import Path

"""
 _______  __    __  .__   __.   ______     _______.
|   ____||  |  |  | |  \ |  |  /      |   /       |
|  |__   |  |  |  | |   \|  | |  ,----'  |   (----`
|   __|  |  |  |  | |  . `  | |  |        \   \
|  |     |  `--'  | |  |\   | |  `----.----)   |
|__|      \______/  |__| \__|  \______|_______/

"""
def read_phy(phy_output_folder):
    """ reads the output of phy """
    keys = ['cluster_KSLabel','cluster_info','cluster_group',
            'cluster_ContamPct','cluster_Amplitude']
    phy = {}
    for key in keys:
        try:
            fname = phy_output_folder.joinpath(key).with_suffix('.tsv')
            phy[key] = pd.read_csv(fname,delimiter='\t')
        except:
            pass
    return phy

def read_kilosort2(kilosort_folder):
    """ reads the kilsosort things and returns a dict"""
    keys = ['whitening_mat_inv','whitening_mat','templates_ind','templates',
            'template_features','template_feature_ind','spike_times',
            'spike_templates','spike_clusters','similar_templates',
            'pc_features','pc_feature_ind','channel_map','amplitudes']
    ks2 = {}
    for key in keys:
        fname = kilosort_folder.joinpath(key).with_suffix('.npy')
        ks2[key] = sp.load(fname)

    return ks2



def glx2asig(bin_path,t_start,t_stop):
    """ reads a .bin file created by spikeglx into a neo AnalogSignal """
    ### glx read
    Reader = glx.Reader(bin_path)
    
    fs = Reader.meta['imSampRate']
    s_read = [sp.int32(t * fs) for t in (t_start.magnitude,t_stop.magnitude)]

    Data_raw, sync = Reader.read_samples(*s_read)
    Data_raw = Data_raw[:,:-1] # strips the analog channel

    # to neo
    Asig = neo.core.AnalogSignal(Data_raw * pq.V, sampling_rate = fs*pq.Hz, t_start=t_start)

    return Asig

def global_mean(Asig,ks2):
    """ global mean subtraction """
    # global median data correction
    data_corr = Asig.magnitude - sp.median(Asig.magnitude,axis=0)[sp.newaxis,:]
    Asig = neo.core.AnalogSignal(data_corr*Asig.units, t_start=Asig.t_start, sampling_rate=Asig.sampling_rate,**Asig.annotations)

    return Asig

def preprocess_ks2based(Asig,ks2):
    """ kilosort2 output aided preprocessing steps: whitening and dropping channels """
    # drop bad channels (as identified by kilosort2)
    valid_channels = ks2['channel_map'].flatten()
    Asig = Asig[:,valid_channels]
    
    # whitening
    data_corr = sp.dot(Asig.magnitude,ks2['whitening_mat'])
    Asig = neo.core.AnalogSignal(data_corr*Asig.units, t_start=Asig.t_start, sampling_rate=Asig.sampling_rate,**Asig.annotations)

    return Asig

def read_spiketrains(ks2, fs, t_stop):
    """ reads kilosort based output into a list of neo SpikeTrains
    does not require the presence of bin_file
    """

    Templates = ks2['spike_clusters']
    nUnits = sp.unique(ks2['spike_clusters']).shape[0]
    spikes = ks2['spike_times']

    Sts = []
    for i in tqdm(range(nUnits),desc='reading all spiketimes'):
        St = neo.core.SpikeTrain((spikes[sp.where(Templates == i)] / fs).rescale(pq.s), t_stop=t_stop)
        Sts.append(St)
    return Sts

def read_spiketrains_2(ks2, fs, t_stop):
    """ reads kilosort based output into a list of neo SpikeTrains
    does not require the presence of bin_file
    """

    Templates = ks2['spike_clusters']
    templates_unique = sp.unique(Templates)
    nUnits = templates_unique.shape[0]
    spike_times = (ks2['spike_times'] / fs).rescale(pq.s)

    Sts = []
    for i, template_id in enumerate(tqdm(templates_unique)):
        times = spike_times[sp.where(Templates == template_id)[0]]
        St = neo.core.SpikeTrain(times, t_stop=t_stop)
        Sts.append(St)
        
    return Sts

def get_TTL_onsets(bin_path, channel_id, chunk_size=60000):
    """ extracts trigger TTL onset times from a .glx file  """

    print(" - processing TTL event detection in file: ",bin_path)
    print(" - channel: ", channel_id)

    # reading data
    # os.chdir(os.path.dirname(bin_path))
    R = glx.Reader(bin_path)

    fs = R.meta['imSampRate']
    nSamples = R.meta['fileTimeSecs']* fs

    onset_inds = []

    for i in tqdm(range(int(nSamples/chunk_size))):
        start = i * chunk_size
        stop = start + chunk_size 
        s = R.read_sync(slice=slice(int(start),int(stop)))

        trig_ch = s[:,channel_id]

        inds = sp.where(sp.diff(trig_ch) == 1)[0]
        if len(inds) > 0:
            onset_inds.append(inds+start)
    onset_inds = sp.array(onset_inds).flatten()

    print(" - " + str(len(onset_inds)) + " events detected")

    onset_times = onset_inds / fs
    return onset_times
    
def read_stim_file(path):
    stims = pd.read_csv(path ,delimiter=',')
    stims['dur'] = stims['dur'] * 1000
    stims[['n','rep','dur']] = stims[['n','rep','dur']].astype('int32')
    stims[['volt','f']] = sp.around(stims[['volt','f']].values,2)
    return stims


"""
.______    __        ______   .___________.___________. __  .__   __.   _______
|   _  \  |  |      /  __  \  |           |           ||  | |  \ |  |  /  _____|
|  |_)  | |  |     |  |  |  | `---|  |----`---|  |----`|  | |   \|  | |  |  __
|   ___/  |  |     |  |  |  |     |  |        |  |     |  | |  . `  | |  | |_ |
|  |      |  `----.|  `--'  |     |  |        |  |     |  | |  |\   | |  |__| |
| _|      |_______| \______/      |__|        |__|     |__| |__| \__|  \______|

"""

def label_stim_artifact(axes,k,StimsDf):
    dur,n,f = StimsDf.groupby('id').get_group(k).iloc[0][['dur','n','f']]
    for j in range(int(n)):
        # start stop of each pulse in s
        start = j * 1/f
        stop = start + dur/1000
        axes.axvspan(start,stop,color='firebrick',alpha=0.5,linewidth=0)

def plot_npx_asig(Asig,ds=10,ysep=0.0001,ax=None, kwargs=None):
    """ plots an analog signal """
    if ax is None:
        fig, ax = plt.subplots()
    else:
        fig = plt.gcf()
    
    if kwargs is None:
        kwargs = dict(lw=0.25, alpha=0.5,color='k')

    ysep = ysep * Asig.units
    nChannels = Asig.shape[1]
    for i in range(nChannels):
        ax.plot(Asig.times[::ds],Asig[::ds,i]+i*ysep,**kwargs)

    return fig, ax