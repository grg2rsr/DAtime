
from pathlib import Path
import readSGLX as glx
from tqdm import tqdm
import pandas as pd
import os
import numpy as np

def load_pyks2_result(ks_folder, imec_bin_path, verbose=True):
    """
    loads all Unit related info that comes out of pyks (all .tsv)
    """

    # read UnitsDf
    tsv_paths = [ks_folder / f for f in os.listdir(ks_folder) if f.endswith('.tsv')]
    UnitsDf = pd.concat([pd.read_csv(path, delimiter='\t') for path in tsv_paths], axis=1)

    # new and specific for kilosort4 - dropping duplicate rows
    UnitsDf = UnitsDf.T.drop_duplicates().T

    # for internal quality metrics
    UnitsDf['good'] = True

    # reading in spike times
    imec_meta = glx.readMeta(imec_bin_path)
    spike_templates = np.load(ks_folder / 'spike_templates.npy')

    # adding number of spikes to UnitsDf
    UnitsDf['n_spikes'] = np.unique(spike_templates,return_counts=True)[1]
    UnitsDf['frate'] = UnitsDf['n_spikes'] / np.float32(imec_meta['fileTimeSecs'])

    # if verbose:
    #     print_summary(UnitsDf)

    return UnitsDf

def print_summary(UnitsDf, f_th=0.05):
    # printing some key output summaries
    n_units = UnitsDf.shape[0]
    for label in UnitsDf['KSLabel'].unique():
        n = np.sum((UnitsDf['KSLabel'] == label).values)
        f = n / n_units * 100
        print("KSLabel = %s\t%i, %.2f%%" % (label, n , f) )

    #  frate related
    # f_th = 0.1
    n = np.sum((UnitsDf['frate'] > f_th).values)
    f = n / n_units * 100
    print("above %.2f Hz: %i, %.2f%%" % (f_th, n, f))

    n = np.sum(np.logical_and((UnitsDf['frate'] > f_th).values, (UnitsDf['KSLabel'] == 'good').values))
    f = n / n_units * 100
    print("intersection of (KSLabel == good) and (fr > %.2f Hz): %i, %.2f%%" % (f_th, n, f))


def load_spikes(ks_folder, imec_bin_path):
    """
    loads spike_times and spike templates from pyks output folder
    converts indices to times [s]
    """
    imec_meta = glx.readMeta(imec_bin_path)
    fs = glx.SampRate(imec_meta) # 1/s
    spike_times = np.load(ks_folder / 'spike_times.npy') / fs
    spike_templates = np.load(ks_folder / 'spike_templates.npy')
    return spike_times, spike_templates


def reformat_spikes(spike_times, spike_templates, unit_ids=None):
    """
    for easier indexing
    """

    Spikes = {}

    if unit_ids is None:
        unit_ids = np.sort(np.unique(spike_templates))

    for unit_id in unit_ids:
        ix = np.where(spike_templates == unit_id)[0]
        Spikes[unit_id] = spike_times[ix]

    return Spikes


# Linking implantation coordinate to allen brain regions
# linking unit location
# this is merging imro tools with pyks tools
