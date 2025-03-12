# %%
# imports
import sys, os
from pathlib import Path
from tqdm import tqdm
import pickle

import numpy as np
import pandas as pd

from brainbox.metrics import single_units as qm

# %%
# path definitions
# maybe it's possible to abbreviate this
exp_folder = Path('/media/georg/htcondor/shared-paton/georg/DAtime/data/batch_24a/2024-06-06_JJP-08672_dh_1-6-1')
run_folder = exp_folder / 'stim_run_1_g0'
imec_bin_path = run_folder / 'stim_run_1_g0_t0.imec0.ap.bin'
ni_bin_path = run_folder / 'stim_run_1_g0_t0.nidq.bin'
ks_folder = run_folder / 'ibl_sorter_results'
results_folder = ks_folder / 'results'

# %%
exp_folder = Path('/media/georg/htcondor/shared-paton/georg/DAtime/data/batch_24a_sorted/2024-06-08_JJP-08628_dh_5-6-1/')
run_folder = exp_folder / 'stim_run_3_g0'
imec_bin_path = run_folder / 'stim_run_3_g0_t0.imec0.ap.bin'
ni_bin_path = run_folder / 'stim_run_3_g0_t0.nidq.bin'
ks_folder = run_folder / 'ibl_sorter_results'
results_folder = ks_folder / 'results'

# %%
spike_times = np.load(ks_folder / 'spike_times.npy')
spike_clusters = np.load(ks_folder / 'spike_clusters.npy')
spike_amps = np.load(ks_folder / 'amplitudes.npy')

# %%
# # calculate spike depth
# # using the absolute amplitude a templates on each channel for a weighted sum

# # the xy coordinates of the channels
# channel_positions = np.load(ks_folder / 'channel_positions.npy')

# # templates = n_templates x n_timesample x n_channels
# templates = np.load(ks_folder / 'templates.npy')

# templates_depths = np.empty(templates.shape[0])
# for i in range(templates.shape[0]):
#     T = templates[i,:,:]
#     W = np.max(np.absolute(T), axis=0) # the weights
#     W = W / np.sum(W)
#     channel_depths = channel_positions[:,1]
#     templates_depths[i] = np.sum(channel_depths * W)

# spike_depths = templates_depths[spike_clusters]

# %%
from ibllib.ephys.ephysqc import phy_model_from_ks2_path
m = phy_model_from_ks2_path(ks_folder, bin_path=imec_bin_path)
cluster_ids = m.spike_clusters
ts = m.spike_times
amps = m.amplitudes
depths = m.depths
r = qm.quick_unit_metrics(cluster_ids, ts, amps, depths)
MetricsDf = pd.DataFrame(r)

# %%
# get UnitsDf and add the metrics
UnitsDf = pd.read_csv(results_folder / 'UnitsDf.csv')
UnitsDf = pd.merge(MetricsDf, UnitsDf, on='cluster_id')

# and store
UnitsDf.to_csv(ks_folder / 'results' / 'UnitsDf.csv', index=None)

# %%
# do something with the metrics
import seaborn as sns
import matplotlib.pyplot as plt
for col in UnitsDf.columns:
    fig, axes = plt.subplots()
    sns.histplot(UnitsDf, x=col)


