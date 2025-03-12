# %%
# imports
import sys, os
import numpy as np
from pathlib import Path
# from readSGLX import readMeta, SampRate, makeMemMapRaw, GainCorrectIM, GainCorrectNI, ExtractDigital
from tqdm import tqdm
import pandas as pd
import pickle
# import pyks_tools as pkt
import sglx_tools as glt


# %%
# path definitions
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
# if data hasn't been synced, do so now
path = ni_bin_path.with_suffix('.tcalib.npy')
if path.exists():
    print("loading clock calib values")
    mb = np.load(path)
else:
    print("no clock calib values found - computing them")
    mb = glt.synchronize_data_streams(ni_bin_path, imec_bin_path)

# %%
# same with events
events_path = ni_bin_path.with_suffix('.events')
if events_path.exists():
   
    print("loading Events")
    with open(events_path, 'rb') as fH:
        Events = pickle.load(fH)
else:
    print("extracting events ")
    Events = glt.extract_events(ni_bin_path, save=True)

# %% [markdown]
# applying synchronization

# %%
# correct all event times
for key in Events.keys():
    Events[key]['times_corr'] = Events[key]['times'] * mb[0] + mb[1]

# and store again
with open(events_path, 'wb') as fH:
    pickle.dump(Events, fH)


