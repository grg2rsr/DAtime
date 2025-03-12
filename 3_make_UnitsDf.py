# %% [markdown]
# # create and save a `UnitsDf.csv`

# %%
import sys, os
from pathlib import Path
import numpy as np
import pandas as pd

import readSGLX as glx
import pyks_tools as pkt

import pynapple as nap

# %%
# path definitions
# maybe it's possible to abbreviate this
exp_folder = Path('/media/georg/htcondor/shared-paton/georg/DAtime/data/2023-02-17_JJP-05313-dh_B_1-2-3/')
run_folder = exp_folder / 'stim_run_2_g0'
imec_bin_path = run_folder / 'stim_run_2_g0_t0.imec0.ap.bin'
ni_bin_path = run_folder / 'stim_run_2_g0_t0.nidq.bin'
ks_folder = run_folder / 'pyks2_output'
results_folder = ks_folder / 'results'

# %%
# path definitions
exp_folder = Path('/media/georg/htcondor/shared-paton/georg/DAtime/data/batch_24a/2024-06-06_JJP-08672_dh_1-6-1')
run_folder = exp_folder / 'stim_run_1_g0'
imec_bin_path = run_folder / 'stim_run_1_g0_t0.imec0.ap.bin'
ni_bin_path = run_folder / 'stim_run_1_g0_t0.nidq.bin'
ks_folder = run_folder / 'ibl_sorter_results'
results_folder = ks_folder / 'results'
os.makedirs(results_folder, exist_ok=True)

# %%
exp_folder = Path('/media/georg/htcondor/shared-paton/georg/DAtime/data/batch_24a_sorted/2024-06-08_JJP-08628_dh_5-6-1/')
run_folder = exp_folder / 'stim_run_3_g0'
imec_bin_path = run_folder / 'stim_run_3_g0_t0.imec0.ap.bin'
ni_bin_path = run_folder / 'stim_run_3_g0_t0.nidq.bin'
ks_folder = run_folder / 'ibl_sorter_results'
results_folder = ks_folder / 'results'

# %%
UnitsDf = pkt.load_pyks2_result(ks_folder, imec_bin_path)
UnitsDf

# %%
# store it
# because it depends on the sorting
os.makedirs(results_folder, exist_ok=True)
UnitsDf.to_csv(results_folder / 'UnitsDf.csv', index=None)

# %%
# loading data for pynappling
spike_times, spike_templates = pkt.load_spikes(ks_folder, imec_bin_path)
unit_ids = UnitsDf['cluster_id']

units_dict = pkt.reformat_spikes(spike_times, spike_templates, unit_ids)
units_Ts = {}
for unit_id, spike_times in units_dict.items():
    units_Ts[unit_id] = nap.Ts(t=spike_times)

units = nap.TsGroup(units_Ts)
units.set_info(UnitsDf)

# %%
# store as a pynapple object
units.save(str(results_folder / 'units.npz'))

# %%



