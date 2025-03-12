# %% [markdown]
# # estimating the firing rates
# of an entire recording

# %%
import os
import numpy as np
import pandas as pd
from pathlib import Path
from readSGLX import readMeta
import pyks_tools as pkt
from tqdm import tqdm
import pynapple as nap

# %%
# path definitions
# maybe it's possible to abbreviate this
exp_folder = Path("/media/georg/htcondor/shared-paton/georg/DAtime/data/2023-02-17_JJP-05313-dh_B_1-2-3/")
run_folder = exp_folder / "stim_run_2_g0"
imec_bin_path = run_folder / "stim_run_2_g0_t0.imec0.ap.bin"
ni_bin_path = run_folder / "stim_run_2_g0_t0.nidq.bin"
ks_folder = run_folder / "pyks2_output"

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
# work directly of pynapple data
units = nap.load_file(str(results_folder / 'units.npz'))

# %%
# firing rate estimation - params setup
from scipy.signal.windows import gaussian

t_start = 0
t_stop = float(readMeta(imec_bin_path)['fileTimeSecs'])
dt_ds = 0.005  # 5 ms resolution
tvec = np.arange(t_start, t_stop, dt_ds)

w_sd = int(0.25 / dt_ds) # 100 ms
w_M = w_sd * 10
w = gaussian(w_M, w_sd) # 5*dt so 25 ms
w[:int(w.shape[0]/2)] = 0 # half gaussian -> making it causal
w = w / w.sum()

# store w for future purposes
np.save(ks_folder / 'results' / 'w.npy', w)


# %%
# firing rate estimation
R = []
for unit_id, Ts in tqdm(units.items()):
    dig = np.digitize(Ts.times(), tvec)
    spikes_ds = np.zeros(tvec.shape[0])
    spikes_ds[dig] = 1
    R.append(np.convolve(spikes_ds, w, mode='same'))

# %%
# and saving
Rates = np.stack(R).T
rates = nap.TsdFrame(t=tvec, d=Rates, columns=units.index)
rates.save( str(results_folder / 'unit_rates.npz') )

# %%



