# %% [markdown]
# # stimulus preprocessing
# purpose: create a `StimsDf.csv` from the SpikeGLX folder

# %%
from pathlib import Path
import pickle
import pandas as pd
import numpy as np
from stim_tools import get_StimsDf, infer_StimsDf

# %%
# path definitions
# maybe it's possible to abbreviate this
exp_folder = Path("/media/georg/htcondor/shared-paton/georg/DAtime/data/230119_JJP-05248-dh_3-3-2/")
# exp_folder = Path("/media/georg/htcondor/shared-paton/georg/DAtime/data/2023-02-17_JJP-05313-dh_B_1-2-3/")
run_folder = exp_folder / "stim_run_2_g0/"
imec_bin_path = run_folder / "stim_run_2_g0_t0.imec0.ap.bin"
ni_bin_path = run_folder / "stim_run_2_g0_t0.nidq.bin"
ks_folder = run_folder / "pyks2_output"

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

# %%
exp_folder = Path('/media/georg/htcondor/shared-paton/georg/DAtime/data/batch_24a_sorted/2024-06-08_JJP-08628_dh_5-6-1/')
run_folder = exp_folder / 'stim_run_3_g0'
imec_bin_path = run_folder / 'stim_run_3_g0_t0.imec0.ap.bin'
ni_bin_path = run_folder / 'stim_run_3_g0_t0.nidq.bin'
ks_folder = run_folder / 'ibl_sorter_results'
results_folder = ks_folder / 'results'

# %%
# get events
events_path = ni_bin_path.with_suffix('.events')
print("loading Events from %s" % events_path)
with open(events_path, 'rb') as fH:
    Events = pickle.load(fH)

from stim_tools import get_StimsDf

StimsDf, stim_classes = get_StimsDf(run_folder, Events)

# store
StimsDf.to_csv(run_folder / "StimsDf.csv", index=None)
path = run_folder / 'stim_classes.pkl'
with open(path, 'wb') as fH:
    pickle.dump(stim_classes, fH)

# %%



