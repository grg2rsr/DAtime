import os
import numpy as np
import pandas as pd

import logging
logger = logging.getLogger()

def get_stim_filepath(run_folder):
    folder = [f for f in os.listdir(run_folder) if f.endswith('_stims')][0]
    logger.info("getting stim file from %s" % folder)
    return run_folder / folder / "stim_list.txt"
    
def get_StimsDf(run_folder, Events):
    # parse stim_file
    stim_filepath = get_stim_filepath(run_folder)
    with open(stim_filepath, 'r')as fH:
        stims_file = fH.readlines()

    # make stim def
    stims_unique = np.unique(stims_file)
    stim_classes = {}
    for i in range(stims_unique.shape[0]):
        s = stims_unique[i].strip().split('\t')
        stim_classes[s[0]] = [dict(eval(s[j+1])) for j in range(len(s)-1)]

    # the stim IDs as they were presented to the animal
    stims = []
    for line in stims_file:
        stims.append(line.split('\t')[0])
    stims = np.array(stims)

    # create StimsDf
    StimsDf = pd.DataFrame([])

    # from events
    StimsDf['t'] = Events[1]['times_corr']
    StimsDf['t_orig'] = Events[1]['times']
    StimsDf['stim_id'] = stims

    # some hardcoded stuff
    StimsDf['VPL'] = False
    StimsDf['SNc'] = False

    StimsDf.loc[StimsDf['stim_id'] == '1','VPL'] = True
    StimsDf.loc[StimsDf['stim_id'] == '2','VPL'] = True

    StimsDf.loc[StimsDf['stim_id'] == '2','SNc'] = True
    StimsDf.loc[StimsDf['stim_id'] == '3','SNc'] = True

    StimsDf['both'] = StimsDf['VPL'] * StimsDf['SNc']
    StimsDf['stim_id'] = StimsDf['stim_id'].astype(str)

    return StimsDf, stim_classes

def reslice_timestamps(T, slice_times, pre, post):
    T_slices = []
    for t in slice_times:
        ix = np.logical_and(T > t+pre, T < t+post)
        T_slices.append(T[ix] - t) # relative times
    return T_slices

def infer_StimsDf(Events):
    """ fairly chaotic function, pieced together from various leftover code blocks
     decided to not touch it for now, clunky but works """

    # hardcoding map
    trigger_map = dict(sync=0, trig=1, da=2, vpl=3)

    vpl_list = reslice_timestamps(Events[trigger_map['vpl']]['times'], Events[trigger_map['trig']]['times'], 0, 3)
    da_list = reslice_timestamps(Events[trigger_map['da']]['times'], Events[trigger_map['trig']]['times'], 0, 3)

    StimsDf = pd.DataFrame([])

    StimsDf['t'] = Events[trigger_map['trig']]['times_corr'] # the calibrated and corrected stimulus times
    StimsDf['VPL'] = [True if l.shape[0] > 0 else False for l in vpl_list]
    StimsDf['SNc'] = [True if l.shape[0] > 0 else False for l in da_list]
    StimsDf['both'] = StimsDf['VPL'] * StimsDf['SNc']
    StimsDf['stim_id'] = '0'

    StimsDf.loc[StimsDf['VPL']  & ~StimsDf['SNc'], 'stim_id'] = '1'
    StimsDf.loc[StimsDf['VPL']  & StimsDf['SNc'], 'stim_id'] = '2'
    StimsDf.loc[~StimsDf['VPL'] & StimsDf['SNc'], 'stim_id'] = '3'

    return StimsDf

def get_stim_dur_offset(stim_dict):
    # stim_dict is dict form of stim as from stim_classes[label]
    # has hardcode that red = vpl stim is on channel 1

    # stim_id = '4'
    # for ch in Stims[stim_id]:
    #     if ch['ch'] == 1:
    #         stim_dur = (ch['n'] / ch['f']) + ch['dur']/1e3
    for ch in stim_dict:
        if ch['ch'] == 1:
            stim_dur = (ch['n'] / ch['f']) + ch['dur']/1e3
            t_offset = ch['t_offset']/1e3
            return stim_dur, t_offset

# # load data
# def get_StimsDf(run_folder, Events):
#     folder = [f for f in os.listdir(run_folder) if f.endswith('_stims')][0]
#     print("getting stim file from %s" % folder)
#     stim_file_path = run_folder / folder / "stim_list.txt"
#     with open(stim_file_path, 'r')as fH:
#         stims_file = fH.readlines()

#     # make stim def
#     stims_unique = np.unique(stims_file)
#     stim_classes = {}
#     for i in range(stims_unique.shape[0]):
#         s = stims_unique[i].strip().split('\t')
#         stim_classes[s[0]] = [dict(eval(s[j+1])) for j in range(len(s)-1)]

#     # the stim IDs as they were presented to the animal
#     stims = []
#     for line in stims_file:
#         stims.append(line.split('\t')[0])
#     stims = np.array(stims)

#     # from events
#     stim_times = Events[1]['times_corr']
#     n_stims = stim_times.shape[0]
#     stims = stims[:n_stims] # FIXME will be obsolete in the future

#     stim_labels = np.sort(np.unique(stims))
#     # n_stim_classes = stim_labels.shape[0]

#     StimsDf = pd.DataFrame([])
#     StimsDf['t'] = stim_times
#     StimsDf['stim_id'] = stims
    
#     # some hardcoded stuff
#     StimsDf['VPL'] = False
#     StimsDf['SNc'] = False

#     StimsDf.loc[StimsDf['stim_id'] == '1','VPL'] = True
#     StimsDf.loc[StimsDf['stim_id'] == '2','VPL'] = True

#     StimsDf.loc[StimsDf['stim_id'] == '2','SNc'] = True
#     StimsDf.loc[StimsDf['stim_id'] == '3','SNc'] = True

#     StimsDf['both'] = StimsDf['VPL'] * StimsDf['SNc']
    
#     return StimsDf, stim_classes

# def infer_StimsDf(Events):
#     from helpers import reslice_timestamps

#     # hardcoding map
#     sync = 0
#     trig = 1
#     da = 2
#     vpl = 3
#     vpl_list = reslice_timestamps(Events[vpl]['times'], Events[trig]['times'],0,3)
#     da_list = reslice_timestamps(Events[da]['times'], Events[trig]['times'],0,3)

#     StimsDf = pd.DataFrame([])

#     StimsDf['t'] = Events[trig]['times_corr']
#     StimsDf['VPL'] = [True if l.shape[0] > 0 else False for l in vpl_list]
#     StimsDf['SNc'] = [True if l.shape[0] > 0 else False for l in da_list]
#     StimsDf['both'] = StimsDf['VPL'] * StimsDf['SNc']
#     StimsDf['stim_id'] = '0'

#     StimsDf.loc[StimsDf['VPL']  & ~StimsDf['SNc'], 'stim_id'] = '1'
#     StimsDf.loc[StimsDf['VPL']  & StimsDf['SNc'], 'stim_id'] = '2'
#     StimsDf.loc[~StimsDf['VPL'] & StimsDf['SNc'], 'stim_id'] = '3'

#     return StimsDf

# # def get_stim_classes(run_folder):
# #     folder = [f for f in os.listdir(run_folder) if f.endswith('_stims')][0]
# #     print("getting stim file from %s" % folder)
# #     stim_file_path = run_folder / folder / "stim_list.txt"
# #     with open(stim_file_path, 'r')as fH:
# #         stims_file = fH.readlines()

# #     # make stim def
# #     stims_unique = np.unique(stims_file)
# #     stim_classes = {}
# #     for i in range(stims_unique.shape[0]):
# #         s = stims_unique[i].strip().split('\t')
# #         stim_classes[s[0]] = [dict(eval(s[j+1])) for j in range(len(s)-1)]
# #     return stim_classes


