# from PulsePal import PulsePalObject # Import PulsePalObject
# %% imports
import numpy as np
import time
import os
import sys

# set up stims
stims = {'1': [dict(ch=1, n=1, dur=1.0, f=10, v=3.3, t_offset=0, type='custom', train_id=1)],
         '2': [dict(ch=1, n=3, dur=1.0, f=20, v=3.3, t_offset=0, type='custom', train_id=1)]}

# add com to all
com_stim = dict(ch=4, n=1, dur=1.0, f=1,  v=3.3, t_offset=0, type='normal')
for stim_id, stim in stims.items():
    stim.append(com_stim)


# %%
def make_stim_list(stims, N, weights=None):
    """ fully random """
    stim_ids = list(stims.keys())
    random_ids = np.random.choice(stim_ids, size=N, replace=True, p=weights)
    stim_list = [stims[i] for i in random_ids]
    return stim_list, random_ids

def store_list(stim_list, random_ids, outpath):
    """ """
    lines = []
    for stim_id, stim in zip(random_ids, stim_list):
        line = '\t'.join([str(stim_id)] + [repr(ch) for j, ch in enumerate(stim)])
        lines.append(line+'\n')

    with open(outpath,'w') as fH:
        for line in lines:
            fH.write(line)

# %% PulsePal related
from PulsePalDebug import PulsePalObject # Import PulsePalObject
port = '/dev/ttyACM0' 
P = PulsePalObject(port)

def program_pulsepal(P, stim):
    """ stim is iterable of dicts """
    for s in stim:
        if s['type'] == 'custom':
            t = [x*1/s['f'] + s['t_offset'] for x in range(s['n'])]
            v = [s['v']] * int(s['n'])
            P.programOutputChannelParam('customTrainID', s['ch'], s['train_id']) # set channel 1 to use train 1
            P.programOutputChannelParam('restingVoltage', s['ch'], 0.0) # set resting V
            P.programOutputChannelParam('phase1Duration', s['ch'], s['dur']/1000) # sets the length of the pulse
            P.programOutputChannelParam('pulseTrainDuration', s['ch'], t[-1] + s['dur']/1000)
            P.sendCustomPulseTrain(s['train_id'], t, v) # upload

        if s['type'] == 'normal':
            P.programOutputChannelParam('phase1Voltage', s['ch'], s['v'])
            P.programOutputChannelParam('pulseTrainDuration', s['ch'], s['dur']/1000)
            P.programOutputChannelParam('phase1Duration', s['ch'], s['dur']/1000)
            P.syncAllParams() #?

    # time.sleep?

# interface related
from pynput.keyboard import Key, Listener

def on_press(key):
    # slightly less bad
    global stim_id
    global stim

    if hasattr(key,'char'):
        stim_id = key.char
        if stim_id in stims.keys():
            stim = stims[stim_id]
            
            # program_pulsepal(P, K.stim)
            program_pulsepal(P, stim)
        else:
            print("stim with %s not found" % key.char)

    if key == Key.space:
        # trigger the respective channels
        channels = [s['ch'] for s in stim]
        trig = [1 if i+1 in channels else 0 for i in range(4)] 
        P.triggerOutputChannels(*trig)

    if key == Key.esc:
        print('quitting')
        return False

def run_manual():
    # Collect events until released
    with Listener(on_press=on_press) as listener:
        listener.join()
    return None

def run_list(stim_list, ISI=(3000,5000)):
    for i,stim in enumerate(stim_list):
        # upload stim 
        program_pulsepal(P, stim)

        # either sleep here or in function
        time.sleep(0.1)

        # fire resp channels
        trig = [1 if i+1 in channels else 0 for i in range(4)] 
        P.triggerOutputChannels(*trig)
        
        # variable sleep duration
        t_sleep = np.random.randint(*ISI)
        time.sleep(t_sleep/1e3)

        # report something 
        print("%i/%i - %.2f%" % (i, len(stim_list), i/len(stim_list) * 100 ))

run_manual()

# exit gracefully
P.abortPulseTrains()
del P # disconnect


# old code

# interact by awaiting keyboard input
# def run_manual():
#     while True:
#         stim_id = input()
#         if stim_id == 'q':
#             break
#         else:
#             stim_id = int(stim_id)
#             # setDisplay(P, 'stim', str(stim_id))

#         stim = stims[stim_id]
#         program_pulsepal(P, stim)

#         # either sleep here or in functino

#         # fire resp channels
#         trig = [1 if i+1 in channels else 0 for i in range(4)] 
#         P.triggerOutputChannels(*trig)