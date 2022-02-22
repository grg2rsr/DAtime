from PulsePal import PulsePalObject # Import PulsePalObject
import numpy as np
import time
import os
import sys

"""
probing stimulus protocol - desired functionality:
on button push
run stim once

full measurement - desired functionality:
start, halt, resume, reset - type of thing
"""

# FIXME - this one currently doesn't work, try patching the object
def setDisplay(P, row1String, row2String):
    messageBytes = row1String + chr(254) + row2String
    messageSize = len(messageBytes)
    messageBytes = chr(P.OpMenuByte) + chr(78) + chr(messageSize) + messageBytes
    P.Port.write(messageBytes)

def make_stim(t=0, v=3.3, n=1, dur=1.0, f=10):
    t = [x*1/f for x in range(n)]
    v = [v]*int(n)
    return t, v

def t_from_f(n=1, f=10, t_offset=0):
    dt = 1/f
    t = [x*dt + t_offset for x in range(n)]
    return t

def set_stim(stim):

    v = [v]*int(stim[n])

def send_stim(t, v, P, train_id=1, ch=1, dur=1.0):
    P.programOutputChannelParam('customTrainID', ch, train_id) # set channel 1 to use train 1
    P.programOutputChannelParam('restingVoltage', ch, 0.0) # set resting V
    P.programOutputChannelParam('phase1Duration', ch, dur/1000) # sets the length of the pulse
    P.programOutputChannelParam('pulseTrainDuration', ch, t[-1] + dur/1000)
    P.sendCustomPulseTrain(train_id, t, v) # upload
   
# set up stims
stims = {1: dict(n=1, dur=1.0, f=10, ch=1),
         2: dict(n=3, dur=1.0, f=10, ch=1),
         3: dict(n=5, dur=1.0, f=10, ch=1),
         4: dict(n=1, dur=5.0, f=50, ch=1),
         5: dict(n=3, dur=5.0, f=50, ch=1),
         6: dict(n=5, dur=5.0, f=50, ch=1),
         7: dict(n=1, dur=1.0, f=50, ch=1),
         8: dict(n=3, dur=1.0, f=50, ch=1),
         9: dict(n=5, dur=1.0, f=50, ch=1)}

## Common stuff
# ini
from PulsePal import PulsePalObject # Import PulsePalObject
port = '/dev/ttyACM0' 
P = PulsePalObject(port)

# define com stim on 4
P.programOutputChannelParam('phase1Voltage', 4, 5)
P.programOutputChannelParam('pulseTrainDuration', 4, 0.001)
P.programOutputChannelParam('phase1Duration', 4, 0.001)
P.syncAllParams()


# interact by awaiting keyboard input
def run():
    while True:
        stim_id = input()
        if stim_id == 'q':
            break
        else:
            stim_id = int(stim_id)
            setDisplay(P, 'stim', str(stim_id))

        stim = stims[stim_id]
        t, v = make_stim(**stim)
        
        send_stim(t, v, P, ch=1, dur=stim['dur'])

        # fire resp channel
        P.triggerOutputChannels(1, 0, 0, 1)

run()

# exit gracefully
P.abortPulseTrains()
del P # disconnect
