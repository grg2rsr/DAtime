# Initializing PulsePal
from PulsePal import PulsePalObject # Import PulsePalObject
port = '/dev/ttyACM0' 
# P = PulsePalObject(ppport) # Create a new instance of a PulsePal object
# print(P.firmwareVersion) # Print firmware version to the console

# ini
P = PulsePalObject(port)


# precision testing


# pulse_dur = 0.1 # this is seconds?
# ipi = 0.2

# pulseTimes = [0, 0.2, 0.5, 1] # Create an array of pulse times in seconds
# voltages = [8,4,-3.5,-10] # Create an array of pulse voltages in volts
# myPulsePal.programOutputChannelParam('customTrainID', 1, 2) # Set output ch1 to use custom train 2 
# myPulsePal.sendCustomPulseTrain(2, pulseTimes, voltages) # Send arrays to PulsePal, defined as custom train 2
# myPulsePal.triggerOutputChannels(1, 0, 0, 0) # Soft-trigger output channel 1 to play its pulse train

# P.programOutputChannelParam('phase1Duration', 1, 0.01)
# P.programOutputChannelParam('phase1Voltage', 4, 5)
# P.programOutputChannelParam('phase1Duration', 4, 0.01)
# P.programOutputChannelParam('interPulseInterval', 1, dur*2)
# P.programOutputChannelParam('interPulseInterval', 4, dur*2)
# P.syncAllParams()


# custom pulse train generation
# v = [stim.v]*int(stim.n)
# t = range(int(stim.n))*1/stim.f
# t = [T + 0.25 for T in t] # HARDCODED DA stim offset here
import numpy as np
t = list(np.arange(0,10,0.010))
V = [3.3]*len(t)


# both channels for now
P.programOutputChannelParam('customTrainID', 1, 1) # set channel 1 to train 1
P.programOutputChannelParam('restingVoltage',1, 0.0) # set resting V
P.sendCustomPulseTrain(1, t, V)
P.programOutputChannelParam('phase1Duration', 1, 0.002) # FIXME this line is unclear to me
P.triggerOutputChannels(1, 0, 0, 0) # software trigger
import time
time.sleep(10)

