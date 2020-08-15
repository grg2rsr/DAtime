# code orga

make a UnitInfo and TrialInfo DataFrame

UnitInfo can hold a path to where the data will be found

# idea on how to extent this to the future without dying of data overload
construct segment from UnitInfo. UnitInfo can be groupbyed, and holds path 
to the data, UnitInfo will also hold info about animal etc

UnitInfo
animalid, unitid, ks2 stuff, depth, area

# relative to bin file
folder: extraced_data
subfolders: unit_{id}
subfolders:

spiketrain.neo
fratez.neo

subfolder: trial_sliced
subfolders 1,2,3,4,5,6 ... 
with each
spiketrain.neo
fratesz.neo

# then, gathering


# new approach
bin2neo
makes 1 large master neo file, and puts out UnitsInfo and TrialsInfo
neo file has all spiketrains and epochs blabla attached
think about which parts here are specific to this experiment to isolate them
