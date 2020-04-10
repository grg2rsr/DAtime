import neo
import scipy as sp
"""
 
                    
   _ __   ___  ___  
  | '_ \ / _ \/ _ \ 
  | | | |  __/ (_) |
  |_| |_|\___|\___/ 
                    
 
"""

def average_asigs(asigs):
    S = sp.stack([asig.magnitude.flatten() for asig in asigs],axis=1)
    asig = neo.core.AnalogSignal(sp.average(S,axis=1),units=asigs[0].units,t_start=asigs[0].t_start,sampling_rate=asigs[0].sampling_rate)
    return asig

def select(neo_objs,value,key="label"):
    return [obj for obj in neo_objs if obj.annotations[key] == value]


"""
 
         _       _   _   _             
   _ __ | | ___ | |_| |_(_)_ __   __ _ 
  | '_ \| |/ _ \| __| __| | '_ \ / _` |
  | |_) | | (_) | |_| |_| | | | | (_| |
  | .__/|_|\___/ \__|\__|_|_| |_|\__, |
  |_|                            |___/ 
 
"""
def add_stim(axes,epoch,DA=True,axis='x'):
    """ epoch is vpl stim"""
    kwargs = dict(alpha=0.25, linewidth=0)
    if not DA:
        for i in range(epoch.times.shape[0]):
            t = epoch.times[i]
            dur = epoch.durations[i]
            if 'x' in axis:
                axes.axvspan(t,t+dur, color='firebrick', **kwargs)
            if 'y' in axis:
                axes.axhspan(t,t+dur, color='firebrick', **kwargs)
    if DA: # FIXME Hardcode
        axes.axvspan(0,2,1,1.1, color='darkcyan', clip_on=False, **kwargs)
    
    return axes

def add_epoch(axes, epoch, above=False, axis='x', alpha=0.25, linewidth=0, color='firebrick'):
    if above:
        hmin, hmax = 1.01, 1.05
        clip_on = False
    else:
        hmin, hmax = 0, 1
        clip_on = True

    kwargs = dict(clip_on=clip_on, alpha=alpha, color=color, linewidth=linewidth)
    for i in range(epoch.times.shape[0]):
        t = epoch.times[i]
        dur = epoch.durations[i]
        if 'x' in axis:
            axes.axvspan(t,t+dur, hmin, hmax, **kwargs)
        if 'y' in axis:
            axes.axhspan(t,t+dur, hmin, hmax, **kwargs)
