
"""
 
    __       _              _       _        
   / _| __ _| | _____    __| | __ _| |_ __ _ 
  | |_ / _` | |/ / _ \  / _` |/ _` | __/ _` |
  |  _| (_| |   <  __/ | (_| | (_| | || (_| |
  |_|  \__,_|_|\_\___|  \__,_|\__,_|\__\__,_|
                                             
 
"""
# %%
dt = 0.01
tt = sp.arange(-1,3,dt)

nUnits = 50
nTrials = 50
nTrials_opto = 30

Rates = sp.zeros((nUnits,nTrials),dtype='object')
SpikeTrains = sp.zeros((nUnits,nTrials),dtype='object')

Rates_opto = sp.zeros((nUnits,nTrials),dtype='object')
SpikeTrains_opto = sp.zeros((nUnits,nTrials),dtype='object')


# generating spike trains
fr_opts = dict(sampling_period=dt*pq.s, kernel=ele.kernels.GaussianKernel(sigma=50*pq.ms))

for i in tqdm(range(nUnits)):
    # peak_time = tt[sp.random.randint(tt.shape[0])]
    peak_time = sp.rand() * tt[-1]
    peak_time_opto = peak_time * 1.2

    rate_gen = sp.stats.distributions.norm(peak_time,0.5).pdf(tt) * 10
    asig = neo.core.AnalogSignal(rate_gen,t_start=tt[0]*pq.s,units=pq.Hz,sampling_period=dt*pq.s)
    for j in range(nTrials):
        st = ele.spike_train_generation.inhomogeneous_poisson_process(asig)
        SpikeTrains[i,j] = st

        r = ele.statistics.instantaneous_rate(st,**fr_opts)
        rz = ele.signal_processing.zscore(r)
        Rates[i,j] = rz

    # opto
    rate_gen = sp.stats.distributions.norm(peak_time_opto,0.5).pdf(tt) * 10
    asig = neo.core.AnalogSignal(rate_gen,t_start=tt[0]*pq.s,units=pq.Hz,sampling_period=dt*pq.s)
    for j in range(nTrials):
        st = ele.spike_train_generation.inhomogeneous_poisson_process(asig)
        SpikeTrains_opto[i,j] = st

        r = ele.statistics.instantaneous_rate(st,**fr_opts)
        rz = ele.signal_processing.zscore(r)
        Rates_opto[i,j] = rz


Rates_ = copy(Rates)
Rates_opto_ = copy(Rates_opto)
