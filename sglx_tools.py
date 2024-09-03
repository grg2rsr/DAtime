""" Strictly SpikeGLX related tools """
import sys, os
import numpy as np
from pathlib import Path
import readSGLX as glx
from tqdm import tqdm

import logging
logger = logging.getLogger()

# def get_ni_bin_path(imec_bin_path, probe_per_folder=True):
#     """
#     get nidaq_path based on bin_path, if in SGLX "probe per folder option" is activated
#     """
#     run_folder = imec_bin_path.parent.parent # this seems like it's quite hardcoded

#     f = [fname for fname in os.listdir(run_folder) if fname.endswith('.bin')]
#     if len(f) > 1:
#         # this should never be the case if probe per folder is active
#         logger.warning("multiple nidaq.bin files found in the folder!")
#     ni_fname = f[0]
#     logger.info("using nidaq file: %s" % f[0])
#     return run_folder / ni_fname

# def get_run_folders(exp_folder):
#     exp_folder = Path(exp_folder)
#     run_folders = [exp_folder / f for f in os.listdir(exp_folder) if (exp_folder / f).is_dir() and f.split('_')[-1][0] == 'g']
#     return run_folders

# def get_paths(exp_folder, g=0, t=0, imec=0):
#     """

#     """
#     def get_g(run_folder):
#         """ just internal helper """
#         return int(str(run_folder).split('_')[-1][1:])

#     exp_folder = Path(exp_folder)
#     run_folders = get_run_folders(exp_folder)
#     run_folder = [f for f in run_folders if get_g(f) == g][0]
#     run_name = '_'.join(run_folder.name.split('_')[:-1])
#     ni_bin_path = run_folder / (run_name + '_g%i_t%i.nidq.bin' % (g,t))
    
#     if not ni_bin_path.exists():
#         logger.critical("file not found: %s" % ni_bin_path)
    
#     imec_folder = run_folder / (run_name + '_g%i_imec%i' % (g,imec))
#     imec_bin_path = imec_folder / (run_name + '_g%i_t%i.imec%i.ap.bin' % (g,t,imec))
    
#     if not imec_bin_path.exists():
#         logger.critical("file not found: %s" % imec_bin_path)
#     ks_folder = imec_folder / 'pyks2_output' # FIXME hardcoded parameter
    
#     if not ks_folder.exists():
#         logger.critical("folder not found: %s" % ks_folder)

#     return imec_bin_path, ni_bin_path, ks_folder

def get_flanks(digArray, meta, ch, on='rising', offset=0):
    """
    find flanks in the data data, digArray is from glx.ExtractDigital
    """
    
    data = digArray[ch,:]

    # get onset_inds
    if on == 'rising':
        inds = np.where(np.diff(data) == 1)[0] + offset
    if on == 'falling':
        inds = np.where(np.diff(data) == -1)[0] + offset

    sRate = glx.SampRate(meta)

    # TODO figure out how to deal with first sample - add to time or not? - think it was no
    # t_offset = int(meta['firstSample'])/float(meta['niSampRate'])

    times = inds / sRate
    return inds, times

def extract_events(bin_path, on='rising', channels=range(8), chunk_size=8000, save=True):
    """ 
    makes a memmap of the bin file, processes it in chunks
    DOCME
    this function is very slow (more than 1h per dataset)

    STORES:
    A dictionary with keys: channels, values: another dict with keys: inds, times with the indices and the times, respectively
    """

    meta = glx.readMeta(bin_path)
    sRate = glx.SampRate(meta)

    n_samples = int(float(meta['fileTimeSecs']) * sRate)
    n_chunks = np.floor(n_samples / chunk_size).astype('int32')
    n_samples - (n_chunks * chunk_size) # <- this is the size of the last read
    n_leftover = n_samples - (n_chunks * chunk_size)
    logger.debug("leftover samples: %i" % n_leftover)

    rawData = glx.makeMemMapRaw(bin_path, meta)

    Events = {}
    for ch in channels:
        Events[ch] = {}
        Events[ch]['inds'] = []
        Events[ch]['times'] = []

    dw = 0 # the hardcoded digitalWord, ignore it

    def process_chunk(rawData, start_ix, stop_ix, Events):
        digArray = glx.ExtractDigital(rawData, start_ix, stop_ix, dw, range(8), meta)
        for ch in channels:
            inds, times = get_flanks(digArray, meta, ch, on=on, offset=start_ix)
            Events[ch]['inds'].append(inds)#  = (inds, times)
            Events[ch]['times'].append(times)
        return Events

    # loop over chunks
    for i in tqdm(range(n_chunks), desc="event extraction from: %s" % bin_path.name):
        start_ix = i * chunk_size
        stop_ix = start_ix + chunk_size
        process_chunk(rawData, start_ix, stop_ix)

        # digArray = glx.ExtractDigital(rawData, start_ix, stop_ix, dw, range(8), meta)
        # for ch in channels:
        #     inds, times = get_flanks(digArray, meta, ch, on=on, offset=start_ix)
        #     Events[ch]['inds'].append(inds)#  = (inds, times)
        #     Events[ch]['times'].append(times)

    # leftover samples
    start_ix = (n_chunks * chunk_size)
    stop_ix = n_samples-1 #
    process_chunk(rawData, start_ix, stop_ix)

    # digArray = glx.ExtractDigital(rawData, start_ix, stop_ix, dw, range(8), meta)
    # for ch in channels:
    #     inds, times = get_flanks(digArray, meta, ch, on=on, offset=start_ix)
    #     Events[ch]['inds'].append(inds)
    #     Events[ch]['times'].append(times)

    # post process / convert
    for ch in channels:
        Events[ch]['inds'] = np.concatenate(Events[ch]['inds'])
        Events[ch]['times'] = np.concatenate(Events[ch]['times'])

    # store
    if save:
        import pickle
        outpath = bin_path.with_suffix('.events')
        logger.info("storing events from %s at %s" % (bin_path, outpath))
        with open(outpath, 'wb') as fH:
            pickle.dump(Events, fH)

    return Events


def synchronize_data_streams(ni_bin_path, imec_bin_path, save=True):
    """
    synchronizes datastreams of ni and imec, stores slope,intercept
    """
    
    # extract events - store all, but use only hardcoded
    ni_times = extract_events(ni_bin_path)[0]['times']
    imec_times = extract_events(imec_bin_path)[6]['times']

    # integrity testing
    n_imec = imec_times.shape[0]
    n_ni = ni_times.shape[0]

    if (n_imec != n_ni):
        logger.warning("unequal number of sync events!")
        logger.warning("ni: %i" % n_ni)
        logger.warning("imec: %i" % n_imec)

    # it is more likely (but not impossible) that the last one is cut on one of them
    # mismatch on the first must be very unlucky (trigger falls into init time diff)
    # between the to data streams
    # if (n_imec == n_ni +1):
    #     imec_times = imec_times[:n_ni]
    #     imec_inds = imec_inds[:n_ni]

    from scipy.stats import linregress
    m, b = linregress(ni_times, imec_times)[:2]
    pfit = np.array([m,b],dtype='float64')
    if save:
        outpath = ni_bin_path.with_suffix('.tcalib')
        np.save(outpath, pfit)
        logger.info("saved time sync calibration values to %s" % outpath)
    return pfit

def sync(times_a, times_b, mode='linear'):
    """ finds transfer function from a->b """
    if mode == 'linear':
        from scipy.stats import linregress
        m, b = linregress(times_a, times_b)[:2]
        return m, b
    # pfit = np.array([m,b],dtype='float64')