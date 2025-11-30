import numpy as np
import argparse
import os
import glob
import datetime
import time
import logging
import h5py
from scipy.io.wavfile import read
import itertools
import sys
sys.path.insert(1, os.path.join(sys.path[0], '../audioset_tagging_cnn/utils'))
import multiprocessing as mp 
from utilities import (create_folder, get_filename, create_logging, pad_or_truncate)
import config


def get_target(audios_dir):
    target = os.path.basename(audios_dir)
    encoding = np.zeros(config.classes_num, dtype=bool)
    for t in target.split("-"):
        idx = config.labels.index(t)
        encoding[idx] = 1
    return encoding


def pack_waveforms_to_hdf5(audios_dir, waveforms_hdf5_path, quiet=True):
    """Pack waveform and target of several audio clips to a single hdf5 file. 
    This can speed up loading and training.
    """

    # Arguments & parameters

    clip_samples = config.clip_samples
    classes_num = config.classes_num
    sample_rate = config.sample_rate
    try:
        create_folder(os.path.dirname(waveforms_hdf5_path))
    except:
        pass

    logs_dir = os.path.join('_logs/pack_waveforms_to_hdf5/', audios_dir)
    create_folder(logs_dir)
    create_logging(logs_dir, filemode='w')
    logging.info('Write logs to {}'.format(logs_dir))
    
    
    # Pack waveform to hdf5
    total_time = time.time()

    audios_num = len(os.listdir(audios_dir))
    
    with h5py.File(waveforms_hdf5_path, 'w') as hf:
        hf.create_dataset('audio_name', shape=((audios_num,)), dtype='S20')
        hf.create_dataset('waveform', shape=((audios_num, clip_samples)), dtype=np.int16)
        hf.create_dataset('target', shape=((audios_num, classes_num)), dtype=np.bool)
        hf.attrs.create('sample_rate', data=sample_rate, dtype=np.int32)

        # Pack waveform & target of several audio clips to a single hdf5 file
        
        for n, name in enumerate(os.listdir(audios_dir)):
            audio_path = os.path.join(audios_dir, name)
            if os.path.isfile(audio_path) and audio_path.lower().endswith('.wav'):
                if not quiet:
                    logging.info('{} {}'.format(name, audio_path))
                try:
                    (_, audio) = read(audio_path)
                except Exception as e:
                    logging.error(f"Error reading {audio_path}: {e}")
                    continue
                audio = pad_or_truncate(audio, clip_samples)

                hf['audio_name'][n] = name.encode()
                hf['waveform'][n] = audio
                hf['target'][n] = get_target(audios_dir)
            else:
                if not quiet:
                    logging.info('{} File does not exist! {}'.format(n, audio_path))

    logging.info('Write to {}'.format(waveforms_hdf5_path))
    logging.info('Pack hdf5 time: {:.3f}'.format(time.time() - total_time))
          

if __name__ == '__main__':

    parser = argparse.ArgumentParser(prog='pack_to_hdf5',description='Pack .wav Files to HDF5 Waveform Dataset For all Labels')
    parser.add_argument('--audios_root')
    parser.add_argument('--output_dir')
    args = parser.parse_args()

    pairs = []

    for r in range(1, config.max_multilabels+1):
        effect_combinations = itertools.combinations(config.labels, r)
        for effect_set in effect_combinations:
            if r != 1 and 'no_effect' in effect_set:
                continue
            label = "-".join(effect_set)
            in_dir = os.path.join(args.audios_root, label)
            if (not os.path.exists(in_dir)):
                continue
            out_path = f"{os.path.join(args.output_dir, label)}.h5"   
            pairs.append((in_dir, out_path))
    
    proc_count = min(mp.cpu_count(), len(pairs))
    procs = [mp.Process(target=pack_waveforms_to_hdf5, args=pair) for pair in pairs]
    
    for p in procs:
        p.start()
    for p in procs:
        p.join()