import numpy as np
import argparse
import os
import datetime
import time
import logging
import h5py
import librosa

import sys
sys.path.insert(1, os.path.join(sys.path[0], '../audioset_tagging_cnn/utils'))

from utilities import create_folder, get_sub_filepaths
import config

import itertools
import multiprocessing as mp

def create_indexes(waveforms_hdf5_path, indexes_hdf5_path):

    with h5py.File(waveforms_hdf5_path, 'r') as hr:
        with h5py.File(indexes_hdf5_path, 'w') as hw:
            audios_num = len(hr['audio_name'])
            hw.create_dataset('audio_name', data=hr['audio_name'][:], dtype='S20')
            hw.create_dataset('target', data=hr['target'][:], dtype=np.bool)
            hw.create_dataset('hdf5_path', data=[waveforms_hdf5_path.encode()] * audios_num, dtype='S200')
            hw.create_dataset('index_in_hdf5', data=np.arange(audios_num), dtype=np.int32)

    print('Write to {}'.format(indexes_hdf5_path))
          

def combine_full_indexes(indexes_hdf5s_dir, full_indexes_hdf5_path):
    """Combine all balanced and unbalanced indexes hdf5s to a single hdf5. This 
    combined indexes hdf5 is used for training with full data (~20k balanced 
    audio clips + ~1.9m unbalanced audio clips).
    """

    classes_num = config.classes_num

    # Paths
    paths = get_sub_filepaths(indexes_hdf5s_dir)
    
    print('Total {} hdf5 to combine.'.format(len(paths)))

    with h5py.File(full_indexes_hdf5_path, 'w') as full_hf:
        full_hf.create_dataset(
            name='audio_name', 
            shape=(0,), 
            maxshape=(None,), 
            dtype='S20')
        
        full_hf.create_dataset(
            name='target', 
            shape=(0, classes_num), 
            maxshape=(None, classes_num), 
            dtype=np.bool)

        full_hf.create_dataset(
            name='hdf5_path', 
            shape=(0,), 
            maxshape=(None,), 
            dtype='S200')

        full_hf.create_dataset(
            name='index_in_hdf5', 
            shape=(0,), 
            maxshape=(None,), 
            dtype=np.int32)

        for path in paths:
            with h5py.File(path, 'r') as part_hf:
                n = len(full_hf['audio_name'][:])
                new_n = n + len(part_hf['audio_name'][:])

                full_hf['audio_name'].resize((new_n,))
                full_hf['audio_name'][n : new_n] = part_hf['audio_name'][:]

                full_hf['target'].resize((new_n, classes_num))
                full_hf['target'][n : new_n] = part_hf['target'][:]

                full_hf['hdf5_path'].resize((new_n,))
                full_hf['hdf5_path'][n : new_n] = part_hf['hdf5_path'][:]

                full_hf['index_in_hdf5'].resize((new_n,))
                full_hf['index_in_hdf5'][n : new_n] = part_hf['index_in_hdf5'][:]
                
    print('Write combined full hdf5 to {}'.format(full_indexes_hdf5_path))


if __name__ == '__main__':
    parser = argparse.ArgumentParser()

    parser.add_argument('--waveforms_hdf5_path', type=str, required=True, help='Path of packed waveforms hdf5.')
    parser.add_argument('--indexes_hdf5_path', type=str, required=True, help='Path to write out indexes hdf5.')

    args = parser.parse_args()
    
    if not os.path.exists(args.indexes_hdf5_path):
        os.mkdir(args.indexes_hdf5_path)

    pairs = []

    for r in range(1, config.max_multilabels+1):
        effect_combinations = itertools.combinations(config.labels, r)
        for effect_set in effect_combinations:
            if r != 1 and 'no_effect' in effect_set:
                continue
            label = "-".join(effect_set)
            in_path = f"{os.path.join(args.waveforms_hdf5_path, label)}.h5"
            
            if (not os.path.exists(in_path)):
                continue
            out_path = f"{os.path.join(args.indexes_hdf5_path, label)}.h5"   
            pairs.append((in_path, out_path))
    
    proc_count = min(mp.cpu_count(), len(pairs))
    procs = [mp.Process(target=create_indexes, args=pair) for pair in pairs]
    
    for p in procs:
        p.start()
    for p in procs:
        p.join()

    combine_full_indexes(args.indexes_hdf5_path, os.path.join(args.indexes_hdf5_path, 'full_indexes.h5'))

    