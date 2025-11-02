import os
import load_config
import argparse

parser = argparse.ArgumentParser(prog='index_files',description='Create Indexes for HDF5 Waveform Datasets')
parser.add_argument('--waveforms_dir')
parser.add_argument('--indexes_dir')
args = parser.parse_args()

# temporary solution
load_config.load_config(globals())
query = """python3 audioset_tagging_cnn/utils/create_indexes.py create_indexes 
--waveforms_hdf5_path={}.h5
--indexes_hdf5_path={}.h5"""

for label in labels:
    os.system(query.format(os.path.join(args.waveforms_dir, label),
                           os.path.join(args.indexes_dir, label)))   