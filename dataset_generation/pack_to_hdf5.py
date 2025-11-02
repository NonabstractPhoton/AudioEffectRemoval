import os
import load_config
import argparse

parser = argparse.ArgumentParser(prog='pack_to_hdf5',description='Pack .wav Files to HDF5 Waveform Dataset For all Labels')
parser.add_argument('--audios_root')
parser.add_argument('--output_dir')
args = parser.parse_args()

# temporary solution
load_config.load_config(globals())
query = "python audioset_tagging_cnn/utils/dataset.py pack_waveforms_to_hdf5 \
    --audios_dir={} \
    --waveforms_hdf5_path={}.h5"
for label in labels:
    os.system(query.format(os.path.join(args.audios_dir, label),
                           os.path.join(args.output_dir, label)))   