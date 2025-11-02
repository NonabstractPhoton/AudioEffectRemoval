import os
import load_config

# temporary solution
load_config.load_config(globals())
query = """python3 audioset_tagging_cnn/utils/create_indexes.py create_indexes 
    --waveforms_hdf5_path=dataset/hdf5s/waveforms/{}.h5"
    --indexes_hdf5_path=dataset/hdf5s/indexes/{}.h5"""

for label in labels:
    os.system(query.format(label, label))   