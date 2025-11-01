touch dataset/hdf5s/waveforms/$1.h5
python audioset_tagging_cnn/utils/dataset.py pack_waveforms_to_hdf5 \
    --audios_dir=dataset/$1 
    --waveforms_hdf5_path=dataset/hdf5s/waveforms/$1.h5