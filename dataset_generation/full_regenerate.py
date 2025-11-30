import os
import sys
sys.path.insert(1, os.path.join(sys.path[0], '../audioset_tagging_cnn/utils'))
import config
import argparse

if __name__ == "__main__":
    parser = argparse.ArgumentParser(prog='full_regenerate',description='Regenerate Full Dataset with Effects Applied')
    parser.add_argument('audios_root', help='Root directory where audio files are contained. The clean files should all be under a subdirectory named no_effect')
    parser.add_argument("--selective_enable", default=7,type=int)
    args = parser.parse_args()
    
    if (os.path.exists(args.audios_root) == False):
        print("audios root directory does not exist")
        exit()

    if (args.selective_enable >> 2) & 1:
        os.system("python {} --in_directory={} --out_directory_root={} {}".format(
            os.path.join(sys.path[0], 'apply_fx.py'),
            os.path.join(args.audios_root, "no_effect"), 
        args.audios_root, 
        " ".join([label for label in config.labels if label != "no_effect"])))

    if (args.selective_enable >> 1) & 1:
        os.system("python {} --audios_root={} --output_dir={}".format(
            os.path.join(sys.path[0], 'pack_to_hdf5.py'),
            args.audios_root, 
            os.path.join(args.audios_root, "hdf5s/waveforms/")))
            
    if (args.selective_enable >> 0) & 1:
        os.system("python {} --waveforms_hdf5_path={} --indexes_hdf5_path={}".format(
            os.path.join(sys.path[0], 'create_indexes.py'),
            os.path.join(args.audios_root, "hdf5s/waveforms/"),
            os.path.join(args.audios_root, "hdf5s/indexes/")))