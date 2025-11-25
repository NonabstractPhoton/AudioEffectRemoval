import argparse 
import os
import sox 
from pedalboard.io import AudioFile
import sys
sys.path.insert(1, os.path.join(sys.path[0], '../audioset_tagging_cnn/utils'))
import config
from pedalboard import Pedalboard, Distortion, Bitcrush
import multiprocessing as mp

import itertools

import config

def main():
    # temporary solution
    parser = argparse.ArgumentParser(prog='apply_fx',description='Apply *Individual* Effects to .wav Datasets')
    parser.add_argument('--in_directory')
    parser.add_argument('--out_directory_root')
    parser.add_argument('effects', nargs='+')
    args = parser.parse_args()
    
    if (len(args.effects) > config.max_multilabels):
        print(f"Maximum number of effects exceeded. Will apply up to {config.max_multilabels} effects at a time.")

    if (os.path.exists(args.in_directory) == False):
        print("input directory does not exist")
        return

    valid_effects = [fx for fx in args.effects if fx in config.labels]

    for r in range(1, min(config.max_multilabels, len(valid_effects))+1):
        effect_combinations = itertools.combinations(valid_effects, r)
        for effect_set in effect_combinations:
            create_effected_dataset(args, effect_set)

def create_effected_dataset(args, effects):


    if (os.path.exists(args.out_directory_root) == False):
        os.mkdir(args.out_directory_root)

    tfm = sox.Transformer()
    fx = []


    fxstring = "-".join(effects)
    
    target_dir = os.path.join(args.out_directory_root, fxstring)
    if (not os.path.exists(target_dir)):
        os.mkdir(target_dir)

    print(f"Applying effects: {fxstring} to create dataset at {target_dir}")

    pedalboard_needed = False

    for effect in effects:
        if (effect == 'chorus'):
            tfm.chorus()
        elif (effect == 'flanger'):
            tfm.flanger(depth=5,regen=10,speed=.6)
        elif (effect == 'reverb'):
            tfm.reverb(100)
        elif effect == 'equalizer':
            tfm.equalizer(220,2,0)
        elif (effect == 'phaser'):
            tfm.phaser()
        elif (effect == 'tremolo'):
            tfm.tremolo()
        elif (effect == 'distortion'):
            pedalboard_needed = True
            fx.append(Pedalboard([Distortion()]))
            
        elif (effect == 'bitcrusher'):
            pedalboard_needed = True
            fx.append(Pedalboard([Bitcrush()]))

        elif (effect == 'overdrive'):
            tfm.overdrive(25,30)
        elif effect == 'compressor':
            tfm.compand()

    # Only list regular .wav files from the input directory to avoid directories causing IsADirectoryError
    dir_list = [f for f in os.listdir(args.in_directory)
                if os.path.isfile(os.path.join(args.in_directory, f)) and f.lower().endswith('.wav')]

    if not dir_list:
        print("No .wav files found in input directory:", args.in_directory)
        return

    proc_count = min(mp.cpu_count(), len(dir_list))
    dir_sublists = [dir_list[i::proc_count] for i in range(proc_count)]


    def apply_fx(sub_list, fx=fx, tfm=tfm, in_dir=args.in_directory, target_dir=target_dir):
        for filename in sub_list:
            out_path = os.path.join(target_dir,filename)
            in_path = os.path.join(in_dir,filename)

            with AudioFile(os.path.join(in_dir,filename)) as in_file:
                
                audio = in_file.read(in_file.frames)
                
                for (effect) in fx:
                    audio = effect(audio, in_file.samplerate)
                    
                tfm.build_file(
                    input_array=audio.T,
                    sample_rate_in=in_file.samplerate,
                    output_filepath=out_path
                )

    processes = [mp.Process(target=apply_fx,args=(dir_sublists[i],)) for i in range(proc_count)]
    
    for p in processes:
        p.start()   
    for p in processes:
        p.join()
    
if __name__ == "__main__":
    main()
