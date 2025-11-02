import argparse 
import os
import sox 
from pedalboard.io import AudioFile

from pedalboard import Pedalboard, Distortion, Bitcrush
import multiprocessing as mp
import load_config

def main():

    # temporary solution
    load_config.load_config(globals())

    parser = argparse.ArgumentParser(prog='apply_fx',description='Apply *Individual* Effects to .wav Datasets')
    parser.add_argument('--in_directory')
    parser.add_argument('--out_directory_root')
    parser.add_argument('effects', nargs='+')
    args = parser.parse_args()

    if (os.path.exists(args.in_directory) == False):
                print("input directory does not exist")
                return
    if (os.path.exists(args.out_directory_root) == False):
        os.mkdir(args.out_directory_root)

    for effect in args.effects:
        if (effect not in labels):
            print("invalid effect")
            continue      

        target_dir = os.path.join(args.out_directory_root, effect)
        if (not os.path.exists(target_dir)):
            os.mkdir(target_dir)
        
        print("Applying {} effect to files in {}".format(effect,args.in_directory))
        pedalboard_needed = False
        tfm = sox.Transformer()
        fx = callable
        if (effect == 'chorus'):
            tfm.chorus()
        elif (effect == 'flanger'):
            tfm.flanger(depth=5,regen=10,speed=.6)
        elif (effect == 'reverb'):
            tfm.reverb(60)
        elif effect == 'equalizer':
            tfm.equalizer(220,2,0)
        elif (effect == 'phaser'):
            tfm.phaser()
        elif (effect == 'tremolo'):
            tfm.tremolo()
        elif (effect == 'distortion'):
            pedalboard_needed = True
            fx = Pedalboard([Distortion()])
            
        elif (effect == 'bitcrusher'):
            pedalboard_needed = True
            fx = Pedalboard([Bitcrush()])

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

        def apply_pedalboard_to_files(sub_list, fx=fx, in_dir=args.in_directory, target_dir=target_dir):
            for filename in sub_list:
                print("Processing file: {}".format(os.path.join(in_dir,filename)))
                with AudioFile(os.path.join(in_dir,filename)) as f:
                    with AudioFile(os.path.join(target_dir,filename), 'w', f.samplerate, f.num_channels) as out_f:
                        audio = f.read(f.frames)
                        effected_audio = fx(audio, f.samplerate)
                        out_f.write(effected_audio)

        def apply_tf_to_files(sub_list, tfm=tfm, in_dir=args.in_directory, target_dir=target_dir):
            for filename in sub_list:
                print("Processing file: {}".format(os.path.join(in_dir,filename)))
                tfm.build_file(os.path.join(in_dir,filename),os.path.join(target_dir,filename))

        processes = []
        if (pedalboard_needed):
            processes = [mp.Process(target=apply_pedalboard_to_files,args=(dir_sublists[i],)) for i in range(proc_count)]
        else:
            processes = [mp.Process(target=apply_tf_to_files,args=(dir_sublists[i],)) for i in range(proc_count)]
        for p in processes:
            p.start()   
        for p in processes:
            p.join()
        
if __name__ == "__main__":
    main()
