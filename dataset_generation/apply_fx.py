import argparse 
import os
from pysndfx import AudioEffectsChain 
from scipy.io.wavfile import read, write
import multiprocessing as mp
import torchaudio.functional as F
from torch import as_tensor
from torch import Tensor
def main():

    # jank but this script will only be ran a few times anyway
    with open('audioset_tagging_cnn/utils/config.py') as f:
        config_code = f.read()
        exec(config_code, globals())

    parser = argparse.ArgumentParser(prog='apply_fx',description='Apply Fx to .wav Datasets')
    parser.add_argument('in_directory')
    parser.add_argument('out_directory_root')
    parser.add_argument('effect')
    args = parser.parse_args()

    if (args.effect not in labels):
        print("invalid effect")
        return
    if (args.effect == "vibrato"):
        print("vibrato effect not supported yet")
        return
    if (os.path.exists(args.in_directory) == False):
        print("input directory does not exist")
        return
    if (os.path.exists(args.out_directory_root) == False):
        os.mkdir(args.out_directory_root)
    
    target_dir = os.path.join(args.in_directory,args.effect)
    if (not os.path.exists(target_dir)):
        os.mkdir(target_dir)

    print("Applying {} effect to files in {}".format(args.effect,args.in_directory))
    
    fx = callable
    if (args.effect == 'chorus'):
        fx = AudioEffectsChain().chorus()
    elif (args.effect == 'flanger'):
        fx = lambda wave: F.flanger(as_tensor(wave), sample_rate).numpy()
    elif (args.effect == 'reverb'):
        fx = AudioEffectsChain().reverb()
    elif args.effect == 'equalizer':
        fx = AudioEffectsChain().equalizer()
    elif (args.effect == 'phaser'):
        fx = AudioEffectsChain().phaser()
    elif (args.effect == 'tremolo'):
        fx = AudioEffectsChain().tremolo()
    elif (args.effect == 'distortion'):
        from audioFX.Fx import Fx
        def distortion(x):
            effect = Fx(sample_rate)
            fx_chain = {"distortion": 1}
            return effect.process_audio(x, fx_chain)
        fx = distortion
    elif (args.effect == 'wah'):
        def wah(x):
            from audioFX.Fx import Fx
            effect = Fx(sample_rate)
            fx_chain = {"wahwah": 1}
            return effect.process_audio(x, fx_chain)
        fx = wah
    elif (args.effect == 'overdrive'):
        def overdrive(x):
            return F.overdrive(as_tensor(x)).numpy()
        fx = overdrive
    elif args.effect == 'compressor':
        fx = AudioEffectsChain().compand()

    dir_list = os.listdir(args.in_directory)
    dir_sublists = [dir_list[i::mp.cpu_count()] for i in range(mp.cpu_count())]
    def apply_fx_to_files(sub_list, fx=fx, in_dir=args.in_directory, target_dir=target_dir):
        for filename in sub_list:
            sr, audio = read(os.path.join(in_dir,filename))
            effected_audio = fx(audio)
            write(os.path.join(target_dir,filename),sr,effected_audio)

    processes = [mp.Process(target=apply_fx_to_files,args=(dir_sublists[i],)) for i in range(mp.cpu_count())]
    for p in processes:
        p.start()   
    for p in processes:
        p.join()
    
    

if __name__ == "__main__":
    main()
