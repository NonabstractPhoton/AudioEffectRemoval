import argparse 
import os
import sox 
from scipy.io.wavfile import read, write
import multiprocessing as mp

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

    target_dir = os.path.join(args.out_directory_root, args.effect)
    if (not os.path.exists(target_dir)):
        os.mkdir(target_dir)
    
    print("Applying {} effect to files in {}".format(args.effect,args.in_directory))
    gpu_needed = False
    audioFx_needed = False
    tfm = sox.Transformer()
    fx = callable
    if (args.effect == 'chorus'):
        fx = AudioEffectsChain().chorus()
    elif (args.effect == 'flanger'):
        import torchaudio.functional as F 
        import torch 
        gpu_needed = True
        def flanger(x,device_index=0):
            return F.flanger(torch.as_tensor(x, dtype=torch.float32).to(torch.device('cuda',device_index)),sample_rate).mul(2**16/2).to(torch.int16).numpy(force=True)
        fx = flanger
    elif (args.effect == 'reverb'):
        tfm.reverb()
    elif args.effect == 'equalizer':
        tfm.equalizer()
    elif (args.effect == 'phaser'):
        tfm.phaser()
    elif (args.effect == 'tremolo'):
        tfm.tremolo()
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
        tfm.overdrive(10,30)
    elif args.effect == 'compressor':
        tfm.compand()

    # Only list regular .wav files from the input directory to avoid directories causing IsADirectoryError
    dir_list = [f for f in os.listdir(args.in_directory)
				if os.path.isfile(os.path.join(args.in_directory, f)) and f.lower().endswith('.wav')]

    if not dir_list:
        print("No .wav files found in input directory:", args.in_directory)
        return

    proc_count = min(mp.cpu_count() if not gpu_needed else 4, len(dir_list))
    dir_sublists = [dir_list[i::proc_count] for i in range(proc_count)]

    def apply_fx_to_files(sub_list, index, fx=fx, in_dir=args.in_directory, target_dir=target_dir):
        for filename in sub_list:
            print("Processing file: {}".format(os.path.join(in_dir,filename)))
            sr, audio = read(os.path.join(in_dir,filename))
            effected_audio = fx(audio, index) if gpu_needed else fx(audio)
            write(os.path.join(target_dir,filename),sr,effected_audio)
    def apply_tf_to_files(sub_list, tfm=tfm, in_dir=args.in_directory, target_dir=target_dir):
        for filename in sub_list:
            print("Processing file: {}".format(os.path.join(in_dir,filename)))
            tfm.build_file(os.path.join(in_dir,filename),os.path.join(target_dir,filename))

    processes = []
    if (gpu_needed or audioFx_needed):
        processes = [mp.Process(target=apply_fx_to_files,args=(dir_sublists[i],i)) for i in range(proc_count)]
    else:
        processes = [mp.Process(target=apply_tf_to_files,args=(dir_sublists[i],)) for i in range(proc_count)]
    for p in processes:
        p.start()   
    for p in processes:
        p.join()
    
if __name__ == "__main__":
    main()
