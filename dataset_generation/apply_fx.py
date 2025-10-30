import argparse 
import os
import sys
sys.path.insert(1,'../audioset_tagging_cnn/utils')
import config


def main():
    parser = argparse.ArgumentParser(prog='apply_fx',description='Apply Fx to .wav Datasets')
    parser.add_argument('in-directory')
    parser.add_argument('out-directory-root')
    parser.add_argument('effect')
    args = parser.parse_args()

    if (args.effect not in config.labels):
        print("invalid effect")
        return;
    if (os)


if __name__ == "__main__":
    main()
