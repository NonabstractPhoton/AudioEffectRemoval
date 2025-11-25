import os
import argparse
import sys
sys.path.insert(1, os.path.join(sys.path[0], '../utils'))
import config as baked_config
from dataset import (FXSet, collate_fn)
    
import torch
from torch.utils.data import DataLoader


import logging
import lightning as L
from models import FXClassifier

def main(config):

    train_set =  FXSet(config.train_indexes_hdf5_path)
    val_set =  FXSet(config.val_indexes_hdf5_path)

    '''
    train_sampler = BalancedTrainSampler(train_set, config.nodes * config.gpus_per_node, os.environ['RANK'],
        indexes_hdf5_path=config.train_indexes_hdf5_path, 
        batch_size=config.batch_size,
        black_list_csv=None)
    
    validation_sampler = EvaluateSampler(val_set, config.nodes * config.gpus_per_node, os.environ['RANK'],
        indexes_hdf5_path=config.val_indexes_hdf5_path, batch_size=config.batch_size)
    '''

    train_loader = DataLoader(dataset=train_set, collate_fn=collate_fn, 
        num_workers=config.num_workers, pin_memory=True, batch_size=config.batch_size, shuffle=True)
    
    validation_loader = DataLoader(dataset=val_set, collate_fn=collate_fn, 
        num_workers=config.num_workers, pin_memory=True, batch_size=config.batch_size, shuffle=False)
    

    model = FXClassifier(sample_rate=44100,window_size=1024, 
        hop_size=320, fmin=50, fmax=14000, 
        learning_rate=config.learning_rate, 
        classes_num=baked_config.classes_num)

    # TODO renable
    torch.compile(model)
    print("Model Compiled")
    
    torch.set_float32_matmul_precision('medium')
    torch.backends.cudnn.conv.fp32_precision = 'tf32'

    trainer = L.Trainer(
        max_epochs=10, accelerator='gpu', devices=config.gpus_per_node, 
        strategy='ddp',num_nodes=config.nodes, default_root_dir=config.working_root,
        log_every_n_steps=50, enable_checkpointing=True, precision="bf16-mixed")

    if (config.checkpoint is not None) and (os.path.isfile(config.checkpoint)):
        trainer.fit(model, train_loader, validation_loader,ckpt_path=config.checkpoint)
    else:    
        trainer.fit(model, train_loader, validation_loader)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--learning_rate', type=float, help='Learning rate for optimizer', default=1e-4)
    parser.add_argument('--batch_size', type=int, help='Batch size for each GPU in training',default=128)
    parser.add_argument('--num_workers', type=int, help='Number of workers for data loading', default=15)
    parser.add_argument('--gpus_per_node'  , type=int, help='Number of GPUs to use', default=1)
    parser.add_argument('--nodes', type=int, help='Number of nodes to use', default=1)
    parser.add_argument('--working_root', type=str, help='Working root directory', default='./')
    parser.add_argument('--train_indexes_hdf5_path', type=str, help='Path to training indexes HDF5 file', required=True)
    parser.add_argument('--val_indexes_hdf5_path', type=str, help='Path to validation indexes HDF5 file', required=True)
    parser.add_argument("--checkpoint", type=str, help="Path to checkpoint file to resume from", default=None)
    config = parser.parse_args()
    main(config)