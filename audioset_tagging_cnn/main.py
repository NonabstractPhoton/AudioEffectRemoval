import os
import argparse
import sys
sys.path.insert(1, os.path.join(sys.path[0], '../utils'))
from data_generator import (AudioSetDataset,BalancedTrainSampler, 
    EvaluateSampler, collate_fn)
    
from torch,utils.data import DataLoader

import lightning as L
from models import Wavegram_Logmel128_Cnn14

def main(config):

    # Adapted Original Code -----------------------------

    train_sampler = BalancedTrainSampler(
        indexes_hdf5_path=config.train_indexes_hdf5_path, 
        batch_size=config.batch_size,
        black_list_csv=None)
    
    # Evaluate sampler
    validation_sampler = EvaluateSampler(
        indexes_hdf5_path=config.val_indexes_hdf5_path, batch_size=config.batch_size)

    train_loader = DataLoader(dataset=dataset, 
        batch_sampler=train_sampler, collate_fn=collate_fn, 
        num_workers=config.num_workers, pin_memory=True)
    
    validation_loader = DataLoader(dataset=dataset, 
        batch_sampler=validation_sampler, collate_fn=collate_fn, 
        num_workers=config.num_workers, pin_memory=True)
    
    # Lightning Code -----------------------------

    model = Wavegram_Logmel128_Cnn14()
    trainer = L.Trainer(max_epochs=10)
    trainer.fit(model, )

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('batch_size', type=int, help='Batch size for training')
    parser.add_argument('num_workers', type=int, help='Number of workers for data loading')
    parser.add_argument('train_indexes_hdf5_path', type=str, help='Path to training indexes HDF5 file'
    parser.add_argument('val_indexes_hdf5_path', type=str, help='Path to validation indexes HDF5 file')
    config = parser.parse_args()
    main(config)