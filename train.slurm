#!/bin/bash
#SBATCH -A m4431
#SBATCH -C gpu
#SBATCH -q regular
#SBATCH -t 00:30:00
#SBATCH -N 2
#SBATCH --ntasks-per-node=1
#SBATCH --gpus-per-node=4
#SBATCH --mail-user=atharv.thakur@rutgers.edu
#SBATCH --mail-type=begin

export MASTER_ADDR=$(scontrol show hostname $SLURM_JOB_NODELIST | head -n 1)
export MASTER_PORT=29500
export OMP_NUM_THREADS=16

ml conda
conda activate fx_removal_env

srun -N $1 --ntasks-per-node=1 --gpus-per-node=4 torchrun \
        --nnodes=$1 \
        --nproc_per_node=4 \
        --rdzv_id=$SLURM_JOB_ID \
        --rdzv_backend=c10d \
        --rdzv_endpoint=$MASTER_ADDR:$MASTER_PORT \
        $SCRATCH/AudioEffectRemoval/audioset_tagging_cnn/pytorch/train.py \
                --working_root=$SCRATCH/AudioEffectRemoval/ \
                --batch_size=96 \
                --gpus_per_node=4 \
                --nodes=$1 \
                --checkpoint="lightning_logs/version_33/checkpoints/epoch=5-step=5640.ckpt" \
                --indexes_hdf5_path=$SCRATCH/AudioEffectRemoval/updated_dataset/hdf5s/indexes/full_indexes.h5 \
