#!/bin/bash
#SBATCH --job-name=train_rpn_jrdb_fulldataset
#SBATCH --output=/home/hu/Projects/slurm_logs/%x_%J_point_rpn_jrdb.log
#SBATCH --mail-type=ALL
#SBATCH --mail-user=huzjkevin@gmail.com
#SBATCH --partition=lopri
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --gres=gpu:1
#SBATCH --time=3-00:00:00
#SBATCH --signal=TERM@120
cd $HOME/Projects/PointRCNN/tools
wandb on
srun --unbuffered python train_rcnn.py --cfg_file cfgs/default.yaml --batch_size 8 --train_mode rpn --epochs 100 --train_with_eval