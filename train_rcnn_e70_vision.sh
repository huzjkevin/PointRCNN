#!/bin/bash
#SBATCH --job-name=train_rcnn_e70
#SBATCH --output=/home/hu/Projects/slurm_logs/%x_%J_train_rcnn_jrdb.log
#SBATCH --mail-type=ALL
#SBATCH --mail-user=huzjkevin@gmail.com
#SBATCH --partition=lopri
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --gres=gpu:1
#SBATCH --time=4-00:00:00
#SBATCH --signal=TERM@120
cd $HOME/Projects/PointRCNN/tools
wandb on
srun --unbuffered python train_rcnn.py --cfg_file cfgs/default.yaml --batch_size 4 --train_mode rcnn --epochs 70  --ckpt_save_interval 2 --ckpt ../output/rcnn/default/ckpt/checkpoint_epoch_18.pth --rpn_ckpt ../output/rpn/default/ckpt/checkpoint_epoch_200.pth


