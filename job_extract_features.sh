#!/bin/bash
#SBATCH --job-name=dino_extact_features
#SBATCH --partition=capella          
#SBATCH --nodes=1                    
#SBATCH --gres=gpu:1                 # GPU number
#SBATCH --cpus-per-task=4            # CPU cores
#SBATCH --mem=16G                    
#SBATCH --time=03:00:00              
#SBATCH --output=log/%x_%j.out           # standard output
#SBATCH --error=log/%x_%j.err     

module purge
module load release/24.04 GCC/12.3.0 OpenMPI/4.1.5 scikit-build/0.17.6 scikit-learn/1.3.1  # dependencies

source /home/qizh093f/dino-main/venv_dino/bin/activate
  
srun torchrun --nproc_per_node=1 eval_extract_features.py \
  --data_path /home/qizh093f/dino-main/train-100 \
  --output_dir /home/qizh093f/dino-main/features \
  --pretrained_weights dino_vit_small.pth \
  --teacher_temp 0.07 --warmup_teacher_temp_epochs 30
#  --pretrained_weights /home/qizh093f/dino-main/output/checkpoint0000.pth \

#VENV=/home/h5/qizh093f/dino-main/venv_dino
#$VENV/bin/python -m pip install --upgrade pip faiss-gpu==1.7.4.post2  # 确保安装在 venv
#srun $VENV/bin/torchrun --nproc_per_node=1 eval_knn.py \
#    --data_path /home/qizh093f/dino-main/imagenette \
#    --pretrained_weights dino_vit_small.pth

