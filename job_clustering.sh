#!/bin/bash
#SBATCH --job-name=dino_cluster
#SBATCH --partition=capella          
#SBATCH --nodes=1                    
#SBATCH --gres=gpu:1                 # GPU number
#SBATCH --cpus-per-task=4            # CPU cores
#SBATCH --mem=16G                    
#SBATCH --time=03:00:00              
#SBATCH --output=log/%x_%j.out           # standard output
#SBATCH --error=log/%x_%j.err     

module purge
module load release/24.04 GCC/12.3.0 OpenMPI/4.1.5 PyTorch/2.1.2-CUDA-12.1.1 PyTorch-bundle/2.1.2-CUDA-12.1.1 scikit-build/0.17.6 scikit-learn/1.3.1  # dependencies

source /home/qizh093f/dino-main/venv_dino/bin/activate
  

srun torchrun --nproc_per_node=1 \
    clustering.py \
    --data_path /home/qizh093f/dino-main/test-100 \
    --pretrained_weights dino_vit_small.pth \
    --k 10 \
    --out_csv clusters.csv

#VENV=/home/h5/qizh093f/dino-main/venv_dino
#$VENV/bin/python -m pip install --upgrade pip faiss-gpu==1.7.4.post2  # 确保安装在 venv
#srun $VENV/bin/torchrun --nproc_per_node=1 eval_knn.py \
#    --data_path /home/qizh093f/dino-main/imagenette \
#    --pretrained_weights dino_vit_small.pth

