#!/bin/bash
#SBATCH --job-name=knn_eval          
#SBATCH --partition=capella          
#SBATCH --nodes=1                    
#SBATCH --gres=gpu:1                 # GPU number
#SBATCH --cpus-per-task=4            # CPU cores
#SBATCH --mem=16G                    
#SBATCH --time=03:00:00              
#SBATCH --output=log/%x_%j.out           # standard output
#SBATCH --error=log/%x_%j.err     
#SBATCH --mail-user=qingyue.zhou@mailbox.tu-dresden.de

#module list     # 记录已加载模块
#nvidia-smi      # 确认 GPU 可见

module purge
module load release/24.04 GCC/12.3.0 OpenMPI/4.1.5 PyTorch/2.1.2-CUDA-12.1.1 PyTorch-bundle/2.1.2-CUDA-12.1.1 scikit-build/0.17.6  # dependencies

source ~/dino-main/venv_dino/bin/activate 
srun torchrun --nproc_per_node=1 eval_knn.py \
       --data_path $WORK/dino-main/imagenette \
       --pretrained_weights dino_vit_small.pth
