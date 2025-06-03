#!/bin/bash
#SBATCH --job-name=dino_attention
#SBATCH --partition=capella          
#SBATCH --nodes=1                    
#SBATCH --gres=gpu:4                 # GPU number
#SBATCH --mem=128G                    
#SBATCH --time=20:00:00              
#SBATCH --output=log/%x_%j.out           # standard output
#SBATCH --error=log/%x_%j.err   
#SBATCH --mail-user=qingyue.zhou@mailbox.tu-dresden.de  

module purge
module load release/24.04 GCC/12.3.0 OpenMPI/4.1.5 scikit-build/0.17.6 scikit-learn/1.3.1  # dependencies

source /home/qizh093f/dino-main/venv_dino/bin/activate
  
export WORK=/home/qizh093f/dino-main
export TORCH_HOME=$WORK/torch_cache          # all ranks -> same cache
mkdir -p $TORCH_HOME

srun torchrun --nproc_per_node=4 visualize_attention.py \
    --arch vit_small \
    --pretrained_weights ./output/default_settings/checkpoint0100.pth \
    --patch_size 8 \
    --image_path train-100/01/single_organoid_100.tif \
    --output_dir ./output/default_settings/attention_maps

##SBATCH --cpus-per-task=4            # CPU cores

#VENV=/home/h5/qizh093f/dino-main/venv_dino
#$VENV/bin/python -m pip install --upgrade pip faiss-gpu==1.7.4.post2  # 确保安装在 venv
#srun $VENV/bin/torchrun --nproc_per_node=1 eval_knn.py \
#    --data_path /home/qizh093f/dino-main/imagenette \
#    --pretrained_weights dino_vit_small.pth

