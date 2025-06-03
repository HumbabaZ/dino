#!/bin/bash
#SBATCH --job-name=dino_train_multinode
#SBATCH --partition=capella          
#SBATCH --nodes=4                    
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
export TORCH_HOME=$WORK/torch_cache          # 让所有 rank 指同一 cache
mkdir -p $TORCH_HOME

#srun torchrun --nproc_per_node=4 main_dino.py \
#    --arch vit_small \
#    --data_path /projects/p_rep_learn_2/datasets \
#    --output_dir $WORK/output \
#    --saveckp_freq 20 \

srun torchrun run_with_submitit.py --nodes 4 --ngpus 4 --arch vit_small --data_path /projects/p_rep_learn_2/datasets --output_dir $WORK/output 



