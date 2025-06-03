#!/bin/bash
#SBATCH --job-name=test
#SBATCH --partition=capella          
#SBATCH --nodes=1                    
#SBATCH --gres=gpu:1                 # GPU number
#SBATCH --mem=32G                    
#SBATCH --time=1:00:00    

echo "Test successful" 