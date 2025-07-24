#!/bin/bash
#SBATCH --job-name=us1
#SBATCH --partition=gpu
#SBATCH --nodes=1
#SBATCH --gres=gpu:4
#SBATCH --cpus-per-task=7          
#SBATCH --time=04:00:00
#SBATCH --output=logs/us1_%j.out
#SBATCH --error=logs/us1_%j.err



conda activate your envs





cd path/to/selectiontools


        srun python main.py \
          --metric LeastConfidence \
          --budget 0.1 \
          --output_dir "outpus/LeastConfidence_10pt" #save results
#metric{LeastConfidence,Entropy Sampling,Margin Sampling,BALDDropout,
# BatchBALD,EGL,SPUQ,UQ_ICL,Learning Loss,LESS,STAFF,Random,BM25
#CoreSet,Kmeans,Cluster Margin,BADGE,Contrastive ,DatasetQuantization,ZIP}
#budget > 0 && budget <= 1