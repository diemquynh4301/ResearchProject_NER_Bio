#!/bin/bash 
#SBATCH --job-name=Qwen3-4B-Instruct-2507_genia_update-100x5
#SBATCH --output=%x_%j.out
#SBATCH --error=%x_%j.err
#SBATCH --gres=gpu:1
#SBATCH --ntasks-per-node=1
#SBATCH --nodes=1
#SBATCH --time=20:00:00 
#SBATCH --cpus-per-task=16
#SBATCH --partition=L40S 

cd ~/ResearchProject_NER_Bio

source ~/miniconda3/etc/profile.d/conda.sh
conda activate mf-ner

python -m update_gl --model_name_or_path="Qwen/Qwen3-4B-Instruct-2507" --output_dir="results/Qwen3-4B-Instruct-2507_genia_update-100x5-1" --max_new_tokens=200000 --n_updates=100