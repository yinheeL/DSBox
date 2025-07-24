# DSBox
1. Introduction

The DSBox is a lightweight, flexible toolkit for active learning and sample selection. It integrates 20 state-of-the-art selection strategies under a unified interface, enabling you to quickly identify valuable training samples and improve model performance with minimal effort.

2. Usage

2.1 Prepare Dataset

Clone this repository.

Download the DEVIGN dataset and place it under data/devign:

[git clone https://github.com/DeepDevign/Devign.git data/devign](https://sites.google.com/view/devign)

2.2 Usage 1: Select Data via select.py

Edit or pass parameters to select.py:


python select.py 


After running, the script will print the selected indices.

2.3 Usage 2: Slurm Job via run.slurm

Configure parameters at the top of run.slurm (partition, GPU count, memory, time，metric，budget).
Submit the job with arguments:

sbatch run.slurm 

The run.slurm script will:

Load necessary modules 


Launch training and evaluation steps.

Output logs to logs/ directory.

3. Requirements

Python >= 3.9

PyTorch >= 2.1.2
