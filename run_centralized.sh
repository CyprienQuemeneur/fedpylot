#!/bin/bash
# FedPylot by Cyprien Quéméneur, GPL-3.0 license
# Example usage: sbatch run_centralized.sh

#SBATCH --nodes=1                        # total number of nodes (only 1 in the centralized setting)
#SBATCH --gpus-per-node=v100l:1          # total of 1 GPU
#SBATCH --ntasks-per-gpu=1               # 1 process is launched
#SBATCH --cpus-per-task=8                # CPU cores per process
#SBATCH --mem-per-cpu=2G                 # host memory per CPU core
#SBATCH --time=3-12:00:00                # time (DD-HH:MM:SS)
#SBATCH --mail-user=myemail@gmail.com    # receive mail notifications
#SBATCH --mail-type=ALL

# Check GPU on compute node
nvidia-smi

# Load modules
module purge
module load python/3.9.6 scipy-stack
module load gcc/9.3.0
module load opencv/4.6.0

# Load pre-existing virtual environment
source ~/venv-py39-fl/bin/activate

# Prepare directory to backup results
saving_path=$(pwd)/results/nuimages10/yolov7/centralized
mkdir -p $saving_path

# Transmit all files besides the datasets and results directories to the local storage of the compute node
rsync -a --exclude="datasets" --exclude="results" ../fedpylot $SLURM_TMPDIR

# Create an empty directory on the compute node local storage to receive the training and validation sets
mkdir -p $SLURM_TMPDIR/fedpylot/datasets/nuimages10

# Transfer the training and validation sets from the network storage to the local storage of the compute node
tar -xf datasets/nuimages10/client0.tar -C $SLURM_TMPDIR/fedpylot/datasets/nuimages10  # train
tar -xf datasets/nuimages10/server.tar -C $SLURM_TMPDIR/fedpylot/datasets/nuimages10   # val

# Move to local storage
cd $SLURM_TMPDIR/fedpylot

# Download pre-trained weights
if [[ $SLURM_PROCID -eq 0 ]]; then
    bash weights/get_weights.sh yolov7
fi

# Run centralized experiment (see yolov7/train.py for more details on the settings)
python yolov7/train.py \
    --client-rank 0 \
    --epochs 150 \
    --weights weights/yolov7/yolov7_training.pt \
    --data data/nuimages10.yaml \
    --batch 32 \
    --img 640 640 \
    --cfg yolov7/cfg/training/yolov7.yaml \
    --hyp data/hyps/hyp.scratch.clientopt.nuimages.yaml \
    --workers 8 \
    --project experiments \
    --name ''

# Backup experiment results to network storage
cp -r ./experiments $saving_path
