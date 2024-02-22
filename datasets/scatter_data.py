# Copyright (C) 2023 Cyprien Quéméneur
# For the full license, please refer to the LICENSE file in the root directory of this project.
# For the full copyright notices, please refer to the NOTICE file in the root directory of this project.

import argparse
import os
import shutil
import tarfile
from mpi4py import MPI


def copy_and_extract(dataset: str, local_storage: str, file: str) -> None:
    """Copy and extract the local dataset."""
    destination_directory = f'{local_storage}/fedyolo-iov/datasets/{dataset}/'
    shutil.copy(f'datasets/{dataset}/{file}', destination_directory)
    tar_file_path = f'{local_storage}/fedyolo-iov/datasets/{dataset}/{file}'
    with tarfile.open(tar_file_path, 'r') as tar:
        tar.extractall(path=destination_directory)


def transfer_local_dataset(dataset: str, node_rank: int) -> None:
    """Copy the local dataset from the network storage to the compute node local storage (meant for temporary files)."""
    slurm_tmpdir = os.environ.get('SLURM_TMPDIR')  # path to the compute node local storage
    print(f'Node of rank {node_rank}. TMPDIR is {slurm_tmpdir}')
    if node_rank == 0:
        copy_and_extract(dataset, slurm_tmpdir, 'server.tar')
    else:
        copy_and_extract(dataset, slurm_tmpdir, f'client{node_rank}.tar')
    print(f"Node of rank {node_rank}."
          f" Datasets directory files: {os.listdir(f'{slurm_tmpdir}/fedyolo-iov/datasets/')}"
          f" Local dataset files: {os.listdir(f'{slurm_tmpdir}/fedyolo-iov/datasets/{dataset}/')}")


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset', type=str, help='name of the dataset')
    args = parser.parse_args()
    comm = MPI.COMM_WORLD
    rank = comm.Get_rank()
    transfer_local_dataset(args.dataset, rank)
