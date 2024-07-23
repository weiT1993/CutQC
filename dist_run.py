import os, math
import os, logging
import torch
import torch.distributed as dist

from cutqc.main import CutQC 
from helper_functions.benchmarks import generate_circ
from datetime import timedelta

# Environment variables set by slurm script
gpus_per_node = int (os.environ["SLURM_GPUS_ON_NODE"])
WORLD_RANK = int(os.environ["SLURM_PROCID"])
WORLD_SIZE = int(os.environ["WORLD_SIZE"])
LOCAL_RANK = WORLD_RANK - gpus_per_node * (WORLD_RANK // gpus_per_node)
MASTER_RANK = 0
backend = "nccl"

import os

def run():
    filename = "aqft_30_10"
    full_path = "{}.pkl".format(filename) # os.path.join  (dirname, "{}.pkl".format(filename))
    print (full_path)
    
    cutqc = CutQC(
        build_only=True,
        load_data=full_path,
        parallel_reconstruction=True,
        local_rank=LOCAL_RANK,
    )

    compute_time = cutqc.build(mem_limit=32, recursion_depth=1)
    approximation_error = cutqc.verify()

    # Define the path for the output text file
    dirname = "data_measurements"
    output_file_path = os.path.join(dirname, "{}.txt".format(filename))

    # Write compute time and approximation error to the file
    with open(output_file_path, 'w') as file:
        file.write(f"Compute Time: {compute_time}\n")
        file.write(f"Approximation Error: {approximation_error}\n")

    
    cutqc.destroy_distributed ()
        


  
def init_processes(backend):
    
    dist.init_process_group(backend, rank=WORLD_RANK, world_size=WORLD_SIZE, timeout=timedelta(hours=1))
    print ("Hello world! This is worker: {}. I have {} siblings!".format (dist.get_rank(), dist.get_world_size()))
    run ()

if __name__ == "__main__":
    print(f"args.backend:{backend}")
    print("Local Rank: {}".format(LOCAL_RANK))
    print("World Rank: {}".format(WORLD_RANK))
    print("World Size: {}".format(WORLD_SIZE))
    print ("GPUS-Avail: {}".format (gpus_per_node))
    init_processes(backend=backend)

    
