# Title: dist_driver.py
# Description: Example of how CutQC can be used to effececiently reconstruct 
# subcircuits

import os
from cutqc.main import CutQC 

# Environment variables set by slurm script
GPUS_PER_NODE = int(os.environ["SLURM_GPUS_ON_NODE"])
WORLD_RANK    = int(os.environ["SLURM_PROCID"])
WORLD_SIZE    = int(os.environ["WORLD_SIZE"])

if __name__ == "__main__":
    full_path       = 'adder_example.pkl'
    compute_backend = 'gpu'
    comm_backend    = 'nccl'
    
    # Load CutQC Instance from Pickle
    print(f'--- Running {full_path} ---')
    cutqc = CutQC (
        parallel_reconstruction = True,
        reconstruct_only = True,
        load_data        = full_path,
        compute_backend  = compute_backend,
        comm_backend     = comm_backend,
        gpus_per_node = GPUS_PER_NODE,
        world_rank     = WORLD_RANK,
        world_size    = WORLD_SIZE
    )

    # Initiate Reconstruct
    compute_time = cutqc.build(mem_limit=32, recursion_depth=1)
    approximation_error = cutqc.verify()

    print('--- Reconstruction Complete ---')    
    print ("Total Reconstruction Time:\t{}".format(compute_time))
    print ("Approxamate Error:\t {}".format (approximation_error))
    cutqc.destroy_distributed()    