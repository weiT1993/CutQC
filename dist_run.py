import os
from os.path import exists
import argparse
from cutqc.main import CutQC 

# Environment variables set by slurm script
GPUS_PER_NODE = int(os.environ["SLURM_GPUS_ON_NODE"])
WORLD_RANK = int(os.environ["SLURM_PROCID"])
WORLD_SIZE = int(os.environ["WORLD_SIZE"])

def write_results(dirname, filename, results):
    """
    Write results to output file
    """
    output_file_path = os.path.join(dirname, f"{filename}.pt")
    with open(output_file_path, 'w') as file:
        file.write(f"Compute Time: {results[0]}\n")
        file.write(f"Approximation Error: {results[1]}\n")
        file.write(f"PT Version, {WORLD_SIZE} nodes")

def run(args):
    # Construct filepath for target pickle
    filename = f"{args.circuit_type}_{args.circuit_size}_{args.max_width}"
    dirname = '_pickel_files'
    full_path = f"{dirname}/{filename}.pkl"
    assert exists(full_path), "Error: Pickle File '{}' Not Found".format (full_path)

    # Load CutQC Instance from Pickle
    print(f'--- Running {full_path} ---')
    cutqc = CutQC (
        reconstruct_only=True,
        load_data=full_path,
        parallel_reconstruction=True,
        local_rank=LOCAL_RANK,
        compute_backend=args.compute_backend
    )

    # Initiate Reconstruct
    compute_time = cutqc.build(mem_limit=32, recursion_depth=1)
    approximation_error = cutqc.verify()

    print ("-- Done --s")
    # Define the path for the output text file
    dirname = "data_measurements"
    filename = f"{filename}_{args.comm_backend}_{args.compute_backend}_nodes{WORLD_SIZE}_v2"
    write_results(dirname, filename, (compute_time, approximation_error))
    
    cutqc.destroy_distributed()
          
def init_processes(args):
    dist.init_process_group(args.comm_backend, rank=WORLD_RANK, world_size=WORLD_SIZE, timeout=timedelta(hours=1))
    print(f"Hello world! This is worker: {dist.get_rank()}. I have {dist.get_world_size()} siblings!")
    run(args)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Optional app description')
    
    # Required positional arguments
    parser.add_argument('circuit_type', type=str, nargs='?')
    parser.add_argument('circuit_size', type=int, nargs='?')
    parser.add_argument('max_width', type=int, nargs='?')
    parser.add_argument('comm_backend', type=str, nargs='?')
    parser.add_argument('compute_backend', type=str, nargs='?')

    args = parser.parse_args()
    
    print(f"args.compute_backend: {args.compute_backend}")
    print(f"args.comm_backend: {args.comm_backend}")
    print(f"Local Rank: {LOCAL_RANK}")
    print(f"World Rank: {WORLD_RANK}")
    print(f"World Size: {WORLD_SIZE}")
    print(f"GPUS-Avail: {gpus_per_node}")
    
    init_processes(args=args)    
