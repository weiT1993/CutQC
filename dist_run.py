import argparse
import os
import torch.distributed as dist
from datetime import timedelta
from cutqc.main import CutQC 


# Environment variables set by slurm script
gpus_per_node = int (os.environ["SLURM_GPUS_ON_NODE"])
WORLD_RANK = int(os.environ["SLURM_PROCID"])
WORLD_SIZE = int(os.environ["WORLD_SIZE"])
LOCAL_RANK = WORLD_RANK - gpus_per_node * (WORLD_RANK // gpus_per_node)
MASTER_RANK = 0

<<<<<<< HEAD
import os

def run (circuit_size, max_subcircuit_width, circuit_type):
    filename = "{}_{}_{}".format (circuit_type, circuit_size, max_subcircuit_width)
    full_path = "pickle_files/{}.pkl".format(filename) # os.path.join  (dirname, "{}.pkl".format(filename))
    print (full_path)
=======
def write_results (dirname, filename, results):
    '''
    Write results to output file
    '''
    output_file_path = os.path.join(dirname, "{}.pt".format(filename))
    with open(output_file_path, 'w') as file:
        file.write(f"Compute Time: {results[0]}\n")
        file.write ("PT Version, {} nodes".format (WORLD_SIZE))        
        
def run(args):
    filename = "{}_{}_{}".format (args.circuit_type, args.circuit_size, args.max_width)
    full_path = "{}.pkl".format(filename) 
    print (f'--- Running {full_path} ---')
>>>>>>> e2b2958dfeffcb6c2e879ecb0b36a223bb3aaaec
    
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
    filename = "{}_{}_nodes{}".format(filename, args.backend, WORLD_SIZE)
    write_results (dirname, filename, (compute_time, approximation_error))    
    cutqc.destroy_distributed ()
          
def init_processes(args):
    
    dist.init_process_group(args.backend, rank=WORLD_RANK, world_size=WORLD_SIZE, timeout=timedelta(hours=1))
    print ("Hello world! This is worker: {}. I have {} siblings!".format (dist.get_rank(), dist.get_world_size()))
    run (args)

if __name__ == "__main__":
<<<<<<< HEAD
    parser = argparse.ArgumentParser(description="Run CutQC with given parameters")
    parser.add_argument('--circuit_size', type=int, required=True, help='Size of the circuit')
    parser.add_argument('--max_subcircuit_width', type=int, required=True, help='Max width of subcircuit')
    parser.add_argument('--circuit_type', type=str, required=True, help='Circuit Type')
    args = parser.parse_args()

    print(f"args.backend:{backend}")
=======
    parser = argparse.ArgumentParser(description='Optional app description')    
    
    # Required positional argument
    parser.add_argument('circuit_type', type=str, nargs='?')
    parser.add_argument('circuit_size', type=int, nargs='?')
    parser.add_argument('max_width', type=int, nargs='?')
    parser.add_argument('backend', type=str, nargs='?')
    args = parser.parse_args ()
    
    print(f"args.backend:{args.backend}")
>>>>>>> e2b2958dfeffcb6c2e879ecb0b36a223bb3aaaec
    print("Local Rank: {}".format(LOCAL_RANK))
    print("World Rank: {}".format(WORLD_RANK))
    print("World Size: {}".format(WORLD_SIZE))
    print ("GPUS-Avail: {}".format (gpus_per_node))
    init_processes(args=args)

    

    
