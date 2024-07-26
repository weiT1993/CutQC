import argparse
import os, math
import os, logging
import torch
import torch.distributed as dist

from cutqc.main import CutQC 
from helper_functions.benchmarks import generate_circ
import os

def run():
    # Instantiate the parser
    parser = argparse.ArgumentParser(description='Optional app description')    
    
    # Required positional argument
    parser.add_argument('circuit_type', type=str, nargs='?')
    parser.add_argument('circuit_size', type=int, nargs='?')
    parser.add_argument('max_width', type=int, nargs='?')
    args = parser.parse_args ()

    filename = "{}_{}_{}".format (args.circuit_type, args.circuit_size, args.max_width)
    full_path = "{}.pkl".format(filename) # os.path.join  (dirname, "{}.pkl".format(filename))
    print (f'--- Running {full_path} ---')
    
    cutqc = CutQC(
        build_only=True,
        load_data=full_path,
        parallel_reconstruction=False
    )

    compute_time = cutqc.build(mem_limit=32, recursion_depth=1)

    # Define the path for the output text file
    dirname = "data_measurements"
    output_file_path = os.path.join(dirname, "{}.tf".format(filename))

    # Write compute time and approximation error to the file
    with open(output_file_path, 'w') as file:
        file.write(f"Compute Time: {compute_time}\n")
        file.write (f"TF Version\n")        
          
if __name__ == "__main__":
    run ()