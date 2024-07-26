import os
from os.path import exists
import argparse

from cutqc.main import CutQC 
from helper_functions.benchmarks import generate_circ

def write_results(dirname, filename, results):
    """
    Write results to output file
    """
    output_file_path = os.path.join(dirname, f"{filename}.tf")
    with open(output_file_path, 'w') as file:
        file.write(f"Compute Time: {results[0]}\n")
        file.write(f"Approximation Error: {results[1]}\n")
        file.write(f"TF Version")

def run(args):
    # Construct filepath for target pickle
    filename = f"{args.circuit_type}_{args.circuit_size}_{args.max_width}"
    dirname = '_pickel_files'
    full_path = f"{dirname}/{filename}.pkl"
    assert exists(full_path), "Error 2003: Pickle File '{}' Not Found".format (full_path)

    # Load CutQC Instance from Pickle
    print(f'--- Running {full_path} ---')
    cutqc = CutQC(
        build_only=True,
        load_data=full_path,
        parallel_reconstruction=False
    )


    # Initiate Reconstruct
    compute_time = cutqc.build(mem_limit=4, recursion_depth=10)
    approximation_error = cutqc.verify()

    # Define the path for the output text file
    dirname = "data_measurements"
    write_results(dirname, filename, (compute_time, approximation_error))

          

if __name__ == "__main__":
    # Required positional arguments
    parser = argparse.ArgumentParser(description='Optional app description')    
    parser.add_argument('circuit_type', type=str, nargs='?')
    parser.add_argument('circuit_size', type=int, nargs='?')
    parser.add_argument('max_width', type=int, nargs='?')
    parser.add_argument('backend', type=str, nargs='?')
    args = parser.parse_args()    
    run(args)    


