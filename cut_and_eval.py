# Description: Driver Cuts and Evals a circuit. The result is saved as pickelfile
import argparse
import os, math
import os, logging
import numpy as np

logging.disable(logging.WARNING)
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "1"
os.environ["CUDA_VISIBLE_DEVICES"] = "-1" 

from cutqc.main import CutQC 
from helper_functions.benchmarks import generate_circ

def get_args ():    
    # Instantiate the parser
    parser = argparse.ArgumentParser(description='Optional app description')    
        
    # Required positional argument
    parser.add_argument('circuit_type', type=str, nargs='?')
    parser.add_argument('circuit_size', type=int, nargs='?')
    parser.add_argument('max_width', type=int, nargs='?')
    args = parser.parse_args ()

    return args
    

if __name__ == "__main__":
    args = get_args ()

    circuit_type = args.circuit_type
    circuit_size = args.circuit_size
    max_width = args.max_width
    verbose = False
    
    filename = "{}_{}_{}.pkl".format (circuit_type, circuit_size, max_width)
    dirname = '_pickel_files'
    os.makedirs(dirname, exist_ok=True)
    save_path = "{}/{}".format(dirname,filename)
    
    print (f'--- Cutting and Evalualting {filename} --- ')

    circuit = generate_circ(
        num_qubits=circuit_size,
        depth=1,
        circuit_type=circuit_type,
        reg_name="q",
        connected_only=True,
        seed=None,
    )
    cutqc = CutQC(
        name="%s_%d" % (circuit_type, circuit_size),
        circuit=circuit,
        cutter_constraints={
            "max_subcircuit_width": max_width,
            "max_subcircuit_cuts": 10,
            "subcircuit_size_imbalance": 2,
            "max_cuts": 10,
            "num_subcircuits": [2, 3, 4, 5, 6],
        },
        verbose=verbose,
    )
    
    print ("-- Cut -- ")    
    cutqc.cut()
    if not cutqc.has_solution:
        raise Exception("The input circuit and constraints have no viable cuts")
    print ("-- Done Cutting -- \n")    
    
    print ("-- Evaluate --")
    cutqc.evaluate(eval_mode="sv", num_shots_fn=None)
    print ("-- Done Evaluating -- \n")
    
    print ("-- Dumping CutQC Object into {} --".format (save_path))
    cutqc.save_cutqc_obj (save_path)

