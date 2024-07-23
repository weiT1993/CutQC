# Description: Driver Cuts and Evals a circuit. The result is saved as pickelfile
import os
import math
import logging
import argparse

logging.disable(logging.WARNING)
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "1"
os.environ["CUDA_VISIBLE_DEVICES"] = "-1" 

from cutqc.main import CutQC 
from helper_functions.benchmarks import generate_circ



def main(circuit_size, max_subcircuit_width, circuit_type):
    circuit_type = circuit_type
    circuit_size = circuit_size
    max_subcircuit_width = max_subcircuit_width
    verbose = False
    
    filename = "{}_{}_{}.pkl".format (circuit_type, circuit_size, max_subcircuit_width)
    
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
            "max_subcircuit_width": max_subcircuit_width,
            "max_subcircuit_cuts": 10,
            "subcircuit_size_imbalance": 2,
            "max_cuts": 10,
            "num_subcircuits": [2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20],
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
    
    print ("-- Dumping CutQC Object into {} --".format (filename))
    cutqc.save_cutqc_obj (filename)



if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run CutQC with given parameters")
    parser.add_argument('--circuit_size', type=int, required=True, help='Size of the circuit')
    parser.add_argument('--max_subcircuit_width', type=int, required=True, help='Max width of subcircuit')
    parser.add_argument('--circuit_type', type=str, required=True, help='Circuit Type')
    args = parser.parse_args()
    
    main(args.circuit_size, args.max_subcircuit_width, args.circuit_type)
