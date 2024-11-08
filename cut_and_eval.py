# Title: cut_and_eval.py
# Description: Example of how to cut and evaluate for the purposes of 
# distributed reconstruction

import os, logging

logging.disable(logging.WARNING)
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "1"
os.environ["CUDA_VISIBLE_DEVICES"] = "-1" 

from cutqc.main import CutQC 
from helper_functions.benchmarks import generate_circ


if __name__ == "__main__":
    filename = "adder_example.pkl"
    circ_type= 'adder'
    circ_size=10
    max_width=10
    
    # Generate Example Circuit and Initialize CutQC
    circuit = generate_circ(
        num_qubits=circ_size,
        depth=1,
        circuit_type=circ_type,
        reg_name="q",
        connected_only=True,
        seed=None,
    )
    
    cutqc = CutQC(
        name="%s_%d" % (circ_type, circ_size),
        circuit=circuit,
        cutter_constraints={
            "max_subcircuit_width": max_width,
            "max_subcircuit_cuts": 10,
            "subcircuit_size_imbalance": 2,
            "max_cuts": 10,
            "num_subcircuits": [2, 3, 4, 5, 6, 8],
        },
    )
    
    print ("--- Cut --- ")    
    cutqc.cut()
    
    if not cutqc.has_solution:
        raise Exception("The input circuit and constraints have no viable cuts")
    
    print ("--- Evaluate ---")
    cutqc.evaluate(eval_mode="sv", num_shots_fn=None)
    
    print ("--- Dumping CutQC Object into {} ---".format (filename))
    cutqc.save_cutqc_obj (filename)

    print ("Completed")

