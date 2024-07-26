import os, math
import os, logging
import torch
from cutqc.main import CutQC 
from helper_functions.benchmarks import generate_circ

logging.disable(logging.WARNING)
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "1"

def example_func ():
    circuit_type = "bv"
    circuit_size = 24
    max_width = 10
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
            "num_subcircuits": [2, 3, 4, 5, 6, 7],
        },
        verbose=True,
        build_only=False,
        load_data=None,
        parallel_reconstruction=False
    )
    
    print ("-- Cut -- ")    
    cutqc.cut()
    if not cutqc.has_solution:
        raise Exception("The input circuit and constraints have no viable cuts")
    print ("-- Done Cutting -- \n")    
    
    print ("-- Evaluate --")
    cutqc.evaluate(eval_mode="sv", num_shots_fn=None)
    print ("-- Done Evaluating -- \n")

    print ("-- Build --")
    cutqc.build(mem_limit=32, recursion_depth=1)
    print ("-- Done Building -- \n")
    
    cutqc.verify ()
    print("Cut: %d recursions." % (cutqc.num_recursions))
    cutqc.clean_data()

example_func ()