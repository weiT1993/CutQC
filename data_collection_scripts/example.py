import os
import math
import logging
import argparse

logging.disable(logging.WARNING)
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "1"
# Comment this line if using GPU
os.environ["CUDA_VISIBLE_DEVICES"] = "-1"

# from cutqc_runtime.main import CutQC # Use this just to benchmark the runtime
from cutqc.main import CutQC  # Use this for exact computation
# from cutqc_runtime.main import CutQC # Use this for exact computation
from helper_functions.benchmarks import generate_circ

def main(circuit_size, max_subcircuit_width, circuit_type):
    circuit_type = circuit_type
    circuit = generate_circ(
        num_qubits=circuit_size,
        depth=1,
        circuit_type=circuit_type,
        reg_name="q",
        connected_only=True,
        seed=None,
    )
    cutqc = CutQC(
        name="%s_%d_%d" % (circuit_type, max_subcircuit_width, circuit_size),
        circuit=circuit,
        cutter_constraints={
            "max_subcircuit_width": max_subcircuit_width,
            # "max_subcircuit_width": math.ceil(circuit.num_qubits / 4 * 3),
            "max_subcircuit_cuts": 10,
            "subcircuit_size_imbalance": 2,
            "max_cuts": 10,
            "num_subcircuits": [2, 3, 4, 5, 6, 7, 8],
        },
        verbose=True,
    )
    
    print("-- Cut --")
    cutqc.cut()
    if not cutqc.has_solution:
        raise Exception("The input circuit and constraints have no viable cuts")
    print("-- Done Cutting -- \n")
    
    print("-- Evaluate --")
    cutqc.evaluate(eval_mode="sv", num_shots_fn=None)
    print("-- Done Evaluating -- \n")

    print("-- Build --")
    cutqc.build(mem_limit=128, recursion_depth=1)
    print("-- Done Building -- \n")
    
    # cutqc.verify()
    # print("Cut: %d recursions." % (cutqc.num_recursions))
    # print(cutqc.approximation_bins)
    cutqc.clean_data()

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run CutQC with given parameters")
    parser.add_argument('--circuit_size', type=int, required=True, help='Size of the circuit')
    parser.add_argument('--max_subcircuit_width', type=int, required=True, help='Max width of subcircuit')
    parser.add_argument('--circuit_type', type=str, required=True, help='Circuit Type')
    args = parser.parse_args()
    
    main(args.circuit_size, args.max_subcircuit_width, args.circuit_type)