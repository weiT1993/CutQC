from time import perf_counter
import os, math, argparse, subprocess, pickle
import os, logging
logging.disable(logging.WARNING)
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '1'
# Comment this line if using GPU
os.environ['CUDA_VISIBLE_DEVICES'] = '-1'

# from cutqc_runtime.main import CutQC
from cutqc.main import CutQC

from qiskit_helper_functions.non_ibmq_functions import generate_circ

if __name__ == '__main__':
    circuit_type = 'supremacy'
    circuit_size = 16
    circuit = generate_circ(num_qubits=circuit_size,depth=8,circuit_type=circuit_type)
    cutqc = CutQC(name='%s_%d'%(circuit_type,circuit_size),
    circuit=circuit,
    cutter_constraints={
        'max_subcircuit_width':math.ceil(circuit.num_qubits/4*3),
        'max_subcircuit_cuts':10,
        'max_subcircuit_size':circuit.num_nonlocal_gates(),
        'max_cuts':10,
        'num_subcircuits':[2,3]
        },verbose=False)
    cutqc.cut()
    if not cutqc.has_solution:
        raise Exception('The input circuit and constraints have no viable cuts')

    cutqc.evaluate(eval_mode='sv', num_shots_fn=None)
    cutqc.build(mem_limit=32,recursion_depth=1)
    print('Cut: %d recursions.'%(cutqc.num_recursions))
    print(cutqc.approximation_bins)
    cutqc.clean_data()