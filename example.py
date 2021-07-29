import os, subprocess, pickle

from qiskit.circuit import QuantumCircuit
from qiskit.quantum_info import random_unitary

from qiskit_helper_functions.non_ibmq_functions import generate_circ

from cutqc.main import CutQC

def make_QV():
    def add_su4(circ, seed, qubits):
        su4 = random_unitary(4, seed=seed).to_instruction()
        su4.label = 'su4_' + str(seed)
        circ.append(su4, qubits)
    largest = QuantumCircuit(8)
    add_su4(largest, 542, [4, 0])
    add_su4(largest, 996, [5, 1])
    add_su4(largest, 402, [6, 3])
    add_su4(largest, 552, [2, 7])
    add_su4(largest, 242, [7, 4])
    add_su4(largest, 212, [6, 3])
    add_su4(largest, 910, [0, 1])
    add_su4(largest, 573, [5, 2])
    add_su4(largest, 48, [2, 1])
    add_su4(largest, 906, [5, 0])
    add_su4(largest, 663, [3, 7])
    add_su4(largest, 193, [4, 6])
    add_su4(largest, 430, [0, 7])
    add_su4(largest, 630, [1, 4])
    add_su4(largest, 167, [5, 3])
    add_su4(largest, 67, [6, 2])
    add_su4(largest, 473, [4, 3])
    add_su4(largest, 121, [5, 0])
    add_su4(largest, 854, [1, 6])
    add_su4(largest, 834, [7, 2])
    add_su4(largest, 529, [2, 1])
    add_su4(largest, 351, [3, 5])
    add_su4(largest, 376, [6, 0])
    add_su4(largest, 857, [7, 4])
    add_su4(largest, 139, [6, 4])
    add_su4(largest, 537, [7, 0])
    add_su4(largest, 338, [1, 3])
    add_su4(largest, 358, [2, 5])
    add_su4(largest, 843, [0, 1])
    add_su4(largest, 100, [3, 6])
    add_su4(largest, 911, [4, 2])
    add_su4(largest, 172, [7, 5])
    return largest

if __name__ == '__main__':
    # Make circuit(s)
    task_1 = {
        'name':'largest_QV',
        'circuit':make_QV(),
        'kwargs':dict(
            max_subcircuit_width=8,
            max_subcircuit_cuts=6,
            max_subcircuit_size=26,
            quantum_cost_weight=1.0,
            max_cuts=10,
            num_subcircuits=[2,3]
        )
    } # Option 1: automatic MIP solver
    task_2 = {
        'name':'largest_QV_manual_cuts',
        'circuit':make_QV(),
        'kwargs':dict(
            subcircuit_vertices=[range(26),[26,27,28],[29,30,31]]
        )
    } # Option 2: manually specify subcircuit partitions
    task_3 = {
        'name':'supremacy',
        'circuit':generate_circ(full_circ_size=20,circuit_type='supremacy'),
        'kwargs':dict(
            max_subcircuit_width=12,
            max_subcircuit_cuts=10,
            max_subcircuit_size=None,
            quantum_cost_weight=1.0,
            max_cuts=10,
            num_subcircuits=[2,3,4]
        )
    }
    task_4 = {
        'name':'BV_manual',
        'circuit':generate_circ(full_circ_size=8,circuit_type='bv'),
        'kwargs':dict(
            subcircuit_vertices=[[0,1,2,3],[4,5,6]]
        )
    }
    task_5 = {
        'name':'BV',
        'circuit':generate_circ(full_circ_size=6,circuit_type='bv'),
        'kwargs':dict(
            max_subcircuit_width=4,
            max_subcircuit_cuts=10,
            max_subcircuit_size=50,
            quantum_cost_weight=1.0,
            max_cuts=10,
            num_subcircuits=[2,3,4]
        )
    }
    
    # Call CutQC
    cutqc = CutQC(tasks=[task_5],verbose=True)
    cutqc.cut()
    def constant_shots_fn(circuit):
        return 1024
    cutqc.evaluate(eval_mode='sv',num_shots_fn=constant_shots_fn,mem_limit=24,num_threads=4)
    cutqc.verify()