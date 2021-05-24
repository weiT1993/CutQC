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
    # Make a circuit
    circuit = make_QV()
    
    cutqc = CutQC(circuit_name='QV_%d'%circuit.num_qubits,circuit=circuit,verbose=False)
    
    # Option 1: automatic MIP solver
    # source_folder = cutqc.cut(max_subcircuit_qubit=3, max_cuts=10, num_subcircuits=[2,3])
    # Option 2: manually specify subcircuit partitions
    source_folder = cutqc.cut(subcircuit_vertices=[range(26),[26,27,28],[29,30,31]])
    
    # Evaluate and verify CutQC results
    dest_folders = cutqc.evaluate(source_folders=[source_folder],eval_mode='ibmq_paris',mem_limit=24,num_nodes=1,num_threads=1,ibmq=None)
    cutqc.verify(source_folders=[source_folder],dest_folders=dest_folders)