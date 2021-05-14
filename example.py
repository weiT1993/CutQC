from qiskit.circuit import QuantumCircuit

from qiskit_helper_functions.non_ibmq_functions import generate_circ

from cutqc.main import CutQC

if __name__ == '__main__':
    # Generate a circuit from qasm file
    circuits = {}
    # circuit = QuantumCircuit().from_qasm_file('circuit.qasm')
    # circuit_name = 'test_%d'%circuit.num_qubits
    # circuits[circuit_name] = {
    #     'circuit':circuit,'max_subcircuit_qubit':8,'max_cuts':20,'num_subcircuits':[3]}
    
    # Generate a circuit from the provided circuit generator
    circuit = generate_circ(full_circ_size=4, circuit_type='supremacy')
    circuit_name = 'supremacy_%d'%circuit.num_qubits
    circuits[circuit_name] = {
        'circuit':circuit,'max_subcircuit_qubit':3,'max_cuts':10,'num_subcircuits':[2,3]}

    # Use CutQC package to evaluate the circuits
    num_nodes = 1
    num_threads = 1
    qubit_limit = 24
    eval_mode = 'sv'
    cutqc = CutQC(verbose=True)
    cutqc.cut(circuits=circuits)
    reconstructed_probs = cutqc.evaluate(circuits=circuits,eval_mode=eval_mode,qubit_limit=qubit_limit,num_nodes=num_nodes,num_threads=num_threads,ibmq=None)
    errors = cutqc.verify(circuits=circuits,num_nodes=num_nodes,num_threads=num_threads,qubit_limit=qubit_limit,eval_mode=eval_mode)