from qiskit_helper_functions.non_ibmq_functions import generate_circ

from cutqc.main import CutQC

if __name__ == '__main__':
    # Generate some test circuits
    circuits = {}
    max_subcircuit_qubit = 3
    for full_circ_size in [4]:
        circuit = generate_circ(full_circ_size=full_circ_size, circuit_type='supremacy')
        circuit_name = 'bv_%d'%full_circ_size
        if circuit.num_qubits==0:
            continue
        else:
            circuits[circuit_name] = {
                'circuit':circuit,'max_subcircuit_qubit':max_subcircuit_qubit,'max_cuts':30,'num_subcircuits':[2,3,4,5]}

    # Use CutQC package to evaluate the circuits
    num_nodes = 1
    num_threads = 1
    qubit_limit = 24
    eval_mode = 'sv'
    cutqc = CutQC(verbose=True)
    cutqc.cut(circuits=circuits)
    reconstructed_probs = cutqc.evaluate(circuits=circuits,eval_mode=eval_mode,qubit_limit=qubit_limit,num_nodes=num_nodes,num_threads=num_threads,ibmq=None)
    errors = cutqc.verify(circuits=circuits,num_nodes=num_nodes,num_threads=num_threads,qubit_limit=qubit_limit,eval_mode=eval_mode)