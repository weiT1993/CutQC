from qiskit import QuantumCircuit
from qiskit.circuit.library import CPhaseGate, HGate, TGate, XGate, YGate, ZGate
from qiskit.converters import circuit_to_dag, dag_to_circuit
import random, itertools


class RandomCircuit(object):
    def __init__(self, width, depth, connection_degree, num_hadamards, seed) -> None:
        """
        Generate a random benchmark circuit
        width: number of qubits
        depth: depth of the random circuit
        connection_degree: max number of direct contacts
        num_hadamards: number of H gates in the encoding layer. Overall number of solutions = 2^num_H
        """
        super().__init__()
        random.seed(seed)
        self.width = width
        self.depth = depth
        self.num_hadamards = num_hadamards
        self.num_targets_ubs = []
        for qubit in range(width):
            max_num_targets = self.width - 1 - qubit
            num_targets_ub = int(connection_degree * max_num_targets)
            if qubit < width - 1:
                num_targets_ub = max(num_targets_ub, 1)
            self.num_targets_ubs.append(num_targets_ub)
        # print('num_targets_ubs = {}'.format(self.num_targets_ubs))

    def generate(self):
        entangled_circuit, num_targets = self.generate_entangled()

        """ Encode random solution states """
        encoding_qubits = random.sample(range(self.width), self.num_hadamards)
        quantum_states = [["0"] for qubit in range(self.width)]
        for qubit in encoding_qubits:
            entangled_circuit.append(instruction=HGate(), qargs=[qubit])
            quantum_states[qubit] = ["0", "1"]
        solution_states_strings = itertools.product(*quantum_states)
        solution_states = []
        for binary_state in solution_states_strings:
            binary_state = "".join(binary_state[::-1])
            state = int(binary_state, 2)
            solution_states.append(state)
        # print('%d 2q gates. %d tensor factors. %d depth.'%(
        #     entangled_circuit.num_nonlocal_gates(),
        #     entangled_circuit.num_unitary_factors(),
        #     entangled_circuit.depth()
        #     ))
        # print('num_targets = {}'.format(num_targets))
        return entangled_circuit, solution_states

    def generate_entangled(self):
        left_circuit = QuantumCircuit(self.width, name="q")
        left_dag = circuit_to_dag(left_circuit)

        right_circuit = QuantumCircuit(self.width, name="q")
        right_dag = circuit_to_dag(right_circuit)

        qubit_targets = {qubit: set() for qubit in range(self.width)}
        while True:
            """
            Apply a random two-qubit gate to either left_dag or right_dag
            """
            random_control_qubit_idx = self.get_random_control(qubit_targets)
            random_target_qubit_idx = self.get_random_target(
                random_control_qubit_idx, qubit_targets
            )

            dag_to_apply = random.choice([left_dag, right_dag])
            random_control_qubit = dag_to_apply.qubits[random_control_qubit_idx]
            random_target_qubit = dag_to_apply.qubits[random_target_qubit_idx]
            dag_to_apply.apply_operation_back(
                op=CPhaseGate(theta=0.0),
                qargs=[random_control_qubit, random_target_qubit],
                cargs=[],
            )
            qubit_targets[random_control_qubit_idx].add(random_target_qubit_idx)

            """
            Apply a random 1-q gate to left_dag
            Apply its inverse to right_dag
            """
            single_qubit_gate = random.choice(
                [HGate(), TGate(), XGate(), YGate(), ZGate()]
            )
            random_qubit = left_dag.qubits[random.choice(range(self.width))]
            left_dag.apply_operation_back(
                op=single_qubit_gate, qargs=[random_qubit], cargs=[]
            )
            right_dag.apply_operation_front(
                op=single_qubit_gate.inverse(), qargs=[random_qubit], cargs=[]
            )

            """ Terminate when there is enough depth """
            if left_dag.depth() + right_dag.depth() >= self.depth:
                break
        entangled_dag = left_dag.compose(right_dag, inplace=False)
        entangled_circuit = dag_to_circuit(entangled_dag)
        num_targets = [len(qubit_targets[qubit]) for qubit in range(self.width)]
        for qubit in range(self.width):
            assert num_targets[qubit] <= self.num_targets_ubs[qubit]
        return entangled_circuit, num_targets

    def get_random_control(self, qubit_targets):
        """
        Get a random control qubit
        Prioritize the ones with spare targets
        Else choose from qubits with #targets>0
        """
        candidates = []
        for qubit in qubit_targets:
            if len(qubit_targets[qubit]) < self.num_targets_ubs[qubit]:
                candidates.append(qubit)
        if len(candidates) > 0:
            return random.choice(candidates)
        else:
            candidates = []
            for qubit, num_targets in enumerate(self.num_targets_ubs):
                if num_targets > 0:
                    candidates.append(qubit)
            return random.choice(candidates)

    def get_random_target(self, control_qubit, qubit_targets):
        """
        Get a random target qubit
        If the control qubit has exhausted its #targets, choose from existing targets
        Else prioritize the ones that have not been used
        """
        if len(qubit_targets[control_qubit]) < self.num_targets_ubs[control_qubit]:
            candidates = []
            for qubit in range(control_qubit + 1, self.width):
                if qubit not in qubit_targets[control_qubit]:
                    candidates.append(qubit)
            return random.choice(candidates)
        else:
            return random.choice(list(qubit_targets[control_qubit]))
