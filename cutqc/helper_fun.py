from os import times
from qiskit.converters import circuit_to_dag


def check_valid(circuit):
    """
    If the input circuit is not fully connected, it does not need CutQC to be split into smaller circuits.
    CutQC hence only cuts a circuit if it is fully connected.
    Furthermore, CutQC only supports 2-qubit gates.
    """
    if circuit.num_unitary_factors() != 1:
        raise ValueError(
            "Input circuit is not fully connected thus does not need cutting. Number of unitary factors = %d"
            % circuit.num_unitary_factors()
        )
    if circuit.num_clbits > 0:
        raise ValueError("Please remove classical bits from the circuit before cutting")
    dag = circuit_to_dag(circuit)
    for op_node in dag.topological_op_nodes():
        if len(op_node.qargs) > 2:
            raise ValueError("CutQC currently does not support >2-qubit gates")
        if op_node.op.name == "barrier":
            raise ValueError("Please remove barriers from the circuit before cutting")


def add_times(times_a, times_b):
    """
    Add the two time breakdowns
    """
    for field in times_b:
        if field in times_a:
            times_a[field] += times_b[field]
        else:
            times_a[field] = times_b[field]
    return times_a
