import os, math, logging
from cutqc.main import CutQC
from helper_functions.benchmarks import generate_circ

if __name__ == "__main__":
    logging.basicConfig(
        level=logging.DEBUG,
        format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
        filename="example.log",
        filemode="w",
    )
    circuit_type = "supremacy"
    circuit_size = 16
    circuit = generate_circ(
        num_qubits=circuit_size,
        depth=1,
        circuit_type=circuit_type,
        reg_name="q",
        connected_only=True,
        seed=None,
    )
    cutqc = CutQC(
        circuit=circuit,
        cutter_constraints={
            "max_subcircuit_width": math.ceil(circuit.num_qubits / 4 * 3),
            "max_subcircuit_cuts": 10,
            "subcircuit_size_imbalance": 2,
            "max_cuts": 10,
            "num_subcircuits": [2, 3],
        },
        verbose=True,
    )
    cutqc.cut()
    cutqc.evaluate(num_shots_fn=None)
    cutqc.build(mem_limit=32, recursion_depth=1)
    cutqc.verify()
    logging.info(f"Cut: {cutqc.num_recursions} recursions.")
    logging.info(cutqc.approximation_bins)
