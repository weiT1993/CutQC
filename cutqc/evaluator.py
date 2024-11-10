import itertools, copy, pickle, subprocess, psutil, os, logging
import numpy as np
from qiskit.converters import circuit_to_dag, dag_to_circuit
from qiskit.circuit.library.standard_gates import HGate, SGate, SdgGate, XGate

from helper_functions.non_ibmq_functions import (
    find_process_jobs,
    scrambled,
    evaluate_circ,
)


def get_num_workers(num_jobs, ram_required_per_worker):
    ram_avail = psutil.virtual_memory().available / 1024**3
    ram_avail = ram_avail / 4 * 3
    num_cpus = int(os.cpu_count() / 4 * 3)
    num_workers = int(min(ram_avail / ram_required_per_worker, num_jobs, num_cpus))
    return num_workers


def run_subcircuit_instances(subcircuit, subcircuit_instance_init_meas, num_shots_fn):
    """
    subcircuit_instance_probs[(instance_init,instance_meas)] = measured probability
    """
    subcircuit_measured_probs = {}
    num_shots = num_shots_fn(subcircuit) if num_shots_fn is not None else None
    num_workers = get_num_workers(
        num_jobs=len(subcircuit_instance_init_meas),
        ram_required_per_worker=2**subcircuit.num_qubits * 4 / 1e9,
    )
    for instance_init_meas in subcircuit_instance_init_meas:
        if "Z" in instance_init_meas[1]:
            continue
        subcircuit_instance = modify_subcircuit_instance(
            subcircuit=subcircuit,
            init=instance_init_meas[0],
            meas=instance_init_meas[1],
        )
        if num_shots is None:
            subcircuit_inst_prob = evaluate_circ(
                circuit=subcircuit_instance, backend="statevector_simulator"
            )
        else:
            subcircuit_inst_prob = evaluate_circ(
                circuit=subcircuit_instance,
                backend="noiseless_qasm_simulator",
                options={"num_shots": num_shots},
            )
        mutated_meas = mutate_measurement_basis(meas=instance_init_meas[1])
        for meas in mutated_meas:
            measured_prob = measure_prob(
                unmeasured_prob=subcircuit_inst_prob, meas=meas
            )
            subcircuit_measured_probs[(instance_init_meas[0], meas)] = measured_prob
    return subcircuit_measured_probs


def mutate_measurement_basis(meas):
    """
    I and Z measurement basis correspond to the same logical circuit
    """
    if all(x != "I" for x in meas):
        return [meas]
    else:
        mutated_meas = []
        for x in meas:
            if x != "I":
                mutated_meas.append([x])
            else:
                mutated_meas.append(["I", "Z"])
        mutated_meas = list(itertools.product(*mutated_meas))
        return mutated_meas


def modify_subcircuit_instance(subcircuit, init, meas):
    """
    Modify the different init, meas for a given subcircuit
    Returns:
    Modified subcircuit_instance
    List of mutated measurements
    """
    subcircuit_dag = circuit_to_dag(subcircuit)
    subcircuit_instance_dag = copy.deepcopy(subcircuit_dag)
    for i, x in enumerate(init):
        q = subcircuit.qubits[i]
        if x == "zero":
            continue
        elif x == "one":
            subcircuit_instance_dag.apply_operation_front(
                op=XGate(), qargs=[q], cargs=[]
            )
        elif x == "plus":
            subcircuit_instance_dag.apply_operation_front(
                op=HGate(), qargs=[q], cargs=[]
            )
        elif x == "minus":
            subcircuit_instance_dag.apply_operation_front(
                op=HGate(), qargs=[q], cargs=[]
            )
            subcircuit_instance_dag.apply_operation_front(
                op=XGate(), qargs=[q], cargs=[]
            )
        elif x == "plusI":
            subcircuit_instance_dag.apply_operation_front(
                op=SGate(), qargs=[q], cargs=[]
            )
            subcircuit_instance_dag.apply_operation_front(
                op=HGate(), qargs=[q], cargs=[]
            )
        elif x == "minusI":
            subcircuit_instance_dag.apply_operation_front(
                op=SGate(), qargs=[q], cargs=[]
            )
            subcircuit_instance_dag.apply_operation_front(
                op=HGate(), qargs=[q], cargs=[]
            )
            subcircuit_instance_dag.apply_operation_front(
                op=XGate(), qargs=[q], cargs=[]
            )
        else:
            raise Exception("Illegal initialization :", x)
    for i, x in enumerate(meas):
        q = subcircuit.qubits[i]
        if x == "I" or x == "comp":
            continue
        elif x == "X":
            subcircuit_instance_dag.apply_operation_back(
                op=HGate(), qargs=[q], cargs=[]
            )
        elif x == "Y":
            subcircuit_instance_dag.apply_operation_back(
                op=SdgGate(), qargs=[q], cargs=[]
            )
            subcircuit_instance_dag.apply_operation_back(
                op=HGate(), qargs=[q], cargs=[]
            )
        else:
            raise Exception("Illegal measurement basis:", x)
    subcircuit_instance_circuit = dag_to_circuit(subcircuit_instance_dag)
    return subcircuit_instance_circuit


def measure_prob(unmeasured_prob, meas):
    if meas.count("comp") == len(meas) or type(unmeasured_prob) is float:
        return unmeasured_prob
    else:
        measured_prob = np.zeros(int(2 ** meas.count("comp")))
        # print('Measuring in',meas)
        for full_state, p in enumerate(unmeasured_prob):
            sigma, effective_state = measure_state(full_state=full_state, meas=meas)
            measured_prob[effective_state] += sigma * p
        return measured_prob


def measure_state(full_state, meas):
    """
    Compute the corresponding effective_state for the given full_state
    Measured in basis `meas`
    Returns sigma (int), effective_state (int)
    where sigma = +-1
    """
    bin_full_state = bin(full_state)[2:].zfill(len(meas))
    sigma = 1
    bin_effective_state = ""
    for meas_bit, meas_basis in zip(bin_full_state, meas[::-1]):
        if meas_bit == "1" and meas_basis != "I" and meas_basis != "comp":
            sigma *= -1
        if meas_basis == "comp":
            bin_effective_state += meas_bit
    effective_state = int(bin_effective_state, 2) if bin_effective_state != "" else 0
    # print('bin_full_state = %s --> %d * %s (%d)'%(bin_full_state,sigma,bin_effective_state,effective_state))
    return sigma, effective_state


def attribute_shots(subcircuit_measured_probs, subcircuit_entries):
    """
    Attribute the subcircuit_instance shots into respective subcircuit entries
    subcircuit_entry_probs[entry_init, entry_meas] = entry_prob
    """
    # num_workers = get_num_workers(
    #     num_jobs=len(jobs),
    #     ram_required_per_worker=2 ** subcircuit.num_qubits
    #     * 4
    #     / 1e9,
    # )
    subcircuit_entry_probs = {}
    for subcircuit_entry_init_meas in subcircuit_entries:
        subcircuit_entry_term = subcircuit_entries[subcircuit_entry_init_meas]
        subcircuit_entry_prob = None
        for term in subcircuit_entry_term:
            coefficient, subcircuit_instance_init_meas = term
            subcircuit_instance_prob = subcircuit_measured_probs[
                subcircuit_instance_init_meas
            ]
            if subcircuit_entry_prob is None:
                subcircuit_entry_prob = coefficient * subcircuit_instance_prob
            else:
                subcircuit_entry_prob += coefficient * subcircuit_instance_prob
        subcircuit_entry_probs[subcircuit_entry_init_meas] = subcircuit_entry_prob
    return subcircuit_entry_probs
