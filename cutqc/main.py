import subprocess, os, logging
from time import perf_counter

from cutqc.helper_fun import check_valid, add_times

from cutqc.cutter import find_cuts

from cutqc.evaluator import run_subcircuit_instances, attribute_shots
from cutqc.post_process_helper import (
    generate_subcircuit_entries,
    generate_compute_graph,
)

from cutqc.dynamic_definition import DynamicDefinition, full_verify


class CutQC:
    """
    The main module for CutQC
    cut --> evaluate results --> verify (optional)
    """

    def __init__(self, circuit, cutter_constraints, verbose):
        """
        Args:
        circuit : the input quantum circuit
        cutter_constraints : cutting constraints to satisfy

        verbose: setting verbose to True to turn on logging information.
        Useful to visualize what happens,
        but may produce very long outputs for complicated circuits.
        """
        check_valid(circuit=circuit)
        self.circuit = circuit
        self.cutter_constraints = cutter_constraints
        self.verbose = verbose
        self.times = {}

    def cut(self) -> None:
        """
        Cut the given circuits
        If use the MIP solver to automatically find cuts, the following are required:
        max_subcircuit_width: max number of qubits in each subcircuit

        The following are optional:
        max_cuts: max total number of cuts allowed
        num_subcircuits: list of subcircuits to try, CutQC returns the best solution found among the trials
        max_subcircuit_cuts: max number of cuts for a subcircuit
        max_subcircuit_size: max number of gates in a subcircuit
        quantum_cost_weight: quantum_cost_weight : MIP overall cost objective is given by
        quantum_cost_weight * num_subcircuit_instances + (1-quantum_cost_weight) * classical_postprocessing_cost

        Else supply the subcircuit_vertices manually
        Note that supplying subcircuit_vertices overrides all other arguments
        """
        if self.verbose:
            logging.info("*" * 20 + "Cut" + "*" * 20)
            logging.info(
                f"width = {self.circuit.num_qubits} depth = {self.circuit.depth()} size = {self.circuit.num_nonlocal_gates()} -->"
            )
            logging.info(self.cutter_constraints)
        cutter_begin = perf_counter()
        self.cut_solution = find_cuts(
            **self.cutter_constraints, circuit=self.circuit, verbose=self.verbose
        )
        if "complete_path_map" in self.cut_solution:
            self.compute_graph, self.subcircuit_entries, self.subcircuit_instances = (
                _generate_metadata(
                    self.cut_solution["counter"],
                    self.cut_solution["subcircuits"],
                    self.cut_solution["complete_path_map"],
                )
            )
            if self.verbose:
                logging.info("--> subcircuit_entries:")
                for subcircuit_idx in self.subcircuit_entries:
                    logging.info(
                        f"Subcircuit_{subcircuit_idx} has {len(self.subcircuit_entries[subcircuit_idx])} entries"
                    )
            self.times["cutter"] = perf_counter() - cutter_begin
        else:
            self.compute_graph, self.subcircuit_entries, self.subcircuit_instances = (
                None,
                None,
                None,
            )
            raise RuntimeError("The input circuit and constraints have no viable cuts")

    def evaluate(self, num_shots_fn):
        """
        num_shots_fn: a function that gives the number of shots to take for a given circuit
        """
        if self.verbose:
            logging.info(
                "*" * 20
                + f"evaluate subcircuits with shots {num_shots_fn is not None}"
                + "*" * 20
            )

        evaluate_begin = perf_counter()
        self.subcircuit_entry_probs = {}
        for subcircuit_index in range(len(self.cut_solution["subcircuits"])):
            subcircuit_measured_probs = run_subcircuit_instances(
                subcircuit=self.cut_solution["subcircuits"][subcircuit_index],
                subcircuit_instance_init_meas=self.subcircuit_instances[
                    subcircuit_index
                ],
                num_shots_fn=num_shots_fn,
            )
            self.subcircuit_entry_probs[subcircuit_index] = attribute_shots(
                subcircuit_measured_probs=subcircuit_measured_probs,
                subcircuit_entries=self.subcircuit_entries[subcircuit_index],
            )
        eval_time = perf_counter() - evaluate_begin
        self.times["evaluate"] = eval_time
        if self.verbose:
            logging.info(f"evaluate took {eval_time} seconds")

    def build(self, mem_limit, recursion_depth):
        """
        mem_limit: memory limit during post process. 2^mem_limit is the largest vector
        """
        if self.verbose:
            logging.info("--> Reconstruct")

        # Keep these times and discard the rest
        self.times = {
            "cutter": self.times["cutter"],
            "evaluate": self.times["evaluate"],
        }

        build_begin = perf_counter()
        dd = DynamicDefinition(
            compute_graph=self.compute_graph,
            num_cuts=self.cut_solution["num_cuts"],
            subcircuit_entry_probs=self.subcircuit_entry_probs,
            mem_limit=mem_limit,
            recursion_depth=recursion_depth,
        )
        dd.build()

        self.times = add_times(times_a=self.times, times_b=dd.times)
        self.approximation_bins = dd.dd_bins
        self.num_recursions = len(self.approximation_bins)
        self.times["build"] = perf_counter() - build_begin
        self.times["build"] += self.times["cutter"]
        self.times["build"] -= self.times["merge_states_into_bins"]

    def verify(self):
        verify_begin = perf_counter()
        reconstructed_prob, self.approximation_error = full_verify(
            full_circuit=self.circuit,
            complete_path_map=self.cut_solution["complete_path_map"],
            subcircuits=self.cut_solution["subcircuits"],
            dd_bins=self.approximation_bins,
        )
        verify_time = perf_counter() - verify_begin
        logging.info(f"verify took {verify_time}. error = {self.approximation_error}.")


def _generate_metadata(counter, subcircuits, complete_path_map):
    compute_graph = generate_compute_graph(
        counter=counter,
        subcircuits=subcircuits,
        complete_path_map=complete_path_map,
    )

    (
        subcircuit_entries,
        subcircuit_instances,
    ) = generate_subcircuit_entries(compute_graph=compute_graph)
    return compute_graph, subcircuit_entries, subcircuit_instances
