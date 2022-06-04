import subprocess, os
from time import perf_counter

from cutqc.helper_fun import check_valid, add_times
from cutqc.cutter import find_cuts
from cutqc.post_process_helper import (
    generate_subcircuit_entries,
    generate_compute_graph,
)
from cutqc_runtime.dynamic_definition import DynamicDefinition


class CutQC:
    """
    The main module for CutQC
    cut --> evaluate results --> verify (optional)
    """

    def __init__(self, name, circuit, cutter_constraints, verbose):
        """
        Args:
        name : name of the input quantum circuit
        circuit : the input quantum circuit
        cutter_constraints : cutting constraints to satisfy

        verbose: setting verbose to True to turn on logging information.
        Useful to visualize what happens,
        but may produce very long outputs for complicated circuits.
        """
        check_valid(circuit=circuit)
        self.name = name
        self.circuit = circuit
        self.cutter_constraints = cutter_constraints
        self.verbose = verbose
        self.times = {}
        self.tmp_data_folder = "cutqc/tmp_data"
        if os.path.exists(self.tmp_data_folder):
            subprocess.run(["rm", "-r", self.tmp_data_folder])
        os.makedirs(self.tmp_data_folder)

    def cut(self):
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
            print("*" * 20, "Cut %s" % self.name, "*" * 20)
            print(
                "width = %d depth = %d size = %d -->"
                % (
                    self.circuit.num_qubits,
                    self.circuit.depth(),
                    self.circuit.num_nonlocal_gates(),
                )
            )
            print(self.cutter_constraints)
        cutter_begin = perf_counter()
        cut_solution = find_cuts(
            **self.cutter_constraints, circuit=self.circuit, verbose=self.verbose
        )
        for field in cut_solution:
            self.__setattr__(field, cut_solution[field])
        if "complete_path_map" in cut_solution:
            self.has_solution = True
            self._generate_metadata()
        else:
            self.has_solution = False
        self.times["cutter"] = perf_counter() - cutter_begin

    def evaluate(self, eval_mode, num_shots_fn):
        pass

    def build(self, mem_limit, recursion_depth):
        """
        mem_limit: memory limit during post process. 2^mem_limit is the largest vector
        """
        if self.verbose:
            print("--> Build %s" % (self.name))

        # Keep these times and discard the rest
        self.times = {"cutter": self.times["cutter"]}

        dd = DynamicDefinition(
            compute_graph=self.compute_graph,
            data_folder=self.tmp_data_folder,
            num_cuts=self.num_cuts,
            mem_limit=mem_limit,
            recursion_depth=recursion_depth,
        )
        dd.build()

        self.times = add_times(times_a=self.times, times_b=dd.times)
        self.approximation_bins = dd.dd_bins
        self.num_recursions = len(self.approximation_bins)
        self.overhead = dd.overhead
        self.times["build"] = sum([self.times[field] for field in self.times])

        if self.verbose:
            print("Overhead = {}".format(self.overhead))

    def clean_data(self):
        subprocess.run(["rm", "-r", self.tmp_data_folder])

    def _generate_metadata(self):
        self.compute_graph = generate_compute_graph(
            counter=self.counter,
            subcircuits=self.subcircuits,
            complete_path_map=self.complete_path_map,
        )
