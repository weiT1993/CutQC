import subprocess, os
import pickle
from time import perf_counter

from cutqc.helper_fun import check_valid, add_times
from cutqc.cutter import find_cuts
from cutqc.evaluator import run_subcircuit_instances, attribute_shots
from cutqc.post_process_helper import (
    generate_subcircuit_entries,
    generate_compute_graph,
)
from cutqc.dynamic_definition import DynamicDefinition, full_verify

import torch.distributed as dist
__host_machine__ = 0

class CutQC:
    """
    The main module for CutQC
    cut --> evaluate results --> verify (optional)
    """
    
    def __init__(self, name=None, circuit=None, cutter_constraints=None, verbose=False, 
                 build_only=False, load_data=None, parallel_reconstruction=False, local_rank=None):
        """
        Args:
        name: name of the input quantum circuit
        circuit: the input quantum circuit
        cutter_constraints: cutting constraints to satisfy
        build_only: Specifies building and no cutting or evaluating should occur.
        load_data (Optional): String of file name to load subcircuit outputs 
                              from a previous CutQC instance. Default is None.
        parallel_reconstruction (Optional): When set to 'True', reconstruction is executed distributed. Default is false.
        verbose: setting verbose to True to turn on logging information.
                 Useful to visualize what happens,
                 but may produce very long outputs for complicated circuits.
        """
        self.name = name
        self.circuit = circuit
        self.cutter_constraints = cutter_constraints
        self.local_rank = local_rank
        self.verbose = verbose
        self.times = {}
        
        self.compute_graph = None
        self.tmp_data_folder = None
        self.num_cuts = None
        self.complete_path_map = None
        self.subcircuits = None

        if build_only:
            # Only host should load in distributed mode
            if parallel_reconstruction and dist.get_rank() == __host_machine__:
                self._load_data(load_data)
            if not parallel_reconstruction:
                self._load_data(load_data)
        elif not build_only:
            self._initialize_for_serial_reconstruction(circuit)    
        
        self.parallel_reconstruction = parallel_reconstruction
        self.local_rank = local_rank
        
        
    
    def _load_data(self, load_data):
        with open(load_data, 'rb') as inp:
            loaded_cutqc = pickle.load(inp)
            self.__dict__.update(vars(loaded_cutqc))

    def _initialize_for_serial_reconstruction(self, circuit):
        check_valid(circuit=circuit)
        self.tmp_data_folder = "cutqc/tmp_data"
        self._setup_tmp_folder()

    def _setup_tmp_folder(self):
        if os.path.exists(self.tmp_data_folder):
            subprocess.run(["rm", "-r", self.tmp_data_folder])
        os.makedirs(self.tmp_data_folder)
    
    def destroy_distributed (self):
        self.dd.graph_contractor.terminate_distributed_process()

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
        """
        eval_mode = qasm: simulate shots
        eval_mode = sv: statevector simulation
        num_shots_fn: a function that gives the number of shots to take for a given circuit
        """
        if self.verbose:
            print("*" * 20, "evaluation mode = %s" % (eval_mode), "*" * 20)
        self.eval_mode = eval_mode
        self.num_shots_fn = num_shots_fn

        evaluate_begin = perf_counter()
        self._run_subcircuits()
        self._attribute_shots()
        self.times["evaluate"] = perf_counter() - evaluate_begin
        if self.verbose:
            print("evaluate took %e seconds" % self.times["evaluate"])

    def build(self, mem_limit, recursion_depth):
        """
        mem_limit: memory limit during post process. 2^mem_limit is the largest vector
        """
        if self.verbose:
            print("--> Build %s" % (self.name))

        # Keep these times and discard the rest
        # self.times = {
        #     "cutter": self.times["cutter"],
        #     "evaluate": self.times["evaluate"],
        # }
        
    
        print ("self.parallel_reconstruction: {}".format (self.parallel_reconstruction))
        self.dd = DynamicDefinition(
            compute_graph=self.compute_graph,
            data_folder=self.tmp_data_folder,
            num_cuts=self.num_cuts,
            mem_limit=mem_limit,
            recursion_depth=recursion_depth,
            parallel_reconstruction=self.parallel_reconstruction,
            local_rank=self.local_rank
        )
        self.dd.build ()

        self.times = add_times(times_a=self.times, times_b=self.dd.times)
        self.approximation_bins = self.dd.dd_bins
        self.num_recursions = len(self.approximation_bins)
        self.overhead = self.dd.overhead
        # self.times["build"] = perf_counter() - build_begin
        # self.times["build"] += self.times["cutter"]
        # self.times["build"] -= self.times["merge_states_into_bins"]

        if self.verbose:
            print("Overhead = {}".format(self.overhead))

        return self.dd.graph_contractor.times["compute"]

    def save_eval_data (self, foldername: str) -> None:
        '''
        Saves subcircuit evaluation data which can be used in a future 
        instance of `cutqc` for reconstruction.
        '''
        subprocess.run(["cp", "-r", self.tmp_data_folder, foldername])
    
    def verify(self):
        verify_begin = perf_counter()
        reconstructed_prob, self.approximation_error = full_verify(
            full_circuit=self.circuit,
            complete_path_map=self.complete_path_map,
            subcircuits=self.subcircuits,
            dd_bins=self.approximation_bins,
        )
        
        print (f"Approximate Error: {self.approximation_error}")
        print("verify took %.3f" % (perf_counter() - verify_begin))
        return self.approximation_error

    def save_cutqc_obj (self, filename : str) -> None:
        '''
        Saves CutQC instance as the pickle file 'FILENAME'
        '''
        with open (filename, 'wb') as outp:
            pickle.dump(self, outp, pickle.HIGHEST_PROTOCOL)
    
    def clean_data(self):
        subprocess.run(["rm", "-r", self.tmp_data_folder])

    def _generate_metadata(self):
        self.compute_graph = generate_compute_graph(
            counter=self.counter,
            subcircuits=self.subcircuits,
            complete_path_map=self.complete_path_map,
        )

        (
            self.subcircuit_entries,
            self.subcircuit_instances,
        ) = generate_subcircuit_entries(compute_graph=self.compute_graph)
        if self.verbose:
            print("--> %s subcircuit_entries:" % self.name)
            for subcircuit_idx in self.subcircuit_entries:
                print(
                    "Subcircuit_%d has %d entries"
                    % (subcircuit_idx, len(self.subcircuit_entries[subcircuit_idx]))
                )

    def _run_subcircuits(self):
        """
        Run all the subcircuit instances
        subcircuit_instance_probs[subcircuit_idx][(init,meas)] = measured prob
        """
        if self.verbose:
            print("--> Running Subcircuits %s" % self.name)
        if os.path.exists(self.tmp_data_folder):
            subprocess.run(["rm", "-r", self.tmp_data_folder])
        os.makedirs(self.tmp_data_folder)
        
        run_subcircuit_instances (
            subcircuits=self.subcircuits,
            subcircuit_instances=self.subcircuit_instances,
            eval_mode=self.eval_mode,
            num_shots_fn=self.num_shots_fn,
            data_folder=self.tmp_data_folder,
        )

    def _attribute_shots(self):
        """
        Attribute the subcircuit_instance shots into respective subcircuit entries
        subcircuit_entry_probs[subcircuit_idx][entry_init, entry_meas] = entry_prob
        """
        if self.verbose:
            print("--> Attribute shots %s" % self.name)
        attribute_shots(
            subcircuit_entries=self.subcircuit_entries,
            subcircuits=self.subcircuits,
            eval_mode=self.eval_mode,
            data_folder=self.tmp_data_folder,
        )
        subprocess.call(
            "rm %s/subcircuit*instance*.pckl" % self.tmp_data_folder, shell=True
        )
