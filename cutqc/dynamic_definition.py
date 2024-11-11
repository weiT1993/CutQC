import itertools, copy, pickle, subprocess, logging
from time import perf_counter
import numpy as np

from helper_functions.non_ibmq_functions import evaluate_circ
from helper_functions.conversions import quasi_to_real
from helper_functions.metrics import MSE

from cutqc.evaluator import get_num_workers

from cutqc.graph_contraction import GraphContractor
from cutqc.helper_fun import add_times
from cutqc.post_process_helper import get_reconstruction_qubit_order


class DynamicDefinition(object):
    def __init__(
        self,
        compute_graph,
        num_cuts,
        subcircuit_entry_probs,
        mem_limit,
        recursion_depth,
    ) -> None:
        super().__init__()
        self.compute_graph = compute_graph
        self.num_cuts = num_cuts
        self.subcircuit_entry_probs = subcircuit_entry_probs
        self.mem_limit = mem_limit
        self.recursion_depth = recursion_depth
        self.dd_bins = {}
        self.times = {
            "get_dd_schedule": 0.0,
            "merge_states_into_bins": 0.0,
            "sort": 0.0,
        }

    def build(self):
        """
        Returns

        dd_bins[recursion_layer] =  {'subcircuit_state','upper_bin'}
        subcircuit_state[subcircuit_idx] = ['0','1','active','merged']
        """
        num_qubits = sum(
            [
                self.compute_graph.nodes[subcircuit_idx]["effective"]
                for subcircuit_idx in self.compute_graph.nodes
            ]
        )
        largest_bins = []  # [{recursion_layer, bin_id}]
        recursion_layer = 0
        while recursion_layer < self.recursion_depth:
            logging.info("-" * 10 + f"Recursion Layer {recursion_layer}" + "-" * 10)
            """Get qubit states"""
            get_dd_schedule_begin = perf_counter()
            if recursion_layer == 0:
                dd_schedule = self.initialize_dynamic_definition_schedule()
            elif len(largest_bins) == 0:
                break
            else:
                bin_to_expand = largest_bins.pop(0)
                dd_schedule = self.next_dynamic_definition_schedule(
                    recursion_layer=bin_to_expand["recursion_layer"],
                    bin_id=bin_to_expand["bin_id"],
                )
            self.times["get_dd_schedule"] += perf_counter() - get_dd_schedule_begin

            merged_subcircuit_entry_probs = self.merge_states_into_bins(dd_schedule)

            """ Build from the merged subcircuit entries """
            graph_contractor = GraphContractor(
                compute_graph=self.compute_graph,
                subcircuit_entry_probs=merged_subcircuit_entry_probs,
                num_cuts=self.num_cuts,
            )
            reconstructed_prob = graph_contractor.reconstructed_prob
            smart_order = graph_contractor.smart_order
            self.times = add_times(times_a=self.times, times_b=graph_contractor.times)

            self.dd_bins[recursion_layer] = dd_schedule
            self.dd_bins[recursion_layer]["smart_order"] = smart_order
            self.dd_bins[recursion_layer]["bins"] = reconstructed_prob
            self.dd_bins[recursion_layer]["expanded_bins"] = []
            # [print(field,self.dd_bins[recursion_layer][field]) for field in self.dd_bins[recursion_layer]]

            """ Sort and truncate the largest bins """
            sort_begin = perf_counter()
            has_merged_states = False
            for subcircuit_idx in dd_schedule["subcircuit_state"]:
                if "merged" in dd_schedule["subcircuit_state"][subcircuit_idx]:
                    has_merged_states = True
                    break
            if recursion_layer < self.recursion_depth - 1 and has_merged_states:
                bin_indices = np.argpartition(
                    reconstructed_prob, -self.recursion_depth
                )[-self.recursion_depth :]
                for bin_id in bin_indices:
                    if reconstructed_prob[bin_id] > 1 / 2**num_qubits / 10:
                        largest_bins.append(
                            {
                                "recursion_layer": recursion_layer,
                                "bin_id": bin_id,
                                "prob": reconstructed_prob[bin_id],
                            }
                        )
                largest_bins = sorted(
                    largest_bins, key=lambda bin: bin["prob"], reverse=True
                )[: self.recursion_depth]
            self.times["sort"] += perf_counter() - sort_begin
            recursion_layer += 1

    def initialize_dynamic_definition_schedule(self):
        schedule = {}
        schedule["subcircuit_state"] = {}
        schedule["upper_bin"] = None

        subcircuit_capacities = {
            subcircuit_idx: self.compute_graph.nodes[subcircuit_idx]["effective"]
            for subcircuit_idx in self.compute_graph.nodes
        }
        subcircuit_active_qubits = self.distribute_load(
            capacities=subcircuit_capacities
        )
        # print('subcircuit_active_qubits:',subcircuit_active_qubits)
        for subcircuit_idx in subcircuit_active_qubits:
            num_zoomed = 0
            num_active = subcircuit_active_qubits[subcircuit_idx]
            num_merged = (
                self.compute_graph.nodes[subcircuit_idx]["effective"]
                - num_zoomed
                - num_active
            )
            schedule["subcircuit_state"][subcircuit_idx] = [
                "active" for _ in range(num_active)
            ] + ["merged" for _ in range(num_merged)]
        return schedule

    def next_dynamic_definition_schedule(self, recursion_layer, bin_id):
        # print('Zoom in recursion layer %d bin %d'%(recursion_layer,bin_id))
        num_active = 0
        for subcircuit_idx in self.dd_bins[recursion_layer]["subcircuit_state"]:
            num_active += self.dd_bins[recursion_layer]["subcircuit_state"][
                subcircuit_idx
            ].count("active")
        binary_bin_idx = bin(bin_id)[2:].zfill(num_active)
        # print('binary_bin_idx = %s'%(binary_bin_idx))
        smart_order = self.dd_bins[recursion_layer]["smart_order"]
        next_dd_schedule = {
            "subcircuit_state": copy.deepcopy(
                self.dd_bins[recursion_layer]["subcircuit_state"]
            )
        }
        binary_state_idx_ptr = 0
        for subcircuit_idx in smart_order:
            for qubit_ctr, qubit_state in enumerate(
                next_dd_schedule["subcircuit_state"][subcircuit_idx]
            ):
                if qubit_state == "active":
                    next_dd_schedule["subcircuit_state"][subcircuit_idx][qubit_ctr] = (
                        int(binary_bin_idx[binary_state_idx_ptr])
                    )
                    binary_state_idx_ptr += 1
        next_dd_schedule["upper_bin"] = (recursion_layer, bin_id)

        subcircuit_capacities = {
            subcircuit_idx: next_dd_schedule["subcircuit_state"][subcircuit_idx].count(
                "merged"
            )
            for subcircuit_idx in next_dd_schedule["subcircuit_state"]
        }
        subcircuit_active_qubits = self.distribute_load(
            capacities=subcircuit_capacities
        )
        # print('subcircuit_active_qubits:',subcircuit_active_qubits)
        for subcircuit_idx in next_dd_schedule["subcircuit_state"]:
            num_active = subcircuit_active_qubits[subcircuit_idx]
            for qubit_ctr, qubit_state in enumerate(
                next_dd_schedule["subcircuit_state"][subcircuit_idx]
            ):
                if qubit_state == "merged" and num_active > 0:
                    next_dd_schedule["subcircuit_state"][subcircuit_idx][
                        qubit_ctr
                    ] = "active"
                    num_active -= 1
            assert num_active == 0
        return next_dd_schedule

    def distribute_load(self, capacities):
        total_load = min(sum(capacities.values()), self.mem_limit)
        total_capacity = sum(capacities.values())
        loads = {subcircuit_idx: 0 for subcircuit_idx in capacities}

        for slot_idx in loads:
            loads[slot_idx] = int(capacities[slot_idx] / total_capacity * total_load)
        total_load -= sum(loads.values())

        for slot_idx in loads:
            while total_load > 0 and loads[slot_idx] < capacities[slot_idx]:
                loads[slot_idx] += 1
                total_load -= 1
        # print('capacities = {}. total_capacity = {:d}'.format(capacities,total_capacity))
        # print('loads = {}. remaining total_load = {:d}'.format(loads,total_load))
        assert total_load == 0
        return loads

    def merge_states_into_bins(self, dd_schedule):
        """
        The first merge of subcircuit probs using the target number of bins
        Saves the overhead of writing many states in the first SM recursion
        """
        begin = perf_counter()
        merged_subcircuit_entry_probs = {}
        for subcircuit_idx in self.compute_graph.nodes:
            merged_subcircuit_entry_probs[subcircuit_idx] = {}
            for subcircuit_entry_init_meas in self.subcircuit_entry_probs[
                subcircuit_idx
            ]:
                unmerged_prob_vector = self.subcircuit_entry_probs[subcircuit_idx][
                    subcircuit_entry_init_meas
                ]
                merged_subcircuit_entry_probs[subcircuit_idx][
                    subcircuit_entry_init_meas
                ] = merge_prob_vector(
                    unmerged_prob_vector=unmerged_prob_vector,
                    qubit_states=dd_schedule["subcircuit_state"][subcircuit_idx],
                )
        self.times["merge_states_into_bins"] += perf_counter() - begin
        return merged_subcircuit_entry_probs


def read_dd_bins(subcircuit_out_qubits, dd_bins):
    num_qubits = sum(
        [
            len(subcircuit_out_qubits[subcircuit_idx])
            for subcircuit_idx in subcircuit_out_qubits
        ]
    )
    reconstructed_prob = np.zeros(2**num_qubits, dtype=np.float32)
    # print(subcircuit_out_qubits)
    for recursion_layer in dd_bins:
        # print('-'*20,'Verify Recursion Layer %d'%recursion_layer,'-'*20)
        # [print(field,dd_bins[recursion_layer][field]) for field in dd_bins[recursion_layer]]
        num_active = sum(
            [
                dd_bins[recursion_layer]["subcircuit_state"][subcircuit_idx].count(
                    "active"
                )
                for subcircuit_idx in dd_bins[recursion_layer]["subcircuit_state"]
            ]
        )
        for bin_id, bin_prob in enumerate(dd_bins[recursion_layer]["bins"]):
            if bin_prob > 0 and bin_id not in dd_bins[recursion_layer]["expanded_bins"]:
                binary_bin_id = bin(bin_id)[2:].zfill(num_active)
                # print('dd bin %s'%binary_bin_id)
                binary_full_state = ["" for _ in range(num_qubits)]
                for subcircuit_idx in dd_bins[recursion_layer]["smart_order"]:
                    subcircuit_state = dd_bins[recursion_layer]["subcircuit_state"][
                        subcircuit_idx
                    ]
                    for subcircuit_qubit_idx, qubit_state in enumerate(
                        subcircuit_state
                    ):
                        qubit_idx = subcircuit_out_qubits[subcircuit_idx][
                            subcircuit_qubit_idx
                        ]
                        if qubit_state == "active":
                            binary_full_state[qubit_idx] = binary_bin_id[0]
                            binary_bin_id = binary_bin_id[1:]
                        else:
                            binary_full_state[qubit_idx] = "%s" % qubit_state
                # print('reordered qubit state = {}'.format(binary_full_state))
                merged_qubit_indices = []
                for qubit, qubit_state in enumerate(binary_full_state):
                    if qubit_state == "merged":
                        merged_qubit_indices.append(qubit)
                num_merged = len(merged_qubit_indices)
                average_state_prob = bin_prob / 2**num_merged
                for binary_merged_state in itertools.product(
                    ["0", "1"], repeat=num_merged
                ):
                    for merged_qubit_ctr in range(num_merged):
                        binary_full_state[merged_qubit_indices[merged_qubit_ctr]] = (
                            binary_merged_state[merged_qubit_ctr]
                        )
                    full_state = "".join(binary_full_state)[::-1]
                    full_state_idx = int(full_state, 2)
                    reconstructed_prob[full_state_idx] = average_state_prob
                #     print('--> full state {} {:d}. p = {:.3e}'.format(full_state,full_state_idx,average_state_prob))
                # print()
    return reconstructed_prob


def full_verify(full_circuit, complete_path_map, subcircuits, dd_bins):
    ground_truth = evaluate_circ(circuit=full_circuit, backend="statevector_simulator")
    subcircuit_out_qubits = get_reconstruction_qubit_order(
        full_circuit=full_circuit,
        complete_path_map=complete_path_map,
        subcircuits=subcircuits,
    )
    reconstructed_prob = read_dd_bins(
        subcircuit_out_qubits=subcircuit_out_qubits, dd_bins=dd_bins
    )
    real_probability = quasi_to_real(
        quasiprobability=reconstructed_prob, mode="nearest"
    )
    approximation_error = (
        MSE(target=ground_truth, obs=real_probability)
        * 2**full_circuit.num_qubits
        / np.linalg.norm(ground_truth) ** 2
    )
    return reconstructed_prob, approximation_error


def merge_prob_vector(unmerged_prob_vector, qubit_states):
    num_active = qubit_states.count("active")
    num_merged = qubit_states.count("merged")
    merged_prob_vector = np.zeros(2**num_active, dtype="float32")
    # print('merging with qubit states {}. {:d}-->{:d}'.format(
    #     qubit_states,
    #     len(unmerged_prob_vector),len(merged_prob_vector)))
    for active_qubit_states in itertools.product(["0", "1"], repeat=num_active):
        if len(active_qubit_states) > 0:
            merged_bin_id = int("".join(active_qubit_states), 2)
        else:
            merged_bin_id = 0
        for merged_qubit_states in itertools.product(["0", "1"], repeat=num_merged):
            active_ptr = 0
            merged_ptr = 0
            binary_state_id = ""
            for qubit_state in qubit_states:
                if qubit_state == "active":
                    binary_state_id += active_qubit_states[active_ptr]
                    active_ptr += 1
                elif qubit_state == "merged":
                    binary_state_id += merged_qubit_states[merged_ptr]
                    merged_ptr += 1
                else:
                    binary_state_id += "%s" % qubit_state
            state_id = int(binary_state_id, 2)
            merged_prob_vector[merged_bin_id] += unmerged_prob_vector[state_id]
    return merged_prob_vector
