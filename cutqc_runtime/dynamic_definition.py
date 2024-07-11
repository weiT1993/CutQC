import copy
from time import perf_counter
import numpy as np

from cutqc_runtime.graph_contraction import GraphContractor
from cutqc.helper_fun import add_times


class DynamicDefinition(object):
    def __init__(
        self, compute_graph, data_folder, num_cuts, mem_limit, recursion_depth
    ) -> None:
        super().__init__()
        self.compute_graph = compute_graph
        self.data_folder = data_folder
        self.num_cuts = num_cuts
        self.mem_limit = mem_limit
        self.recursion_depth = recursion_depth
        self.dd_bins = {}
        self.overhead = {"additions": 0, "multiplications": 0}

        self.times = {"get_dd_schedule": 0, "sort": 0}

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
            # print('-'*10,'Recursion Layer %d'%(recursion_layer),'-'*10)
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

            """ Build from the merged subcircuit entries """
            graph_contractor = GraphContractor(
                compute_graph=self.compute_graph,
                dd_schedule=dd_schedule,
                num_cuts=self.num_cuts,
            )
            reconstructed_prob = graph_contractor.reconstructed_prob
            smart_order = graph_contractor.smart_order
            recursion_overhead = graph_contractor.overhead
            self.overhead["additions"] += recursion_overhead["additions"]
            self.overhead["multiplications"] += recursion_overhead["multiplications"]
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
