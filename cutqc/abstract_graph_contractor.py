from abc import ABC, abstractmethod
from time import perf_counter
import numpy as np
from cutqc.post_process_helper import ComputeGraph

class AbstractGraphContractor(ABC):

    @abstractmethod
    def _compute(self):
        pass
    
    def reconstruct(self, compute_graph: ComputeGraph, subcircuit_entry_probs: dict, num_cuts: int) -> None:
        self.compute_graph = compute_graph
        self.subcircuit_entry_probs = subcircuit_entry_probs
        self.overhead = {"additions": 0, "multiplications": 0}
        self.num_cuts = num_cuts
        self._set_smart_order()
        
        start_time = perf_counter()
        res = self._compute()    
        end_time = perf_counter() - start_time
        self.times['compute'] = end_time

        return res
    
    def _set_smart_order(self) -> None:
        """
        Sets the order in which Kronecker products are computed (greedy subcircuit order).
        """
        subcircuit_entry_lengths = {}
        for subcircuit_idx in self.subcircuit_entry_probs:
            first_entry_init_meas = list(self.subcircuit_entry_probs[subcircuit_idx].keys())[0]
            length = len(self.subcircuit_entry_probs[subcircuit_idx][first_entry_init_meas])
            subcircuit_entry_lengths[subcircuit_idx] = length

        # Sort according to subcircuit lengths (greedy-subcircuit-order)
        self.smart_order = sorted(
            subcircuit_entry_lengths.keys(),
            key=lambda subcircuit_idx: subcircuit_entry_lengths[subcircuit_idx],
        )
        
        self.subcircuit_entry_lengths = [subcircuit_entry_lengths[i] for i in self.smart_order]
        print(f"subcircuit_entry_length: {self.subcircuit_entry_lengths}", flush=True)
        self.result_size = np.prod(self.subcircuit_entry_lengths)


    def _get_subcircuit_entry_prob(self, subcircuit_idx: int):
        """
        Returns The subcircuit Entry Probability for the subcircuit at index
        'SUBCIRCUIT_IDX' 
        """

        subcircuit_entry_init_meas = self.compute_graph.get_init_meas(subcircuit_idx)
        return self.subcircuit_entry_probs[subcircuit_idx][subcircuit_entry_init_meas]
    
    @abstractmethod
    def _get_paulibase_probability(self, edge_bases: tuple, edges: list):
        pass