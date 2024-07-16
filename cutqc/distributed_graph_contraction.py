"""
File: distributed_graph_contraction.py
Original Author: Wei Tang (tangwei13579@gmail.com)
Current Version Author: Charles "Chuck" Garcia (chuckgarcian@utexas.edu)
Description: Distributed implementation of Wei Tang's original TensorFlow CutQC implementation.
"""
# TODO: make cutting and eval optional and optional to load in cut and eval data

import itertools
from time import perf_counter
from typing import List
import numpy as np
import torch
import torch.distributed as dist
from cutqc.post_process_helper import ComputeGraph

# TODO: Add support for 'gloo' communication backend (cpu)
# TODO: Autodetect communication backend 
# TODO: Make it cleaner node configuration and support for more distributed schedulers other then slurm.
# TODO: Make a general graph contractor class and inherit from it

class DistributedGraphContractor(object):
    def __init__(self) -> None: 
        super().__init__()
        self.times = {}            
        self.reconstructed_prob = None
        
        # Used to compute
        self.compute_graph = None
        self.subcircuit_entry_probs = None
        self.num_cuts = None
    
    def reconstruct (self, compute_graph: ComputeGraph, subcircuit_entry_probs: dict, num_cuts: int) -> None:
        '''
        Performs subcircuit reconstruction.                 
        '''
        
        # Setups Graph Contractor for contraction
        self.compute_graph = compute_graph
        self.subcircuit_entry_probs = subcircuit_entry_probs
        self.num_cuts = num_cuts
        self._set_smart_order ()
        self.overhead = {"additions": 0, "multiplications": 0}
        
        return self._compute ()    

    def _set_smart_order (self) -> None:
        '''
        Sets the order in which kronecker products are computed in. Specefically 
        order is to sort greedy-subcircuit-order.
        '''

        # Retrieve list of all subcircuit lengths
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

    def _compute(self):
        '''
        Performs distributed graph contraction. Returns the reconstructed probability
        '''
        
        edges = self.compute_graph.get_edges(from_node=None, to_node=None)

        # Assemble sequence of uncomputed kronecker products, to distribute to nodes later
        partial_compute_begin = perf_counter()
        summation_terms_sequence = []
        counter = 0

        for edge_bases in itertools.product(["I", "X", "Y", "Z"], repeat=len(edges)):
            summation_term = self._get_paulibase_probability(self, edge_bases, edges)
            summation_terms_sequence.append(torch.stack(summation_term, dim=0))
            counter += 1
            
        self.compute_graph.remove_bases_from_edges(edges=self.compute_graph.edges)
        partial_compute_time = perf_counter() - partial_compute_begin
        scale_begin = perf_counter()

        # Disribute and Execute reconstruction on nodes
        num_batches = dist.get_world_size - 1 # No batch for host
        reconstructed_prob = self._send_distributed (summation_terms_sequence, num_batches)
        
        return reconstructed_prob
        
        reconstructed_prob = tf.math.scalar_mul(
            1 / 2**self.num_cuts, reconstructed_prob
        ).numpy()
        scale_time = perf_counter() - scale_begin

        self.times["compute"] = (
            partial_compute_time / counter * 4 ** len(edges) + scale_time
        )
        self.overhead["additions"] = int(
            self.overhead["additions"] / counter * 4 ** len(edges)
        )
        self.overhead["multiplications"] = int(
            self.overhead["multiplications"] / counter * 4 ** len(edges)
        )
            
    def _send_distributed(self, dataset: List[torch.Tensor], num_batches: int) -> torch.Tensor:
        '''
        Decomposes `dataset`list into 'num_batches' number of batches and distributes
        to worker processes. 
        '''

        # Batch all uncomputed product tuples into batches
        batches = torch.stack(dataset).chunk (chunks=(num_batches))
        
        # Send off to nodes for compute
        for dst_rank, batch in enumerate(batches):
            # TODO: Convert to non-blocking send
            shape_data = batch.shape
            dist.send (torch.tensor(shape_data).cuda(), dst=dst_rank+1) # Exclude Rank 0 Host 
            dist.send (batch.cuda (),  dst=dst_rank+1) 
        
        # Receive Results 
        # TODO: Receive and reduce outputs outside of this function -- beyond the scope of this function
        output_buff = torch.zeros (self.tensor_size**self.smart_order).cuda ()  
        dist.reduce(output_buff, dst=0, op=dist.ReduceOp.SUM)

        return output_buff * (1/2)

    def _get_subcircuit_entry_prob(self, subcircuit_idx: int):
        """
        Returns The subcircuit Entry Probability for the subcircuit at index
        'SUBCIRCUIT_IDX' 
        """

        subcircuit_entry_init_meas = self.compute_graph.get_init_meas(subcircuit_idx)
        return self.subcircuit_entry_probs[subcircuit_idx][subcircuit_entry_init_meas]

    def _get_paulibase_probability(self, edge_bases: tuple, edges: list):
        """
        Returns probability contribution for the basis 'EDGE_BASES' in the circuit
        cutting decomposition.
        """
        
        # Create list of kronecker product terms
        product_tuple = [self._get_subcircuit_entry_prob(self, subcircuit_idx) for subcircuit_idx in self.smart_order]
        return product_tuple
        
        gc.compute_graph.assign_bases_to_edges(edge_bases=edge_bases, edges=edges)

        # Kronecker prodcut tuple
        product_tuple = []
        
        for subcircuit_idx in gc.smart_order:
            product_tuple.append(get_subcircuit_entry_prob(gc, subcircuit_idx))
            
        return product_tuple


            
