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
# TODO: Consider using RPC paradigm for workers

__host_machine__ = 0

class DistributedGraphContractor(object):
    def __init__(self) -> None: 
        # Starts main worker loop
        self.device = torch.device("cuda:{}".format(dist.get_rank()))
        torch.cuda.device(self.device)
        if dist.get_rank () != __host_machine__: self._initiate_worker_loop ()

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
        
        # Set up Graph Contractor for contraction
        self.compute_graph = compute_graph
        self.subcircuit_entry_probs = subcircuit_entry_probs
        self.num_cuts = num_cuts
        self._set_smart_order()
        self.overhead = {"additions": 0, "multiplications": 0}
        
        return self._compute()    

    def terminate_distributed_process (self):
        '''
        Sends signal to workers to finish their execution.
        '''
        print ("Called Terminatet! rank= {}".format(dist.get_rank()))
    
        # device = torch.device("cuda:{}".format(__host_machine__))
        
        # shutdown_flag = torch.ones(1, dtype=torch.int64).to(device)
        # for dst_rank in range (1,  dist.get_world_size):
        #     handle = dist.isend (shutdown_flag,  dst=dst_rank+1) 
        #     handle.wait ()
        
        # dist.barrier()
        # handle.wait ()        
        return
    
    def _set_smart_order (self) -> None:
        '''
        Sets the order in which Kronecker products are computed. Specifically, 
        the order is to sort by greedy subcircuit order.
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
        
        self.max_effective = subcircuit_entry_lengths[self.smart_order[-1]] # For Paddingg

    def _compute(self):
        '''
        Performs distributed graph contraction. Returns the reconstructed probability
        '''        
        edges = self.compute_graph.get_edges(from_node=None, to_node=None)

        # Assemble sequence of uncomputed kronecker products, to distribute to nodes later
        summation_terms_sequence = []
        counter = 0

        for edge_bases in itertools.product(["I", "X", "Y", "Z"], repeat=len(edges)):
            summation_term = self._get_paulibase_probability(edge_bases, edges)
            summation_term = torch.stack(summation_term, dim=0)
            summation_terms_sequence.append (summation_term)
            counter += 1
            
        self.compute_graph.remove_bases_from_edges(edges=self.compute_graph.edges)

        # Disribute and Execute reconstruction on nodes
        torch.cuda.synchronize (self.device) # Sync workers with host
        num_batches = dist.get_world_size () - 1 # No batch for host
        reconstructed_prob = self._send_distributed (summation_terms_sequence, num_batches)
        
        return reconstructed_prob
            
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
        output_buff = torch.zeros (self.max_effective**len(self.smart_order)).cuda ()  
        dist.reduce(output_buff, dst=0, op=dist.ReduceOp.SUM)

        return output_buff * (1/2)

    def _get_subcircuit_entry_prob(self, subcircuit_idx: int):
        """
        Returns The subcircuit Entry Probability for the subcircuit at index
        'SUBCIRCUIT_IDX' 
        """

        subcircuit_entry_init_meas = self.compute_graph.get_init_meas(subcircuit_idx)
        return self.subcircuit_entry_probs[subcircuit_idx][subcircuit_entry_init_meas]

    def _pad_to_length (self, arr : np.array, n : int):
        '''
        Returns the padded vector 'ARR' such that it is of length 'N'
        '''
        assert arr.ndim == 1, "Invalid numpy array dimension -- Must be vector"
        
        pad_amount = n - arr.shape[0]
        return np.pad(arr, (0, pad_amount), mode='constant') if pad_amount > 0 else arr

    def _get_paulibase_probability(self, edge_bases: tuple, edges: list):
        """
        Returns probability contribution for the basis 'EDGE_BASES' in the circuit
        cutting decomposition.
        """
        self.compute_graph.assign_bases_to_edges(edge_bases=edge_bases, edges=edges)

        # Create list of kronecker product terms
        product_list = []

        for subcircuit_idx in self.smart_order:
            subcircuit_entry_prob = self._get_subcircuit_entry_prob(subcircuit_idx)
            subcircuit_entry_prob = self._pad_to_length (subcircuit_entry_prob, self.max_effective) 
            product_list.append(torch.tensor (subcircuit_entry_prob))

        return product_list
        

    def _initiate_worker_loop (self):
        '''
        Primary worker loop.

        Each worker receives a portion of the workload from the host/master node.
        Once done with computation, all nodes perform a collective reduction
        operation back to the host. Synchronization among nodes is provided via
        barriers and blocked message passing.
        '''
        vectorized_kronecker = torch.func.vmap (compute_kronecker_product)
        print ("dist.get_rank (): {}".format (dist.get_rank ()))        
        
        # Shutdown signal sent from host    
        # shutdown_tensor = torch.ones(1, dtype=torch.int64).to(device)    
        # shutdown_sig = dist.irecv (shutdown_tensor, src=__host_machine__)
        
        while True:
            torch.cuda.synchronize (self.device)

            # Break from distributed loop
            # if (shutdown_sig.is_completed ()): break                
        
            # Get shape of the batch we are receiving 
            shape_tensor = torch.zeros([3], dtype=torch.int64, device=self.device) 
            dist.recv(tensor=shape_tensor, src=__host_machine__) 
            
            # Create an empty batch tensor and receive its data
            print(f"Rank {dist.get_rank()}, shapetuple = {shape_tensor}")
            batch_received = torch.empty(tuple(shape_tensor), device=self.device) 
            dist.recv(tensor=batch_received, src=__host_machine__)    
        
            # Execute kronecker products in parallel (vectorization)
            res = vectorized_kronecker (batch_received)
            res = res.sum(dim=0)
            print(f"Res: {res.shape}") 
            
            # Send Back to host
            dist.reduce(res, dst=__host_machine__, op=dist.ReduceOp.SUM)

        exit ()


@torch.jit.script
def compute_kronecker_product(components) -> torch.Tensor:
    '''
    Computes sequence of Kronecker products, where operands are tensors in 'components'.
    '''
    res = components[0]
    for kron_prod in components[1:]:
        res = torch.kron(res, kron_prod)
    return res


            
