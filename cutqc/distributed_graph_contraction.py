"""
File: distributed_graph_contraction.py
Original Author: Wei Tang (tangwei13579@gmail.com)
Current Version Author: Charles "Chuck" Garcia (chuckgarcian@utexas.edu)
Description: Distributed implementation of Wei Tang's original TensorFlow CutQC implementation.
"""

import itertools
from time import perf_counter
from typing import List
import numpy as np
import torch
import torch.distributed as dist

from helper_functions.metrics import MSE
from cutqc.post_process_helper import ComputeGraph

# TODO: Support for more distributed schedulers other then slurm.
# TODO: Make a general graph contractor class and inherit from it. - Ellie

__host_machine__ = 0

class DistributedGraphContractor(object):
    def __init__(self, local_rank=None) -> None: 
        # Starts main worker loop
        self.local_rank = local_rank
        self.mp_backend = torch.device("cuda:{}".format(local_rank) if (dist.get_backend () == 'nccl') else "cpu")
        self.compute_device = torch.device("cuda:{}".format(local_rank))
        
        if dist.get_rank() != __host_machine__: self._initiate_worker_loop()
        super().__init__()        
        self.times = {'compute': 0}            

        # Used to compute
        self.compute_graph = None
        self.subcircuit_entry_probs = None
        self.num_cuts = None
        self.reconstructed_prob = None
        
    def reconstruct(self, compute_graph: ComputeGraph, subcircuit_entry_probs: dict, num_cuts: int) -> None:
        '''
        Performs subcircuit reconstruction.                 
        '''
        # Set up Graph Contractor for contraction
        self.compute_graph = compute_graph
        self.subcircuit_entry_probs = subcircuit_entry_probs
        self.num_cuts = num_cuts
        self._set_smart_order()
        self.overhead = {"additions": 0, "multiplications": 0}
        
        start_time = perf_counter()
        res = self._compute()    
        end_time = perf_counter() - start_time
        self.times['compute'] += end_time
        
        return res 

    def terminate_distributed_process(self):
        '''
        Sends signal to workers to finish their execution.
        '''
        
        termination_signal = torch.tensor([-1], dtype=torch.int64).to(self.mp_backend)
        for rank in range(1, dist.get_world_size()):
            dist.send(termination_signal, dst=rank)
            
        print ("DESTROYING NOW ! {}".format ( self.times['compute']), flush=True)
        dist.destroy_process_group()

    def _set_smart_order(self) -> None:
        '''
        Sets the order in which Kronecker products are computed. Specifically, 
        the order is to sort by greedy subcircuit or der.
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
        
        self.max_effective = subcircuit_entry_lengths[self.smart_order[-1]] # For Padding
        self.subcircuit_entry_lengths = [subcircuit_entry_lengths[i] for i in self.smart_order]
        print ("subcircuit_entry_lengh: {}".format(self.subcircuit_entry_lengths), flush=True)
        self.result_size = np.prod(self.subcircuit_entry_lengths)

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
            summation_terms_sequence.append(torch.stack(summation_term, dim=0))        
            counter += 1
        
        self.compute_graph.remove_bases_from_edges(edges=self.compute_graph.edges)
        
        # Distribute and Execute reconstruction on nodes
        
        num_batches = dist.get_world_size() - 1 # No batch for host
        reconstructed_prob = self._send_distributed(summation_terms_sequence, num_batches)

        return reconstructed_prob.cpu().numpy()
            
    def _send_distributed(self, dataset: List[torch.Tensor], num_batches: int) -> torch.Tensor:
        '''
        Decomposes `dataset`list into 'num_batches' number of batches and distributes
        to worker processes. 
        '''
        
        # Batch all uncomputed product tuples into batches
        if (len(dataset) < num_batches):
            print ("LEN(DATASET): {}".format (len(dataset)))
            print ("NUMBER BATCHES: {}".format (num_batches))
            raise ValueError ("Invalid number of requested batches -- Too many nodes allocated") 
        
        batches = torch.stack(dataset).tensor_split(num_batches)
        tensor_sizes_data = torch.tensor(self.subcircuit_entry_lengths, dtype=torch.int64, requires_grad=False).to(self.mp_backend) # Used to strip zero padding 
        tensor_sizes_shape = tensor_sizes_data.shape 
        
        print (f'batches Host: {batches}', flush=True)
        # Send off to nodes for compute
        for dst_rank, batch in enumerate(batches):
            # TODO: Convert to non-blocking send
            dist.send(torch.tensor(tensor_sizes_shape, dtype=torch.int64, requires_grad=False).to(self.mp_backend), dst=dst_rank+1) 
            dist.send(tensor_sizes_data, dst=dst_rank+1)

            dist.send(torch.tensor(batch.shape, requires_grad=False).to(self.mp_backend), dst=dst_rank+1) # Exclude Rank 0 Host 
            dist.send(batch.to(self.mp_backend),  dst=dst_rank+1) 
        
        # Receive Results 
        output_buff = torch.zeros(self.result_size, dtype=torch.float32, requires_grad=False).to(self.mp_backend)  
        
        dist.reduce(output_buff, dst=0, op=dist.ReduceOp.SUM)
        return torch.mul(output_buff, (1/2**self.num_cuts))

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
        self.compute_graph.assign_bases_to_edges(edge_bases=edge_bases, edges=edges)

        # Create list of kronecker product terms
        product_list = []
        
        for size, subcircuit_idx in zip(self.subcircuit_entry_lengths, self.smart_order):
            subcircuit_entry_prob = self._get_subcircuit_entry_prob(subcircuit_idx)
            
            pad_amount = self.max_effective - size
            new_component = torch.tensor(subcircuit_entry_prob, dtype=torch.float32, requires_grad=False)
            new_component = torch.nn.functional.pad(new_component, (0, pad_amount)) 
            product_list.append(new_component)

        return product_list

    def _initiate_worker_loop(self):
        '''
        Primary worker loop.

        Each worker receives a portion of the workload from the host/master node.
        Once done with computation, all nodes perform a collective reduction
        operation back to the host. Synchronization among nodes is provided via
        barriers and blocked message passing.
        '''
        # torch.set_default_device(self.device)        
        # torch.cuda.set_device (self.device)
        # Host will send signal to break from loop
        while True:
            with torch.no_grad ():

                
                # Receive Tensor list information
                tensor_sizes_shape = torch.empty([1], dtype=torch.int64).to(self.mp_backend)
                dist.recv(tensor=tensor_sizes_shape, src=__host_machine__)     
                print ("TENSORSIZES SHAPE {}".format(tensor_sizes_shape.item()), flush=True)
                
                # Check for termination signal
                if tensor_sizes_shape.item() == -1:
                    dist.destroy_process_group()
                    exit ()
                
                # Used to remove padding 
                tensor_sizes = torch.empty(tuple(tensor_sizes_shape), dtype=torch.int64).to(self.mp_backend)
                dist.recv(tensor=tensor_sizes, src=__host_machine__)    

                
                # Get shape of the batch we are receiving 
                shape_tensor = torch.empty([3], dtype=torch.int64).to(self.mp_backend)
                dist.recv(tensor=shape_tensor, src=__host_machine__) 


                # Create an empty batch tensor and receive its data
                batch_received = torch.empty(tuple(shape_tensor), dtype=torch.float32).to(self.mp_backend)
                dist.recv(tensor=batch_received, src=__host_machine__)    
                
                
                # Execute kronecker products in parallel (vectorization)
                torch.cuda.device (self.compute_device)
                lambda_fn = lambda x: compute_kronecker_product(x, tensor_sizes.to (self.compute_device))
                vec_fn = torch.func.vmap(lambda_fn)
                val = batch_received.to(self.compute_device) 
                res = vec_fn(val)
                res = res.sum (dim=0)
                
                # Send Back to host
                dist.reduce(res.to(self.mp_backend), dst=__host_machine__, op=dist.ReduceOp.SUM)
            

# @torch.jit.script
def compute_kronecker_product(components, tensor_sizes):
    '''
    Computes sequence of Kronecker products, where operands are tensors in 'components'.
    '''
    # Initialize the result with the first component, adjusted by its size
    components_list = torch.unbind(components, dim=0)
    res = components_list[0][:tensor_sizes[0]]
    
    # Sequentially compute the Kronecker product with the remaining components
    for component, size in zip(components_list[1:], tensor_sizes[1:]):
        new = torch.kron(res, component[:size])
        del (res)
        del (component)        
        res = new
        
    
    return res


def get_difference(expected, actual):
    diff = torch.abs(expected - actual) 
    return torch.min(diff)
