"""
File: distributed_graph_contraction.py
Original Author: Wei Tang (tangwei13579@gmail.com)
Current Version Author: Charles "Chuck" Garcia (chuckgarcian@utexas.edu)
Description: Distributed implementation of Wei Tang's original TensorFlow CutQC implementation.
"""

import itertools
from time import perf_counter
from typing import List, Optional
import numpy as np
import torch
import torch.distributed as dist
from cutqc.abstract_graph_contractor import AbstractGraphContractor
from cutqc.post_process_helper import ComputeGraph


__host_machine__ = 0


class DistributedGraphContractor(AbstractGraphContractor):
    """
     Distributed Graph Contractor Implementation
    
     Args:
            local_rank (int): Node identifier value
            compute_backend (str): Device used for compute (Default is GPU)
            
    """
    def __init__(self, local_rank: Optional[int] = None, compute_backend: str = 'gpu') -> None:
        self.local_rank = local_rank
        
        # Set up compute devices based on backend
        self.mp_backend = torch.device(f"cuda:{local_rank}" if dist.get_backend() == 'nccl' else "cpu") # Deviced used MP
        self.compute_device = torch.device(f"cuda:{local_rank}") if compute_backend == 'gpu' else self.mp_backend
        print ("Worker {}, compute_device: {}".format (dist.get_rank(), self.compute_device), flush=True)

        if dist.get_rank() != __host_machine__:
            self._initiate_worker_loop()
        
        self.times = {'compute': 0}
        self.compute_graph = None
        self.subcircuit_entry_probs = None
        self.reconstructed_prob = None


    def terminate_distributed_process(self):
        """
        Sends signal to workers to finish their execution.
        """
        termination_signal = torch.tensor([-1], dtype=torch.int64).to(self.mp_backend)
        for rank in range(1, dist.get_world_size()):
            dist.send(termination_signal, dst=rank)
        
        print(f"DESTROYING NOW! {self.times['compute']}", flush=True)
        dist.destroy_process_group()

    def _get_paulibase_probability (self, edge_bases: tuple, edges: list):
        """
        Returns probability contribution for the basis 'edge_bases' in the circuit
        cutting decomposition.
        """
        with torch.no_grad():
            self.compute_graph.assign_bases_to_edges(edge_bases=edge_bases, edges=edges)

            # Create list of kronecker product terms
            flat_size = np.sum(self.subcircuit_entry_lengths)
            flat = torch.empty(flat_size)
            idx = 0
            
            # Store all probability tensors into single flattened tensor
            for size, subcircuit_idx in zip(self.subcircuit_entry_lengths, self.smart_order):
                subcircuit_entry_prob = self._get_subcircuit_entry_prob(subcircuit_idx)
                flat[idx:idx+size] = torch.tensor(subcircuit_entry_prob, dtype=torch.float32)
                idx += size

        return flat

    def _send_distributed(self, dataset: List[torch.Tensor], num_batches: int) -> torch.Tensor:
        """
        Decomposes `dataset` list into 'num_batches' number of batches and distributes
        to worker processes.
        """
        torch.set_default_device(self.mp_backend)

        with torch.no_grad():
            if len(dataset) < num_batches:
                raise ValueError("Invalid number of requested batches -- Too many nodes allocated")
            
            batches = torch.stack(dataset).tensor_split(num_batches)
            tensor_sizes = torch.tensor(self.subcircuit_entry_lengths, dtype=torch.int64)
            tensor_sizes_shape = torch.tensor(tensor_sizes.shape, dtype=torch.int64)

            if dist.get_backend() == 'gloo':
                op_list = []
                # List of sending objects
                for dst, batch in enumerate(batches, start=1):
                    op_list.extend([
                        dist.P2POp(dist.isend, tensor_sizes_shape, dst),
                        dist.P2POp(dist.isend, tensor_sizes, dst),
                        dist.P2POp(dist.isend, torch.tensor(batch.shape, dtype=torch.int64), dst),
                        dist.P2POp(dist.isend, batch, dst),
                    ])
                handles = dist.batch_isend_irecv(op_list)
            else:
                # NCCL backend
                for dst_rank, batch in enumerate(batches, start=1):
                    # Non-Blocking send on NCCL
                    dist.isend(tensor_sizes_shape, dst=dst_rank)
                    dist.isend(tensor_sizes, dst=dst_rank)
                    dist.isend(torch.tensor(batch.shape), dst=dst_rank)
                    dist.isend(batch.to(self.compute_device), dst=dst_rank)
            
            # Receive Results
            output_buff = torch.zeros(self.result_size, dtype=torch.float32)
            dist.reduce(output_buff, dst=0, op=dist.ReduceOp.SUM)
        
        return torch.mul(output_buff, (1/2**self.num_cuts))

    def _compute(self) -> np.ndarray:
        """
        Performs distributed graph contraction. Returns the reconstructed probability.
        """
        edges = self.compute_graph.get_edges(from_node=None, to_node=None)
        summation_terms_sequence = []

        # Assemble sequence of uncomputed kronecker products
        for edge_bases in itertools.product(["I", "X", "Y", "Z"], repeat=len(edges)):
            summation_terms = self._get_paulibase_probability(edge_bases, edges)
            summation_terms_sequence.append(summation_terms)

        self.compute_graph.remove_bases_from_edges(edges=self.compute_graph.edges)
        
        # Distribute and Execute reconstruction on nodes
        num_batches = dist.get_world_size() - 1  # No batch for host
        reconstructed_prob = self._send_distributed(summation_terms_sequence, num_batches)

        return reconstructed_prob.cpu().numpy()
    

    def _receive_from_host(self):
        """
        Receives tensors sent by host. Returns batch and unpadded sizes.
        """
        torch.set_default_device(self.mp_backend)
        
        with torch.no_grad():
            tensor_sizes_shape = torch.empty([1], dtype=torch.int64)
            dist.recv(tensor=tensor_sizes_shape, src=0)
            
            # Check for termination signal
            if tensor_sizes_shape.item() == -1:
                print(f"WORKER {dist.get_rank()} DYING", flush=True)
                dist.destroy_process_group()
                exit()

            # Used to unflatten
            tensor_sizes = torch.empty(tensor_sizes_shape, dtype=torch.int64)
            dist.recv(tensor=tensor_sizes, src=0)

            # Get shape of the batch we are receiving
            batch_shape = torch.empty([2], dtype=torch.int64)
            dist.recv(tensor=batch_shape, src=0)
            
            # Create an empty batch tensor and receive its data
            batch = torch.empty(tuple(batch_shape), dtype=torch.float32)
            dist.recv(tensor=batch, src=0)
        
        return batch, tensor_sizes

    def _initiate_worker_loop(self):
        """
        Primary worker loop.

        Each worker receives a portion of the workload from the host/master node.
        Once done with computation, all nodes perform a collective reduction
        operation back to the host. Synchronization among nodes is provided via
        barriers and blocked message passing.
        """
        while True:
            batch, tensor_sizes = self._receive_from_host()
        
            # Execute kronecker products in parallel (vectorization)
            torch.cuda.device(self.compute_device)
            lambda_fn = lambda x: compute_kronecker_product(x, tensor_sizes)
            vec_fn = torch.func.vmap(lambda_fn)
            res = vec_fn(batch.to(self.compute_device))
            res = res.sum(dim=0)
            
            # Send Back to host
            dist.reduce(res.to(self.mp_backend), dst=__host_machine__, op=dist.ReduceOp.SUM)

from functools import reduce
def compute_kronecker_product(flattened: torch.Tensor, sizes: torch.Tensor) -> torch.Tensor:
    """
    Computes sequence of Kronecker products, where operands are tensors in 'components'.
    """
    tensors = torch.split(flattened, tuple(sizes))
    return reduce(torch.kron, tensors)
