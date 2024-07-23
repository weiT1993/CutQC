import itertools, math
from time import perf_counter
import numpy as np
import logging, os
from cutqc.post_process_helper import ComputeGraph

# TODO: Rewrite how overhead and timing data are collect so it is clean.

logging.disable(logging.WARNING)
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"
import tensorflow as tf

class GraphContractor(object):
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

        start_time = perf_counter()
        res = self._compute()    
        end_time = perf_counter() - start_time
        self.times['compute'] += end_time

        return res
    
    def _compute(self):
        '''
        Internal function that actualy does the reconstruct
        '''
        edges = self.compute_graph.get_edges(from_node=None, to_node=None)

        partial_compute_begin = perf_counter()
        reconstructed_prob = tf.zeros_like(get_paulibase_probability(self, ["I"] * len(edges), edges))
        counter = 0

        # Compute Kronecker sums over the different basis
        for edge_bases in itertools.product(["I", "X", "Y", "Z"], repeat=len(edges)):
            summation_term = get_paulibase_probability(self, edge_bases, edges)
            reconstructed_prob = tf.add(reconstructed_prob, summation_term)
            self.overhead["additions"] += len(summation_term)
            counter += 1
            
        self.compute_graph.remove_bases_from_edges(edges=self.compute_graph.edges)
        partial_compute_time = perf_counter() - partial_compute_begin

        scale_begin = perf_counter()
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
        return reconstructed_prob
    
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
            print ("Length: gf {}".format (length))
            subcircuit_entry_lengths[subcircuit_idx] = length

        # Sort according to subcircuit lengths (greedy-subcircuit-order)
        self.smart_order = sorted(
            subcircuit_entry_lengths.keys(),
            key=lambda subcircuit_idx: subcircuit_entry_lengths[subcircuit_idx],
        )


def get_subcircuit_entry_prob(gc: GraphContractor, subcircuit_idx: int):
    """
    Returns The subcircuit Entry Probability for the subcircuit at index
    'SUBCIRCUIT_IDX' of the graph contractor object 'GC'.
    """

    subcircuit_entry_init_meas = gc.compute_graph.get_init_meas(subcircuit_idx)
    return gc.subcircuit_entry_probs[subcircuit_idx][subcircuit_entry_init_meas]

def get_paulibase_probability(gc: GraphContractor, edge_bases: tuple, edges: list):
    """
    Returns the probability contribution for the basis 'EDGE_BASES' in the circuit
    cutting decomposition.
    """

    summation_term = None
    gc.compute_graph.assign_bases_to_edges(edge_bases=edge_bases, edges=edges)

    for subcircuit_idx in gc.smart_order:
        subcircuit_entry_prob = get_subcircuit_entry_prob(gc, subcircuit_idx)
        if summation_term is None:
            summation_term = subcircuit_entry_prob
        else:
            summation_term = tf.reshape(
                tf.tensordot(summation_term, subcircuit_entry_prob, axes=0),
                [-1],
            )
            gc.overhead["multiplications"] += len(summation_term)

    return summation_term
