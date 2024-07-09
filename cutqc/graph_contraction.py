import itertools, math
from time import perf_counter
import numpy as np
import logging, os

logging.disable(logging.WARNING)
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"
import tensorflow as tf

tf.executing_eagerly()
def compute_summation_term(prob_vectors: tf.Tensor) -> tf.Tensor: 
    """
    Returns the computed kronecker product sum
    """
    tf.executing_eagerly()
    print ("\n")
    # print (type(prob_vectors))
    # print (f"Before Squeeze shape: {prob_vectors.shape}")
    # prob_vectors = tf.squeeze(prob_vectors,axis=0)
    print (type(prob_vectors))
    print (f"After Squeeze shape: {prob_vectors.shape}")
    
    # Initialy set it to an identity value
    summation_term = tf.constant (1) 
    
    # Compute Kronecker
    for subcircuit_entry_prob in prob_vectors:
        new_shape = tf.shape (subcircuit_entry_prob) * tf.shape (summation_term)
        tf.autograph.experimental.set_loop_options(
        shape_invariants=[summation_term, new_shape])
        
        print ("\nsubcircuit_entry_prob")
        print (type(subcircuit_entry_prob))
        print (f"subcircuit_entry_prob.shape: {subcircuit_entry_prob.shape}")
        summation_term = tf.tensordot(summation_term, subcircuit_entry_prob, axes=0)
            
    tf.reshape(summation_term, [-1])
    exit ()
    return summation_term

def old_compute_summation_term(*argv):
    print (type(argv))
    print (f"shape: {argv}")
    summation_term = None
    for subcircuit_entry_prob in argv:
        print ("\nsubcircuit_entry_prob")
        print (type(subcircuit_entry_prob))
        print (f"subcircuit_entry_prob.shape: {subcircuit_entry_prob.shape}")
        if summation_term is None:
            summation_term = subcircuit_entry_prob
        else:
            summation_term = tf.reshape(
                tf.tensordot(summation_term, subcircuit_entry_prob, axes=0), [-1]
            )
    return summation_term



class GraphContractor(object):
    def __init__(self, compute_graph, subcircuit_entry_probs, num_cuts) -> None:
        super().__init__()
        self.times = {}
        self.compute_graph = compute_graph
        self.subcircuit_entry_probs = subcircuit_entry_probs
        self.num_cuts = num_cuts
        self.subcircuit_entry_lengths = {}
        for subcircuit_idx in subcircuit_entry_probs:
            first_entry_init_meas = list(subcircuit_entry_probs[subcircuit_idx].keys())[
                0
            ]
            length = len(subcircuit_entry_probs[subcircuit_idx][first_entry_init_meas])
            self.subcircuit_entry_lengths[subcircuit_idx] = length
        self.num_qubits = 0
        for subcircuit_idx in compute_graph.nodes:
            self.num_qubits += compute_graph.nodes[subcircuit_idx]["effective"]

        self.smart_order = sorted(
            self.subcircuit_entry_lengths.keys(),
            key=lambda subcircuit_idx: self.subcircuit_entry_lengths[subcircuit_idx],
        )
        self.max_effective = self.subcircuit_entry_lengths[self.smart_order[-1]]
        
        self.overhead = {"additions": 0, "multiplications": 0}
        # self.reconstructed_prob = self.compute()
        self.reconstructed_prob = self.old_compute()

    def old_compute(self):
        edges = self.compute_graph.get_edges(from_node=None, to_node=None)

        make_dataset_begin = perf_counter()
    
        cg_dataset = []
        for edge_bases in itertools.product(["I", "X", "Y", "Z"], repeat=len(edges)):
            self.compute_graph.assign_bases_to_edges(edge_bases=edge_bases, edges=edges)
    
            summation_term = get_subcircuit_entry_terms (self)

            self.overhead["multiplications"] -= len(summation_term[0])
            # cg_print (summation_term)
        
            # dataset_elem = tf.data.Dataset.from_tensors(summation_term)
            cg_dataset.append (summation_term)
            # dataset_elem = tf.data.Dataset.from_tensors(tuple(summation_term))
            # if dataset is None:
            #     dataset = dataset_elem
            # else:
            #     dataset = dataset.concatenate(dataset_elem)
        
        # cg_print (np.array(cg_dataset))
        dataset  = tf.data.Dataset.from_tensor_slices (cg_dataset)
    
    
        # exit()
        self.compute_graph.remove_bases_from_edges(edges=self.compute_graph.edges)
        # ds = dataset.batch (
        #     batch_size=1, deterministic=False
        # )
        
        cg_print (np.array(list(dataset.as_numpy_iterator())))
        
        
        self.times["make_dataset"] = perf_counter() - make_dataset_begin

        compute_begin = perf_counter()
        dataset = dataset.map(
            compute_summation_term,
            deterministic=False,
        )

        reconstructed_prob = None
        for x in dataset:
            if reconstructed_prob is None:
                reconstructed_prob = x
            else:
                self.overhead["additions"] += len(reconstructed_prob)
                reconstructed_prob += x
        reconstructed_prob = tf.math.scalar_mul(
            1 / 2**self.num_cuts, reconstructed_prob
        ).numpy()
        self.times["compute"] = perf_counter() - compute_begin
        return reconstructed_prob

    def compute(self):
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


def cg_print (obj):
    print (f"Object Type: {type(obj)}")
    # print (f"Object: {obj}")
    print (f"Shape: {obj.shape}")

def pad_to_length (arr : np.array, n : int):
    '''
    Returns the padded vector 'ARR' such that it is of length 'N'
    '''
    assert arr.ndim == 1, "Invalid numpy array dimension -- Must be vector"
    
    pad_amount = n - arr.shape[0]
    return np.pad(arr, (0, pad_amount), mode='constant') if pad_amount > 0 else arr

def get_subcircuit_entry_prob(gc: GraphContractor, subcircuit_idx: int):
    """
    Returns The subcircuit Entry Probability for the subcircuit at index
    'SUBCIRCUIT_IDX' of the graph contractor object 'GC'.
    """

    subcircuit_entry_init_meas = gc.compute_graph.get_init_meas(subcircuit_idx)
    return gc.subcircuit_entry_probs[subcircuit_idx][subcircuit_entry_init_meas]

def get_subcircuit_entry_terms (gc: GraphContractor):
        summation_term = []
        cumulative_len = 1
        
        # Extract all subcircuit prob vectors and pad them to be same shape
        for subcircuit_idx in gc.smart_order:
            subcircuit_entry_prob = get_subcircuit_entry_prob(gc, subcircuit_idx)
            subcircuit_entry_prob = pad_to_length (subcircuit_entry_prob, gc.max_effective) 
            summation_term.append(subcircuit_entry_prob)
            cumulative_len *= len(subcircuit_entry_prob)
            gc.overhead["multiplications"] += cumulative_len
        

        return tf.constant(summation_term)

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
