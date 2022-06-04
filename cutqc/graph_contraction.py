import itertools, math
from time import perf_counter
import numpy as np
import logging, os

logging.disable(logging.WARNING)
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"
import tensorflow as tf


def compute_summation_term(*argv):
    summation_term = None
    for subcircuit_entry_prob in argv:
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
        self.overhead = {"additions": 0, "multiplications": 0}
        self.reconstructed_prob = self.compute()

    def compute(self):
        edges = self.compute_graph.get_edges(from_node=None, to_node=None)

        make_dataset_begin = perf_counter()
        dataset = None
        for edge_bases in itertools.product(["I", "X", "Y", "Z"], repeat=len(edges)):
            self.compute_graph.assign_bases_to_edges(edge_bases=edge_bases, edges=edges)
            summation_term = []
            cumulative_len = 1
            for subcircuit_idx in self.smart_order:
                subcircuit_entry_init_meas = self.compute_graph.get_init_meas(
                    subcircuit_idx=subcircuit_idx
                )
                subcircuit_entry_prob = self.subcircuit_entry_probs[subcircuit_idx][
                    subcircuit_entry_init_meas
                ]
                summation_term.append(subcircuit_entry_prob)
                cumulative_len *= len(subcircuit_entry_prob)
                self.overhead["multiplications"] += cumulative_len
            self.overhead["multiplications"] -= len(summation_term[0])
            dataset_elem = tf.data.Dataset.from_tensors(tuple(summation_term))
            if dataset is None:
                dataset = dataset_elem
            else:
                dataset = dataset.concatenate(dataset_elem)
        self.compute_graph.remove_bases_from_edges(edges=self.compute_graph.edges)
        dataset = dataset.batch(
            batch_size=1, num_parallel_calls=tf.data.AUTOTUNE, deterministic=False
        )
        self.times["make_dataset"] = perf_counter() - make_dataset_begin

        compute_begin = perf_counter()
        dataset = dataset.map(
            compute_summation_term,
            num_parallel_calls=tf.data.AUTOTUNE,
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
