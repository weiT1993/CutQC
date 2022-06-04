import itertools
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
    def __init__(self, compute_graph, dd_schedule, num_cuts) -> None:
        super().__init__()
        self.times = {}
        self.compute_graph = compute_graph
        self.num_cuts = num_cuts
        self.subcircuit_entry_lengths = {}
        self.pseudo_subcircuit_entry_probs = {}
        print("%d cuts" % self.num_cuts)
        self.reconstruction_length = 1
        for subcircuit_idx in dd_schedule["subcircuit_state"]:
            qubit_states = dd_schedule["subcircuit_state"][subcircuit_idx]
            num_active_qubits = qubit_states.count("active")
            length = int(2**num_active_qubits)
            self.reconstruction_length *= length
            print(
                "subcircuit %d : %d active qubits" % (subcircuit_idx, num_active_qubits)
            )
            self.subcircuit_entry_lengths[subcircuit_idx] = length
            self.pseudo_subcircuit_entry_probs[subcircuit_idx] = np.random.rand(
                length
            ).astype(np.float32)

        self.smart_order = sorted(
            self.subcircuit_entry_lengths.keys(),
            key=lambda subcircuit_idx: self.subcircuit_entry_lengths[subcircuit_idx],
        )
        self.overhead = {"additions": 0, "multiplications": 0}
        self.reconstructed_prob = self.compute()

    def compute(self):
        edges = self.compute_graph.get_edges(from_node=None, to_node=None)

        partial_compute_begin = perf_counter()
        reconstructed_prob = None
        counter = 0
        for edge_bases in itertools.product(["I", "X", "Y", "Z"], repeat=len(edges)):
            self.compute_graph.assign_bases_to_edges(edge_bases=edge_bases, edges=edges)
            summation_term = None
            for subcircuit_idx in self.smart_order:
                subcircuit_entry_prob = self.pseudo_subcircuit_entry_probs[
                    subcircuit_idx
                ]
                if summation_term is None:
                    summation_term = subcircuit_entry_prob
                else:
                    summation_term = tf.reshape(
                        tf.tensordot(summation_term, subcircuit_entry_prob, axes=0),
                        [-1],
                    )
                    self.overhead["multiplications"] += len(summation_term)
            if reconstructed_prob is None:
                reconstructed_prob = summation_term
            else:
                reconstructed_prob += summation_term
                self.overhead["additions"] += len(summation_term)
            counter += 1
            if counter == 4**3:
                break
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
