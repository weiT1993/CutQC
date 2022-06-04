import random as rand

class Qbit:

    def __init__(self, index, prev_1q_gate):
        self.index = index
        self.prev_1q_gate = prev_1q_gate
        self.gate_dict = {'T':('Y','X'), 'Y':('X','T'), 'X': ('T','Y')}

    def h(self):
        self.prev_1q_gate = 'H'
        return self.index

    def random_gate(self):
        # After a CZ-gate, randomly select X_1_2 or Y_1_2
        gate_choices = ['X','Y']
        coin_flip = rand.randint(0,1)
        self.prev_1q_gate = gate_choices[coin_flip]
        return gate_choices[coin_flip]

