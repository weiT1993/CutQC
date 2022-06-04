import random as rand

class Qbit:

    def __init__(self, index, prev_1q_gate):
        self.index = index
        self.prev_1q_gate = prev_1q_gate
        self.gate_dict = {'X':('Y','W'), 'Y':('X','W'), 'W': ('X','Y')}


    def random_gate(self):
        # Uniformly select a random single qubit gate
        if self.prev_1q_gate is None:
            coin_flip = rand.randint(0,2)
            gate_choices = ['X', 'Y', 'W']
        else:
            coin_flip = rand.randint(0,1)
            gate_choices = self.gate_dict[self.prev_1q_gate]
        self.prev_1q_gate = gate_choices[coin_flip]
        return gate_choices[coin_flip]

