from .Qbit_Sycamore import Qbit
from .ABCD_layer_generation import get_layers
from qiskit import QuantumCircuit, QuantumRegister, ClassicalRegister
import math
import sys
import numpy as np


class Qgrid:
    """
    Class to implement the quantum supremacy circuits as found
    in Arute, F., Arya, K., Babbush, R. et al. 'Quantum supremacy using a
    programmable superconducting processor'. Nature 574, 505–510 (2019)
    doi:10.1038/s41586-019-1666-5

    (https://www.nature.com/articles/s41586-019-1666-5)

    Each instance is a 2D array whose entries at Qbit objects.
    A supremacy circuit can be generated for a given instance
    by calling the gen_circuit() method.

    Attributes
    ----------
    n : int
        number of rows in the grid
    m : int
        number of columns in the grid
    d : int
        depth of the supremacy circuit (excludes measurement i.e. d+1)
    regname : str
        optional string to name the quantum and classical registers. This
        allows for the easy concatenation of multiple QuantumCircuits.
    qreg : QuantumRegister
        Qiskit QuantumRegister holding all of the qubits
    creg : ClassicalRegister
        Qiskit ClassicalRegister holding all of the classical bits
    circ : QuantumCircuit
        Qiskit QuantumCircuit that represents the supremacy circuit
    grid : array
        n x m array holding Qbit objects
    ABCD_layers : list
        List of qubit indices for 2-qubit gates for the A, B, C, and D layers of
        the supremacy circuit.
    order : list
        list of indices indicting the order the cz layers should be placed
    singlegates : bool
        Boolean indicating whether to include single qubit gates in the circuit
    """
    def __init__(self, n, m, d, order=None, singlegates=True,
                 barriers=True, measure=False, regname=None):
        self.n = n
        self.m = m
        self.d = d

        if regname is None:
            self.qreg = QuantumRegister(n*m)
            self.creg = ClassicalRegister(n*m)
        else:
            self.qreg = QuantumRegister(n*m, name=regname)
            self.creg = ClassicalRegister(n*m, name='c'+regname)
        # It is easier to interface with the circuit cutter
        # if there is no Classical Register added to the circuit
        self.measure = measure
        if self.measure:
            self.circ = QuantumCircuit(self.qreg, self.creg)
        else:
            self.circ = QuantumCircuit(self.qreg)

        self.grid = self.make_grid(n,m)
        self.ABCD_layers = get_layers(n,m)
        self.barriers = barriers
        self.singlegates = singlegates

        if order is None:
            # Use the default Google order for full supremacy circuit
            # In the Nature paper Supp Info. Table 3 shows that the
            # full supremacy circuit pattern is: ABCDCDAB
            # ABCD_layers = [layerA, layerB, layerC, layerD]
            self.order = [0,1,2,3,2,3,0,1]
        else:
            # Convert given order string to list of ints
            self.order = [int(c) for c in order]


    def make_grid(self,n,m):
        temp_grid = []
        index_ctr = 0
        for row in range(n):
            cur_row = []
            for col in range(m):
                cur_row += [Qbit(index_ctr,None)]
                index_ctr += 1
            temp_grid += [cur_row]

        return temp_grid


    def get_index(self, index1=None, index2=None):
        if index2 is None:
            return self.grid[index1[0]][index1[1]].index
        else:
            return self.grid[index1][index2].index


    def print_circuit(self):
        print(self.circ.draw(scale=0.6, output='text', reverse_bits=False))


    def save_circuit(self):
        str_order = [str(i) for i in self.order]
        if self.mirror:
            str_order.append('m')
        fn = 'supr_{}x{}x{}_order{}.txt'.format(self.n,self.m,self.d,"".join(str_order))
        self.circ.draw(scale=0.8, filename=fn, output='text', reverse_bits=False, 
                line_length=160)


    def measure_circuit(self):
        self.circ.barrier()
        for i in range(self.n):
            for j in range(self.m):
                qubit_index = self.get_index(i,j)
                self.circ.measure(self.qreg[qubit_index], self.creg[qubit_index])


    def apply_random_1q_gate(self, n, m):
        qb_index = self.get_index(n, m)
        gate = self.grid[n][m].random_gate()
        if gate is 'X':
            # Apply a sqrt-X gate to qubit at qb_index
            self.circ.rx(math.pi/2, self.qreg[qb_index])
        elif gate is 'Y':
            # Apply a sqrt-Y gate to qubit at qb_index
            self.circ.ry(math.pi/2, self.qreg[qb_index])
        elif gate is 'W':
            # Apply a sqrt-W gate to qubit at qb_index
            # W = (X + Y) / sqrt(2)
            self.circ.z(self.qreg[qb_index])
        else:
            Exception('ERROR: unrecognized gate: {}'.format(gate))


    def gen_circuit(self):
        print('Generating {}x{}, {}+1 supremacy circuit'.format(self.n,self.m,self.d))

        # Iterate through d layers
        for i in range(self.d):
            # apply single qubit gates
            for n in range(self.n):
                for m in range(self.m):
                    self.apply_random_1q_gate(n,m)

            # Apply entangling 2-qubit gates
            cur_q2s = self.ABCD_layers[self.order[i % len(self.order)]]
            for q2 in cur_q2s:
                ctrl = self.get_index(q2[0])
                trgt = self.get_index(q2[1])
                # The 2 qubit gate implemented by Google's Sycamore chip
                # is fSim(pi/2, pi/6) given in Eqn. 53 of the Nature Supp info
                self.circ.cz(self.qreg[ctrl],self.qreg[trgt])

            if self.barriers:
                self.circ.barrier()
        # End d layers

        # Measurement
        if self.measure:
            self.measure_circuit()

        return self.circ

    def gen_qasm(self):
        return self.circ.qasm()



