from qiskit import QuantumCircuit, QuantumRegister, ClassicalRegister
import sys
import math
import numpy as np


class BV:
    """
    Generate an instance of the Bernstein-Vazirani algorithm.

    Attributes
    ----------
    secret : str
        the secret bitstring that BV will find with a single oracle query
    barriers : bool
        include barriers in the circuit
    measure : bool
        should a ClassicalRegister and measurements be added to the circuit
    regname : str
        optional string to name the quantum and classical registers. This
        allows for the easy concatenation of multiple QuantumCircuits.
    qr : QuantumRegister
        Qiskit QuantumRegister holding all of the quantum bits
    circ : QuantumCircuit
        Qiskit QuantumCircuit that represents the uccsd circuit
    """

    def __init__(self, secret=None, barriers=True, measure=False, regname=None):

        if secret is None:
            raise Exception('Provide a secret bitstring for the Bernstein-Vazirani circuit, example: 001101')
        else:
            if type(secret) is int:
                self.secret = str(secret)
            else:
                self.secret = secret

        # set flags for circuit generation
        self.nq = len(self.secret)
        self.measure = measure
        self.barriers = barriers

        # create a QuantumCircuit object with 1 extra qubit
        if regname is None:
            self.qr = QuantumRegister(self.nq+1)
        else:
            self.qr = QuantumRegister(self.nq+1, name=regname)
        self.circ = QuantumCircuit(self.qr)

        # add ClassicalRegister if measure is True
        if self.measure:
            self.cr = ClassicalRegister(self.nq)
            self.circ.add_register(self.cr)

        # add the extra ancilla qubit
        #self.anc = QuantumRegister(1, 'anc')
        #self.circ.add_register(self.anc)


    def gen_circuit(self):
        """
        Create a circuit implementing the Bernstein-Vazirani algorithm

        Returns
        -------
        QuantumCircuit
            QuantumCircuit object of size nq
        """

        # initialize ancilla in 1 state
        self.circ.x(self.qr[-1])

        # create initial superposition
        self.circ.h(self.qr)
        #self.circ.h(self.anc)

        # implement the black box oracle
        # for every bit that is 1 in the secret, place a CNOT gate
        # with control qr[i] and target anc[0]
        # (secret is little endian - index 0 is at the top of the circuit)
        for i, bit in enumerate(self.secret[::-1]):
            if bit is '1':
                self.circ.cx(self.qr[i], self.qr[-1])

        # add barriers
        if self.barriers:
            self.circ.barrier()

        # collapse superposition
        self.circ.h(self.qr)
        #self.circ.h(self.anc)

        # measure qubit register
        if self.measure:
            self.circ.measure(self.qr[:-1], self.cr)

        return self.circ



