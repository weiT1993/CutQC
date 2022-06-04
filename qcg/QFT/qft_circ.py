import sys
import math
import numpy as np
from qiskit import QuantumCircuit, QuantumRegister, ClassicalRegister


class QFT:
    """
    Class which generates the circuit to perform the Quantum Fourier
    Transform (or its inverse) as described in Mike & Ike Chapter 5.

    (Michael A Nielsen and Isaac L Chuang. Quantum computation and quantum
     information (10th anniv. version), 2010.)

    For another example see Figure 1 of Daniel E Browne 2007 New J. Phys. 9 146

    A QFT or iQFT circuit can be generated with a given instance of the
    QFT class by calling the gen_circuit() method.

    Attributes
    ----------
    width : int
        number of qubits
    inverse : bool
        Set to true to generate the inverse quantum fourier transform
    kvals : bool
        optional parameter that will change the angle of the controlled
        rotations so that when the circuit is printed it will display
        the same k values that are shown in Mike & Ike Chpt 5, Fig 5.1
        (NOTE: the generated circuit will no longer be valid! This is
         for visualization purposes only.)
    barriers : bool
        should barriers be included in the generated circuit
    measure : bool
        should a classical register & measurement be added to the circuit
    regname : str
        optional string to name the quantum and classical registers. This
        allows for the easy concatenation of multiple QuantumCircuits.
    qr : QuantumRegister
        Qiskit QuantumRegister holding all of the quantum bits
    cr : ClassicalRegister
        Qiskit ClassicalRegister holding all of the classical bits
    circ : QuantumCircuit
        Qiskit QuantumCircuit that represents the uccsd circuit
    """

    def __init__(
        self,
        width,
        approximation_degree,
        inverse=False,
        kvals=False,
        barriers=True,
        measure=False,
        regname=None,
    ):

        # number of qubits
        self.nq = width
        self.approximation_degree = approximation_degree

        # set flags for circuit generation
        self.inverse = inverse
        self.kvals = kvals
        self.barriers = barriers
        self.measure = measure

        # create a QuantumCircuit object
        if regname is None:
            self.qr = QuantumRegister(self.nq)
            self.cr = ClassicalRegister(self.nq)
        else:
            self.qr = QuantumRegister(self.nq, name=regname)
            self.cr = ClassicalRegister(self.nq, name="c" + regname)
        # Have the option to include measurement if desired
        if self.measure:
            self.circ = QuantumCircuit(self.qr, self.cr)
        else:
            self.circ = QuantumCircuit(self.qr)

    def inv_qft(self):
        """
        Implement the inverse QFT on self.circ

        j ranges from nq-1 -> 0
        k ranges from nq-1 -> j+1

        For each j qubit, a controlled cu1 gate is applied with target=j,
        control=k (for each k).

        cu1 = 1  0
              0  e^(-2pi*i / 2^(k-j+1))
        """
        for j in range(self.nq - 1, -1, -1):
            for k in range(self.nq - 1, j, -1):
                if self.kvals:
                    self.circ.cu1(-1 * (k - j + 1), self.qr[k], self.qr[j])
                else:
                    self.circ.cu1(
                        -1 * (2 * np.pi) / (2 ** (k - j + 1)), self.qr[k], self.qr[j]
                    )
            self.circ.h(self.qr[j])

            if self.barriers:
                self.circ.barrier()

    def reg_qft(self):
        """
        Implement the QFT on self.circ

        j ranges from 0   -> nq-1
        k ranges from j+1 -> nq-1

        For each j qubit, a controlled cu1 gate is applied with target=j,
        control=k (for each k).

        cu1 = 1  0
              0  e^(2pi*i / 2^(k-j+1))
        """
        for j in range(self.nq):
            self.circ.h(self.qr[j])
            for k in range(j + 1, self.nq):
                if self.kvals:
                    self.circ.cu1(k - j + 1, self.qr[k], self.qr[j])
                else:
                    if k - j + 1 <= self.approximation_degree:
                        self.circ.cu1(
                            (2 * np.pi) / (2 ** (k - j + 1)), self.qr[k], self.qr[j]
                        )

            if self.barriers:
                self.circ.barrier()

    def gen_circuit(self):
        """
        Create a circuit implementing the UCCSD ansatz

        Given the number of qubits and parameters, construct the
        ansatz as given in Whitfield et al.

        Returns
        -------
        QuantumCircuit
            QuantumCircuit object of size nq with no ClassicalRegister and
            no measurements
        """

        if self.inverse:
            self.inv_qft()
        else:
            self.reg_qft()

        if self.measure:
            self.circ.barrier()
            self.circ.measure(self.qr, self.cr)

        return self.circ
