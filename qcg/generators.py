from .Supremacy import Qgrid_original, Qgrid_Sycamore
from .QAOA import hw_efficient_ansatz
from .VQE import uccsd_ansatz
from .QFT import qft_circ
from .QWalk import quantum_walk
from .Dynamics import quantum_dynamics
from .BernsteinVazirani import bernstein_vazirani
from .Arithmetic import ripple_carry_adder

def gen_supremacy(height, width, depth, order=None, singlegates=True,
                  mirror=False, barriers=False, measure=False, regname=None):
    """
    Calling this function will create and return a quantum supremacy
    circuit based on the implementations in
    https://www.nature.com/articles/s41567-018-0124-x and
    https://github.com/sboixo/GRCS.
    """

    grid = Qgrid_original.Qgrid(height, width, depth, order=order,
                                mirror=mirror, singlegates=singlegates,
                                barriers=barriers, measure=measure,
                                regname=regname)

    circ = grid.gen_circuit()

    return circ


def gen_sycamore(height, width, depth, order=None, singlegates=True,
                  barriers=False, measure=False, regname=None):
    """
    Calling this function will create and return a quantum supremacy
    circuit as found in https://www.nature.com/articles/s41586-019-1666-5
    """

    grid = Qgrid_Sycamore.Qgrid(height, width, depth, order=order,
                                singlegates=singlegates, barriers=barriers,
                                measure=measure, regname=regname)

    circ = grid.gen_circuit()

    return circ


def gen_hwea(width, depth, parameters='optimal', seed=None, barriers=False,
             measure=False, regname=None):
    """
    Create a quantum circuit implementing a hardware efficient
    ansatz with the given width (number of qubits) and
    depth (number of repetitions of the basic ansatz).
    """

    hwea = hw_efficient_ansatz.HWEA(width, depth, parameters=parameters,
                                    seed=seed, barriers=barriers,
                                    measure=measure, regname=regname)

    circ = hwea.gen_circuit()

    return circ


def gen_uccsd(width, parameters='random', seed=None, barriers=False,
              regname=None):
    """
    Generate a UCCSD ansatz with the given width (number of qubits).
    """

    uccsd = uccsd_ansatz.UCCSD(width, parameters=parameters, seed=seed,
                               barriers=barriers, regname=regname)

    circ = uccsd.gen_circuit()

    return circ


def gen_qft(width, approximation_degree, inverse=False, kvals=False, barriers=True, measure=False,
            regname=None):
    """
    Generate a QFT (or iQFT) circuit with the given number of qubits
    """

    qft = qft_circ.QFT(width, approximation_degree, inverse=inverse, kvals=kvals, barriers=barriers,
                       measure=measure, regname=regname)

    circ = qft.gen_circuit()

    return circ


def gen_qwalk(n, barriers=True, regname=None):
    """
    Generate a quantum walk circuit with specified value of n
    """

    qwalk = quantum_walk.QWALK(n, barriers=barriers, regname=regname)

    circ = qwalk.gen_circuit()

    return circ


def gen_dynamics(H, barriers=True, measure=False, regname=None):
    """
    Generate a circuit to simulate the dynamics of a given Hamiltonian
    """

    dynamics = quantum_dynamics.Dynamics(H, barriers=barriers, measure=measure,
                                         regname=regname)

    circ = dynamics.gen_circuit()

    return circ


def gen_BV(secret=None, barriers=True,  measure=False, regname=None):
    """
    Generate an instance of the Bernstein-Vazirani algorithm which queries a
    black-box oracle once to discover the secret key in:

    f(x) = x . secret (mod 2)

    The user must specify the secret bitstring to use: e.g. 00111001
    (It can be given as a string or integer)
    """

    bv = bernstein_vazirani.BV(secret=secret, barriers=barriers,
                               measure=measure, regname=regname)

    circ = bv.gen_circuit()

    return circ


def gen_adder(nbits=None, a=0, b=0, use_toffoli=False, barriers=True,
              measure=False, regname=None):
    """
    Generate an n-bit ripple-carry adder which performs a+b and stores the
    result in the b register.

    Based on the implementation of: https://arxiv.org/abs/quant-ph/0410184v1
    """

    adder = ripple_carry_adder.RCAdder(nbits=nbits, a=a, b=b,
                use_toffoli=use_toffoli, barriers=barriers, measure=measure,
                regname=regname)

    circ = adder.gen_circuit()

    return circ





