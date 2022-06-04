import sys
from qiskit import Aer, execute
from quantum_circuit_generator.generators import gen_hwea

n = 6
circ = gen_hwea(n,1)
print(circ)

simulator = Aer.get_backend('statevector_simulator')
result = execute(circ,simulator).result()
sv = result.get_statevector(circ)
print(sv)

# entanglement measure
def sgn_star(n,i):
    if n == 2:
        return 1
    
    i_binary = '{:b}'.format(i)
    Ni = 0
    for char in i_binary:
        if char == '1':
            Ni += 1
    
    if i >= 0 and i <= 2**(n-3)-1:
        return (-1)**Ni
    elif i >= 2**(n-3) and i <= 2**(n-2)-1:
        return (-1)**(n+Ni)
    else:
        print('i out of range for sgn*')
        sys.exit(2)


def tau(a,n):

    istar = 0
    for i in range(0,2**(n-2)):
        istar_term = sgn_star(n,i) * (a[2*i]*a[(2**n-1)-2*i] - a[2*i+1]*a[(2**n-2)-2*i])
        istar += istar_term

    return 2 * abs(istar)


print(tau(sv,n))
