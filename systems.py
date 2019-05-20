import numpy as np
#  import math
#  from scipy.linalg import expm
from scipy import sparse
from scipy.sparse.linalg import expm
from math import sqrt, pi
#  import time

sx = np.array([[0, 1], [1, 0]])
sy = np.array([[0, -1j], [1j, 0]])
sz = np.array([[1, 0], [0, -1]])
id2 = np.array([[1, 0], [0, 1]])
sp = np.array([[0, 1], [0, 0]])
sm = np.array([[0, 0], [1, 0]])


def tensor_prod(*arg):
    """
    tensor_prod(a1, a2) = np.kron(a1, a2).
    tensor_prod(a1, a2, ..., an) = np.kron(tensor_prod(a1, ..., an-1), an)
    """
    res = arg[0]
    for i in range(1, len(arg)):
        res = np.kron(res, arg[i])
    return res


def onesite_sigmaop(n_sites, position, spin_direction):
    """spin_direction = 'x', 'y', or 'z' """
    pauli_m = {'x': sx, 'y': sy, 'z': sz}[spin_direction]
    arg = (pauli_m if i == position else id2 for i in range(n_sites))
    return tensor_prod(*arg)


def twosite_sigmaop(n_sites, pos1, pos2, op1, op2):
    """op1, op2= 'x', 'y', 'z', 'p', or 'm' """
    s1, s2 = ({'x': sx, 'y': sy, 'z': sz, 'p': sp, 'm': sm}[d] for d in (op1,
                                                                         op2))
    arg = (s1 if i == pos1 else s2 if i == pos2 else id2 for i in
           range(n_sites))
    return tensor_prod(*arg)


class Gate(object):
    def __init__(self, hamiltonian=None, gate=None, symbol=None):
        if hamiltonian is not None and gate is not None:
            raise ValueError('hamiltonian and gate cannot be both specified.')
        if hamiltonian is None:
            self.gate = sparse.csc_matrix(gate)
            self.ham = None
        else:
            self.ham = sparse.csc_matrix(hamiltonian)
            self.gate = sparse.linalg.expm(-1j * self.ham)
            #  self.gate = expm(-1j * hamiltonian)
            #  self.ham = hamiltonian

        self.symbol = symbol

    def __matmul__(self, operator):
        if type(operator) is Gate:
            return Gate(gate=self.gate @ operator.gate)
        else:
            return self.gate @ operator

    def __rmatmul__(self, operator):
        if type(operator) is Gate:
            return Gate(gate=operator.gate @ self.gate)
        return operator @ self.gate

    def __repr__(self):
        if self.symbol is not None:
            return self.symbol
        elif self.ham is not None:
            return f'e^(-i \n{self.ham}\n)'
        else:
            return f'{self.gate}'

    def __rmul__(self, i):
        return i*self.gate


class System(object):
    pass


class SpinSystem(System):
    """
    A Spin System with one-qubit gates (onequbit_gate), two-quibit gates
    (twoqubit_gate) and methods to generate spin states and evolve them.
    The exact spin Hamiltonian is left unspecified.
    """
    def __init__(self, n_sites, ham_params, bc='periodic', store_gates=True,
                 **other_parameters):
        self.n_sites = n_sites
        self.dim = 2**n_sites
        self.shape = (2**n_sites, 2**n_sites)
        self.J = ham_params['J']
        self.ham_params = ham_params
        self.init_hamiltonian(bc, **ham_params)
        self.state = np.zeros(self.dim)
        self.store_gates = store_gates
        print("store_gates = ", self.store_gates)
        if self.store_gates:
            self.storage_one = {}
            self.storage_all = {}

    def onequbit_gate(self, site, parameter, kind='sz'):
        """onequbit_gate(i, a, kind) = exp(-i*a*h_i) for some one-site
        Hamiltonians at site i"""
        args = (site, parameter, kind)
        if self.store_gates and args in self.storage_one:
            return self.storage_one[args]
        else:
            if kind == 'sz':
                ham = parameter*onesite_sigmaop(self.n_sites, site, 'z')
                gate = Gate(ham,
                            symbol=f'{kind}_{site}({parameter/pi:.2f}*pi)')
                if self.store_gates:
                    self.storage_one[args] = gate
                return gate

            if kind == 'sx':
                ham = parameter*onesite_sigmaop(self.n_sites, site, 'x')
                gate = Gate(ham,
                            symbol=f'{kind}_{site}({parameter/pi:.2f}*pi)')
                if self.store_gates:
                    self.storage_one[args] = gate
                return gate
            if kind == 'sy':
                ham = parameter*onesite_sigmaop(self.n_sites, site, 'y')
                gate = Gate(ham,
                            symbol=f'{kind}_{site}({parameter/pi:.2f}*pi)')
                if self.store_gates:
                    self.storage_one[args] = gate
                return gate
            else:
                raise ValueError('This type of gate is not implemented.')

    def twoqubit_gate(self, sites, parameter, kind='sxsx'):
        """twoqubit_gate((i, j), a, kind) = exp(-i*a*h_ij) for some two-site
        Hamiltonians at sites i and j"""
        site1, site2 = sites
        if kind == 'sxsx':
            ham = parameter * twosite_sigmaop(self.n_sites, site1, site2, 'x',
                                              'x')
            return Gate(ham, symbol=f'sx_{site1} sx_{site2}({parameter:.2f})')
        else:
            raise ValueError('This type of gate is not implemented.')

    def allqubit_gate(self, parameter, kind='sxsx'):
        """allqubit_gate(a, kind) = exp(-i*a*h) for some all to all (e.g. s^x_i
        s^x_j) coupling Hamiltonians"""
        args = (parameter, kind)
        if self.store_gates and args in self.storage_all:
            return self.storage_all[args]
        ham = np.zeros(self.shape)
        if kind == 'sxsx':
            for site1 in range(self.n_sites):
                for site2 in range(site1+1, self.n_sites):
                    ham += twosite_sigmaop(self.n_sites, site1, site2,
                                           'x', 'x')
            gate = Gate(parameter*ham, symbol=f'{kind}_all({parameter:.2f})')
            if self.store_gates:
                self.storage_all[args] = gate
            return gate
        else:
            raise ValueError('This type of gate is not implemented.')

    def init_hamiltonian(self, bc):
        raise NotImplementedError

    def evolve(self, time=None, unitary=None, inplace=False,
               starting_state=None):
        if unitary is None and time is None:
            raise ValueError('Indicate either the unitary or the time.')
        elif unitary is not None and time is not None:
            raise ValueError('Indicate either the unitary or the time, '
                             'not both.')
        if starting_state is None:
            starting_state = self.state
        else:
            if inplace:
                raise ValueError(
                    'Using evolve inplace with starting_state != self.state.'
                )
        if unitary is None:
            unitary = expm(-1j*time*self.hamiltonian)
        evolved_state = unitary @ starting_state
        if inplace:
            self.state = evolved_state
            return None
        else:
            return evolved_state

    def apply_gates(self, gates, inplace=False):
        evolved_state = self.state
        #  print('Time to apply gate =')
        #  start = time.time()
        for g in gates:
            #  print('sparsity of gate = ',
            #        np.count_nonzero(g.gate)/(g.gate.size))
            evolved_state = g @ evolved_state
        #  end = time.time()
        #  print(end - start)
        if inplace:
            self.state = evolved_state
            return None
        else:
            return evolved_state

    def random_state(self, product_state=True, seed=None, inplace=False):
        """Returns a random state in the Hilbert space. If inplace, self.state
        is modified instead."""
        if seed is not None:
            np.random.seed(seed)

        if product_state:
            #  randomly directed vector on the unit sphere
            #  f(theta)g(phi)dtheta dphi = sin(theta)/4pi
            #  g(phi) = 1/2pi, f(theta) = sin(theta)/2
            #  -> F(theta) = (1-cos(theta))/2
            #  F^-1(u) = arccos(1 - 2u)
            phis = [2*pi*np.random.rand() for _ in range(self.n_sites)]
            thetas = [np.arccos(1-2*np.random.rand()) for _ in
                      range(self.n_sites)]
            spinors = [np.array([np.cos(theta/2),
                                 np.exp(1j*phi)*np.sin(theta/2)])
                       for phi, theta in zip(phis, thetas)]
            rstate = tensor_prod(*spinors)
        else:
            vec = np.random.normal(0, 1, self.dim)
            mag = sqrt(sum(x**2 for x in vec))
            rstate = np.array([x/mag for x in vec])
        if inplace:
            self.state = rstate
            return None
        else:
            return rstate

    def ferro_state(self, inplace=False):
        """Returns the ferromagnetic state of the Hilbert space. If inplace,
        self.state is modified instead"""
        #  vec = tensor_prod(*((1,0) for _ in range(self.n_sites)))
        #  mag = sqrt(sum(x**2 for x in vec))
        #  fstate = np.array([x/mag for x in vec])
        fstate = np.zeros(self.dim)
        fstate[0] = 1.0
        if inplace:
            self.state = fstate
            return None
        else:
            return fstate

    def antiferro_state(self, inplace=False):
        """Returns the antiferromagnetic state of the Hilbert space. If
        inplace, self.state is modified instead"""
        spins = ((1, 0) if _ % 2 == 0 else (0, 1) for _ in range(self.n_sites))
        vec = tensor_prod(*spins)
        mag = sqrt(sum(x**2 for x in vec))
        afstate = np.array([x/mag for x in vec])
        afstate = np.zeros(self.dim)
        afstate[0] = 1.0
        if inplace:
            self.state = afstate
            return None
        else:
            return afstate

    def compare_states(self, state1, state2=None):
        if state2 is not None:
            return compare(state1, state2)
        else:
            return compare(self.state, state1)


class LongRangeIsing(SpinSystem):
    """
    H = H_X + H_Z
    H_X = 1/N sum_{l<m} s_l^x s_m^x + g sum_l s_l^x
    H_Z = h sum_l s_l^z
    """
    def init_hamiltonian(self, bc, J, g, h):
        ham = np.zeros(self.shape)
        for site1 in range(self.n_sites):
            ham += h * onesite_sigmaop(self.n_sites, site1, 'z')
            ham += g * onesite_sigmaop(self.n_sites, site1, 'x')
            for site2 in range(site1+1, self.n_sites):
                ham += J * \
                    twosite_sigmaop(self.n_sites, site1, site2, 'x', 'x')
        if bc == 'periodic':
            pass
        elif bc != 'open':
            raise ValueError('Boundary conditions not defined.')
        self.hamiltonian = sparse.csc_matrix(ham)


class SpSm(SpinSystem):
    """
    A spin system with a nearest-neighbor SpSm + H.c. (target) Hamiltonian.
    The system needs the n_sites (numer of site) argument to be defined.
    """
    def init_hamiltonian(self, bc, J):
        """Contruct the sum_i Sp_i Sm_{i+1} + H.c Hamiltonian"""
        ham = np.zeros(self.shape)
        for site in range(self.n_sites - 1):
            ham += twosite_sigmaop(self.n_sites, site, site+1, 'p', 'm')
        if bc == 'periodic':
            if self.n_sites > 2:
                ham += twosite_sigmaop(self.n_sites, self.n_sites-1, 0, 'p',
                                       'm')
        elif bc != 'open':
            raise ValueError('Boundary conditions not defined.')
        ham += ham.conj().T
        self.hamiltonian = J * sparse.csc_matrix(ham)


def abs2(x):
    return x.real**2 + x.imag**2


def compare(state1, state2):
    return abs2(np.vdot(state1, state2))


#  def system_3sites():
#      ham = tensor_prod(sp, sm, id2) + tensor_prod(id2, sp, sm)
#      ham += ham.conj().T

#      hs = (
#          tensor_prod(sx, sx, id2),
#          tensor_prod(id2, sx, sx),
#          tensor_prod(sz, id2, id2),
#          tensor_prod(id2, sz, id2),
#          tensor_prod(id2, id2, sz),
#          tensor_prod(sz, id2, id2),
#          tensor_prod(id2, sz, id2),
#          tensor_prod(id2, id2, sz)
#      )
#      alphas = (0.5, 0.5, -pi/4, -pi/4, -pi/4, pi/4, pi/4, pi/4)
#      gates = [expm(-1j*a*h) for a, h in zip(alphas, hs)]
#      return {'ham': ham, 'gates': gates}


#  def system_2sites():
#      ham = np.kron(sp, sm) + np.kron(sm, sp)
#      hs = (
#          np.kron(sx, sx),
#          np.kron(sz, id2) + np.kron(id2, sz),
#          np.kron(sz, id2) + np.kron(id2, sz)
#      )
#      alphas = (0.5, -pi/4, pi/4)
#      gates = [expm(-1j*a*h) for a, h in zip(alphas, hs)]
#      return {'ham': ham, 'gates': gates}
