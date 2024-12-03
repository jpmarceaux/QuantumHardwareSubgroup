import numpy as np, itertools
from termcolor import colored
from lib.color_compass import bacon_shor_group, pauli2vector
from scipy.special import comb


class Compass2DLattice():
    """
    convention: 
    X coords extend vertically |
    Z coords extend horizontally --
    store the coloring as a list with values in {-1, 0, 1}
    
    Red  ~ -1 ~ Z-type cuts
    Blue ~ +1 ~ X-type cuts
    White ~ 0
    
    preallocate logical X and L as cuts accross the lattice
    """
    def __init__(self, dimX, dimZ):
        self.dimX = dimX
        self.dimZ = dimZ
        self.colors = [0] * (dimX-1)*(dimZ-1)
        self.stabs = bacon_shor_group(dimX, dimZ)
        self.Lx = ''.join(['X']*dimX+['_']*dimX*(dimZ-1))
        self.Lz = ''.join((['Z']+['_']*(dimX-1))*dimZ)
        self.logicals = [self.Lx,self.Lz]
        
    def __str__(self):
        vertex_rows = []
        face_rows = []
        dimX = self.dimX
        dimZ = self.dimZ
        for i in range(dimX):
            vertex_string = ''
            for j in range(dimZ):
                vertex_string += str(i*dimZ + j).zfill(3)
                if (j != dimZ-1):
                    vertex_string += '---'
            vertex_rows.append(vertex_string)
                
        for i in range(dimX-1):
            face_string = ''
            for j in range(dimZ-1):
                if(self.colors[i*(dimZ-1) + j] == -1):
                    face_string += ' | '+colored(' # ', 'red')
                elif(self.colors[i*(dimZ-1) + j] == +1):
                    face_string += ' | '+colored(' # ', 'blue')
                elif(self.colors[i*(dimZ-1) + j] == 0):
                    face_string += ' |    '
                else:
                    raise ValueError(f'Invalid color type {self.colors[i*dimZ+j]}')
                if j == dimZ-2:
                    face_string += ' |'
            face_rows.append(face_string)
        sout = ''
        for idx, row in enumerate(vertex_rows):
            sout += row +'\n'
            if idx != len(vertex_rows)-1:
                sout += face_rows[idx]+'\n'
        return sout
    
    def size(self):
        return self.dimX*self.dimZ
        
    def getS(self):
        return self.stabs[0]+self.stabs[1]
    
    def getSx(self):
        return self.stabs[0]
    
    def getSz(self):
        return self.stabs[1]
    
    def getDims(self):
        return (self.dimX, self.dimZ)
    
    def max_stab_number(self):
        return self.dimX*self.dimZ - 1
    
    def pcheckZ(self):
        """returns the Z parity check matrix"""
        return np.vstack([pauli2vector(s) for s in self.getSz()])
        
    def pcheckX(self):
        """returns the X parity check matrix"""
        return np.vstack([pauli2vector(s) for s in self.getSx()])
    
    def parity_check_matrix(self):
        """returns the full parity check matrix"""
        return np.vstack([self.pcheckX(), self.pcheckZ()])
    
    def logicalcheck(self):
        """returns the logical check matrix"""
        return np.vstack([pauli2vector(s) for s in self.logicals])
    
    def display(self, pauli):
        dimX = self.dimX
        dimZ = self.dimZ
        if (len(pauli) != dimX*dimZ):
            raise ValueError("Pauli string dimension mismatch with lattice size")
        sout = ''
        slist = list(pauli)
        for i in range(dimX):
            for j in range(dimZ):
                if slist[i*dimZ+j] == 'X':
                    sout += ' X '
                elif slist[i*dimZ+j] == 'Z':
                    sout += ' Z '
                else:
                    sout += '   '
                if (j != dimZ-1):
                    sout += '---'
            if (i != dimX -1):
                sout += '\n'
                sout += ' |    '*dimZ
            sout += '\n'
        print(sout)
        
    def color_lattice(self, colors, verbose=False):
        """
        replace color state with input and recalculate stab and gauge groups 
        """
        dimX = self.dimX
        dimZ = self.dimZ
        if len(colors) != (dimX-1)*(dimZ-1):
            raise ValueError("Color dimension mismatch with lattice size")
        
        self.colors = colors
        colors = np.array(colors,dtype=np.int8).reshape(dimX-1, dimZ-1)

        stabsX_inds = []
        for row in range(dimX-1):
            qubits = [row*dimZ, row*dimZ + dimZ]
            for col in range(dimZ-1):
                if colors[row, col] == 1:
                    stabsX_inds.append(qubits)
                    qubits = []
                qubits.append(row*dimZ + (col+1))
                qubits.append(row*dimZ + (col+1) + dimZ)
            stabsX_inds.append(qubits)

        stabsZ_inds = []
        for col in range(dimZ-1):
            qubits = [col, col + 1]
            for row in range(dimX-1):
                if colors[row, col] == -1:
                    stabsZ_inds.append(qubits)
                    qubits = []
                qubits.append((row+1)*dimZ + col)
                qubits.append((row+1)*dimZ + col + 1)
            stabsZ_inds.append(qubits)

        self.stabs = [
            [''.join(['X' if i in qubits else '_' for i in range(dimX*dimZ)]) for qubits in stabsX_inds],
            [''.join(['Z' if i in qubits else '_' for i in range(dimX*dimZ)]) for qubits in stabsZ_inds]
        ]
    def make_surface(self):
        checkerboard = np.array([
            1 if (i + j) % 2 == 0 else -1 
            for i in range(self.dimX-1) for j in range(self.dimZ-1)
        ])
        self.color_lattice(checkerboard)

def generate_errors_by_weight(N, wmin, wmax):
    '''
    Generate all possible error strings of weight wmin to wmax on N qubits.
    '''
    paulis = ['X', 'Y', 'Z']
    error_strs = [] #['-'*N]
    error_vecs = [] #[[0]*N*2]
    for weight in range(wmin, wmax+1):
        for replacements in itertools.product(paulis, repeat=weight):
            for positions in itertools.combinations(range(N), weight):
                pstr = ['-'] * N
                for pos, rep in zip(positions, replacements):
                    pstr[pos] = rep
                error_strs.append(''.join(pstr))
                error_vecs.append(pauli2vector(pstr))
    return np.array(error_strs), np.array(error_vecs, dtype=np.uint8)

# def pstr_list_to_vecs(pstr_list):
# # as fast as pauli2vector function
#     return np.array([stim.PauliString(pstr).to_numpy() for pstr in pstr_list], 
#                     dtype=np.uint8).reshape(len(pstr_list),-1)

def get_joint_distribution(lat, error_vecs, error_probs, axes='LXZ'):
    '''
    Compute joint error distribution 
        P(L,X,Z) if axes = 'LXZ'
        P(L,Gamma) if axes = 'LG'

    Args:
        lat: 2D lattice for compass code
        error_vecs: relevelant error vectors
        error_probs: corresponding error probabilities
        axes: axes of the joint distribution
    '''

    # convert Pauli strings to vectors
    num_X, num_Z, num_L = len(lat.getSx()), len(lat.getSz()), len(lat.logicals)
    SXs = lat.pcheckX()
    SZs = lat.pcheckZ()
    logicals = lat.logicalcheck()
    Omega = np.kron(np.array([[0,1],[1,0]], dtype=np.uint8), np.eye(lat.size(), dtype=np.uint8))


    # compute the syndromes in their decimal values
    logical_checks = (error_vecs @ Omega @ logicals.T % 2) @ 2**np.arange(num_L)[::-1]
    X_checks = (error_vecs @ Omega @ SXs.T % 2) @ 2**np.arange(num_X)[::-1]
    Z_checks = (error_vecs @ Omega @ SZs.T % 2) @ 2**np.arange(num_Z)[::-1]

    # generate joint distribution matrix
    if axes == 'LXZ':
        joint_dist = np.zeros((2**num_L, 2**num_X, 2**num_Z))
        for i in range(joint_dist.shape[0]):
            for j in range(joint_dist.shape[1]):
                for k in range(joint_dist.shape[2]):
                    joint_dist[i,j,k] = error_probs[np.where((logical_checks == i) & (X_checks == j) & (Z_checks == k))[0]].sum()
    else:
        raise ValueError(f'Invalid axes {axes}')
    return joint_dist

def error_distribution_func_1q(pauli_string, rates):
    px = rates[0]
    py = rates[1]
    pz = rates[2]
    probs = {'I': 1-px-py-pz, '-': 1-px-py-pz, 'X': px, 'Y': py, 'Z': pz}
    # prob is the joint probability of the error string
    prob = 1
    for p in pauli_string:
        prob *= probs[p]
    return prob

def generate_errors_square_lattice(d, rates, error_set, min_probs = 1e-6):

    if error_set == 'all':
        wmax = d*d
    elif error_set == 'detectable':
        wmax = d-1
    elif error_set == 'most_probable':
        # wmax = round(np.log(min_rate)/np.log(max(rates)))
        N = d*d
        pmax = max(rates)
        pI = 1 - sum(rates)
        ws = np.arange(1,N+1)
        probs_by_weight = np.vectorize(comb)(N, ws) * pmax**ws * pI**(N-ws)
        wmax = np.where(probs_by_weight < min_probs)[0][0] + 1
    # print('wmax:', wmax)

    error_strs, error_vecs = generate_errors_by_weight(d*d, 0, wmax)
    error_probs = np.array([error_distribution_func_1q(p, rates) for p in error_strs])
    # print(len(error_strs), error_probs.sum())

    if error_set == 'most_probable' and wmax > d-1:
        # sort by probability
        sort_inds = np.argsort(error_probs)[::-1]
        error_strs = error_strs[sort_inds]
        error_vecs = error_vecs[sort_inds]
        error_probs = error_probs[sort_inds]

        # cut-off at 99.999% probability
        cumulative_probs = np.cumsum(error_probs)
        # cutoff = np.where(cumulative_probs >= 0.9999)[0]
        cutoff = np.where(cumulative_probs >= 1-pmax**2)[0]
        cutoff = cutoff[0]+1 if len(cutoff) > 0 else len(error_probs)
        error_strs = error_strs[:cutoff]
        error_vecs = error_vecs[:cutoff]
        error_probs = error_probs[:cutoff]
    
    return error_strs, error_vecs, error_probs

def compute_cond_entropy(joint_dist, axis):
    marginal_dist = np.where(joint_dist > 0, joint_dist.sum(axis=axis), 1)
    cond_dist = np.where(joint_dist > 0, joint_dist/marginal_dist, 1)
    return (-joint_dist*np.log2(cond_dist)).sum()
    # return np.where(joint_dist > 0, -joint_dist*np.log2(joint_dist/joint_dist.sum(axis)), 0).sum()

def compute_entropy(prob):
    return np.where(prob > 1e-10, -prob*np.log2(prob), 0).sum()

