import numpy as np
from termcolor import colored
from tqdm import tqdm
from lib.color_compass import bacon_shor_group, bsgauge_group, pauli2vector, pauli_weight, twisted_product


class Lattice2D():
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
        self.gauge = bsgauge_group(dimX, dimZ)
        self.Lx = ''.join(['X']*dimX+['_']*dimX*(dimZ-1))
        self.Lz = ''.join((['Z']+['_']*(dimX-1))*dimZ)
        self.logicals = [self.Lx,self.Lz]
        
    def size(self):
        return self.dimX*self.dimZ
        
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
    
    def getG(self):
        return self.gauge[0]+self.gauge[1]
    
    def getGx(self):
        return self.gauge[0]
    
    def getGz(self):
        return self.gauge[1]
    
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
        dimX = self.dimX-1
        dimZ = self.dimZ-1
        if(len(colors) != dimX*dimZ):
            raise ValueError("Color dimension mismatch with lattice size")
        
        self.stabs = bacon_shor_group(self.dimX, self.dimZ)
        self.gauge = bsgauge_group(self.dimX, self.dimZ)
        self.colors = colors
        
        
        for cidx, c in enumerate(colors):
            if c == -1:
                self.update_groups((int(np.floor(cidx/dimZ)), cidx%dimZ), -1)
            elif c == +1:
                self.update_groups((int(np.floor(cidx/dimZ)), cidx%dimZ), +1)
            if verbose:
                print(f'\nStep {cidx}:')
                print('X:', self.getSx())
                print('Z:', self.getSz())

        
    def update_groups(self, coords, cut_type):
        """
        cut the stabilizer group by coloring the face with the given type
            AND
        update the gauge group 
    
        algo: 
        [0] pick the gauge operator g to cut around
        [1] find s \in S that has weight-2 overlap with g
        [2] divide that s 
        [3] update the gauge group 
        """
        (i, j) = coords
        dimX = self.dimX
        dimZ = self.dimZ
        [Sx, Sz] = self.getSx(), self.getSz()
        [Gx, Gz] = self.getGx(), self.getGz()
        
        if cut_type == -1:
            # -1 = red which is a Z-cut
            g = ['_'] * dimX*dimZ
            g[i*dimZ + j] = 'Z'
            g[i*dimZ + j + 1] = 'Z'
            
            gvec = pauli2vector(''.join(g))
            
            # cut the relevant stabilizer
            for idx, s in enumerate(Sz):
                # find the overlapping stabilizer
                if pauli_weight(np.bitwise_xor(gvec, pauli2vector(s))) == pauli_weight(s) - 2:
                    # cut s into two vertical parts 
                    s1 = ['_'] * dimX*dimZ
                    s2 = ['_'] * dimX*dimZ
                    for k in range(0, i+1):
                        s1[k*dimZ + j] = s[k*dimZ + j]
                        s1[k*dimZ + j+1] = s[k*dimZ + j+1]
                    for k in range(i+1, dimX):
                        s2[k*dimZ + j] = s[k*dimZ + j]
                        s2[k*dimZ + j+1] = s[k*dimZ + j+1]
                    del Sz[idx]
                    Sz.append(''.join(s1))
                    Sz.append(''.join(s2))
                    break
            
            # make new gauge operator and update gauge group 
            gauge = ['_'] * dimX*dimZ
            for k in range(0, j+1):
                gauge[k + i*dimZ] = 'Z'
                gauge[k + i*dimZ + 1] = 'Z'
            Gx_new = []
            for g in Gx:
                if twisted_product(pauli2vector(''.join(g)), pauli2vector(''.join(gauge))) == 0:
                    Gx_new.append(g)
            Gx = Gx_new
                
        elif cut_type == +1:
            # +1 = blue that is a X-cut:
            g = ['_'] * dimX*dimZ
            g[i*dimZ + j] = 'X'
            g[(i+1)*dimZ + j ] = 'X'
            
            gvec = pauli2vector(''.join(g))
            
            # cut the relevant stabilizer
            for idx, s in enumerate(Sx):
                # find the overlapping stabilizer
                if pauli_weight(np.bitwise_xor(gvec, pauli2vector(s))) == pauli_weight(s) - 2:
                    # cut s into two horizontal parts 
                    s1 = ['_'] * dimX*dimZ
                    s2 = ['_'] * dimX*dimZ
                    for k in range(0, j+1):
                        s1[i*dimZ + k] = s[i*dimZ + k]
                        s1[(i+1)*dimZ + k] = s[(i+1)*dimZ + k]
                    for k in range(j+1, dimZ):
                        s2[i*dimZ + k] = s[i*dimZ + k]
                        s2[(i+1)*dimZ + k] = s[(i+1)*dimZ + k]
                    del Sx[idx]
                    Sx.append(''.join(s1))
                    Sx.append(''.join(s2))
                    break
            
            # make new gauge operator and update gauge group 
            gauge = ['_'] * dimX*dimZ
            for k in range(0, j+1):
                gauge[k + i*dimZ] = 'X'
                gauge[k + (i+1)*dimZ] = 'X'
            Gz_new = []
            for g in Gz:
                if twisted_product(pauli2vector(''.join(g)), pauli2vector(''.join(gauge))) == 0:
                    Gz_new.append(g)
            Gz = Gz_new

        # update the groups
        self.stabs = [Sx, Sz]
        self.gauge = [Gx, Gz]
        
def make_surface_code_lattice(dimX,dimZ,start='X', verbose=False):
    if start == 'X':
        first_two_cells = [1,-1]
    elif start == 'Z':
        first_two_cells = [-1,1]
    first_row = (first_two_cells*int(np.ceil(dimZ/2)))
    first_two_rows = first_row[:dimZ-1] + first_row[1:dimZ]
    whole_lattice = first_two_rows*int(np.ceil(dimX/2))
    whole_lattice = whole_lattice[:(dimZ-1)*(dimX-1)]
    print(whole_lattice)
    lat = Lattice2D(dimX, dimZ)
    if verbose:
        print('X:', lat.getSx())
        print('Z:', lat.getSz())
    lat.color_lattice(whole_lattice, verbose)
    return lat

def generate_all_pauli_strings(n):
    """
    generate all possible n-qubit Pauli strings
    """
    if n == 1:
        return ['I', 'X', 'Y', 'Z']
    else:
        return [p + q for p in generate_all_pauli_strings(1) for q in generate_all_pauli_strings(n-1)]
    
def index_to_pauli_string(idx, n):
    """
    convert an index to a Pauli string by converting the index to quaternary
    """
    return np.base_repr(idx, base=4).zfill(n).replace('0', 'I').replace('1', 'X').replace('2', 'Y').replace('3', 'Z')

def pauli_to_error_class(pauli, logicals, Sx, Sz):
    """
    convert a Pauli string to an error class representative
    """
    pauli_vec = pauli2vector(pauli)
    lX_vec, lZ_vec = [pauli2vector(logicals[0]), pauli2vector(logicals[1])]
    sX_vec = [pauli2vector(s) for s in Sx]
    sZ_vec = [pauli2vector(s) for s in Sz]

    logical_checks = [twisted_product(pauli_vec, lX_vec), twisted_product(pauli_vec, lZ_vec)]
    Sx_checks = [twisted_product(pauli_vec, s) for s in sX_vec]
    Sz_checks = [twisted_product(pauli_vec, s) for s in sZ_vec]
    return logical_checks, Sx_checks, Sz_checks


def error_partition_from_lattice(pauli, lat):
    """
    given a Pauli string and a lattice, return the error partition
    """
    logical_checks, Sx_checks, Sz_checks = pauli_to_error_class(pauli, [lat.Lx, lat.Lz], lat.getSx(), lat.getSz())
    return logical_checks, Sx_checks, Sz_checks

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

def binary_tuple_to_decimal_idx(t):
    # return sum([2**i for i in range(len(t)) if t[i] == 1])
    return t @ 2**np.arange(len(t))[::-1]

def make_joint_distribution(lat, rates, auto_break=False):
    """
    Calculates P(L, \Gamma) and P(L | \Gamma = \gamma) for a given lattice and error rates
    """
    Sx = lat.getSx()
    Sz = lat.getSz()
    # Lx = lat.Lx
    # Lz = lat.Lz
    joint_distribution = np.zeros((4, 2**len(Sx), 2**len(Sz)))
    pauli_index_range = range(4**lat.size())
    for pauli_idx in tqdm(pauli_index_range):
        pauli = index_to_pauli_string(pauli_idx, lat.size())
        logical_checks, Sx_checks, Sz_checks = error_partition_from_lattice(pauli, lat)
        prob = error_distribution_func_1q(pauli, rates)

        syndrome_Sx = tuple(Sx_checks)
        syndrome_Sz = tuple(Sz_checks)
        syndrome_logical = tuple(logical_checks)

        Sx_idx = binary_tuple_to_decimal_idx(syndrome_Sx)
        Sz_idx = binary_tuple_to_decimal_idx(syndrome_Sz)
        logical_idx = binary_tuple_to_decimal_idx(syndrome_logical)

        joint_distribution[logical_idx, Sx_idx, Sz_idx] += prob

        if auto_break and pauli_idx > int(5e4):
            return 0
        
    return joint_distribution

def calculate_logical_marginal(joint_distribution):
    """
    Calculates P(L) for a given joint distribution P(L, Sx, Sz)
    """
    return np.sum(joint_distribution, axis=(1,2))
   
def calculate_syndrome_marginal(joint_distribution):
    """
    Calculates P(Sx, Sz) for a given joint distribution P(L, Sx, Sz)
    """
    return np.sum(joint_distribution, axis=0)

def calculate_syndrome_given_logical(joint_distribution):
    """
    Calculates P(Sx, Sz | L) for a given joint distribution P(L, Sx, Sz)
    """
    return joint_distribution/np.sum(joint_distribution, axis=0)

def calculate_logical_prob_given_syndrome(joint_distribution, sx_idx, sz_idx):
    """
    Calculates P(L | Sx, Sz) for a given joint distribution P(L, Sx, Sz)
    """
    return joint_distribution[:, sx_idx, sz_idx]/np.sum(joint_distribution[:, sx_idx, sz_idx])

def calculate_conditional_entropy_logical_on_stabilizer(joint_distribution):
    """
    Calculates H(L | Sx, Sz) for a given joint distribution P(L, Sx, Sz)
    """
    conditional_entropy = 0
    for l_idx in range(joint_distribution.shape[0]):
        for sx_idx in range(joint_distribution.shape[1]):
            for sz_idx in range(joint_distribution.shape[2]):
                p_stab = np.sum(joint_distribution[:, sx_idx, sz_idx])
                p_joint = joint_distribution[l_idx, sx_idx, sz_idx]
                if abs(p_joint) > 1e-10:
                    conditional_entropy += p_joint*np.log2(p_joint/p_stab)
    return -conditional_entropy

def calculate_logical_entropy(joint_distribution):
    """
    Calculates H(L) for a given joint distribution P(L, Sx, Sz)
    """
    l_marginal = calculate_logical_marginal(joint_distribution)
    return -np.sum([p*np.log2(p) for p in l_marginal if p > 0])

