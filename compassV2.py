import numpy as np
from termcolor import colored
import stim
import itertools as itt
import random
import warnings
import random
from pymatching import Matching


    
    
def stabs_to_mat(stabs):
    """
    Convert stabs to binary parity check matrix
    
    assumes the stabs have CSS structure, with X- and Z-type separated
    """
    mat = np.array([[1 if i != 'I' else 0 for i in s] for s in stabs])
    return mat
        
def correction_to_string(vec, correction_type):
    """
    Convert binary vector to a Pauli string
    """
    op = ''.join([correction_type if i==1 else 'I' for i in vec])
    return op

def pauli_weight(pauli):
    """Get weight of pauli operator"""
    if type(pauli) is list:
        pvec = np.array(pauli)
    elif type(pauli) is str:
        pvec = pauli2vector(pauli)
    else:
        pvec = pauli
    return np.sum(np.bitwise_or(pvec[:int(len(pvec)/2)], pvec[int(len(pvec)/2):]))

def pauli2binary(pstr):
    """
    convert pstr to a binary vector
    """
    bstr = [0]*2*len(pstr)
    for idx, c in enumerate(pstr):
        if c == 'X':
            bstr[idx] = 1
        elif c == 'Z':
            bstr[idx+len(pstr)] = 1
        elif c == 'Y':
            bstr[idx] = 1
            bstr[idx+len(pstr)] = 1
    return bstr

def pauli2vector(pstr):
    """
    convert pstr to a binary vector
    """
    bstr = [0]*2*len(pstr)
    for idx, c in enumerate(pstr):
        if c == 'X':
            bstr[idx] = 1
        elif c == 'Z':
            bstr[idx+len(pstr)] = 1
        elif c == 'Y':
            bstr[idx] = 1
            bstr[idx+len(pstr)] = 1
    return np.array(bstr)

def binary2pauli(blst):
    """
    convert binary list to Pauli string
    """
    L = int(len(blst)/2)
    pstr = []
    for l in range(L):
        if blst[l] == 0 and blst[l+L] == 0:
            pstr.append('I')
        elif blst[l] == 1 and blst[l+L] == 0:
            pstr.append('X')
        elif blst[l] == 0 and blst[l+L] == 1:
            pstr.append('Z')
        elif blst[l] == 1 and blst[l+L] == 1:
            pstr.append('Y')
    return ''.join(pstr)

def twisted_product(stab_binary, pauli_binary):
    """
    take twisted product of stabilizer with pauli to calculate commutator 
    """
    L = int(len(stab_binary)/2)
    return (stab_binary[:L]@pauli_binary[L:] + stab_binary[L:]@pauli_binary[:L]) % 2

def parity_check(stabs, pauli):
    if len(stabs) == 0:
        return np.array([0])
    if type(pauli[0]) is str:
        bvec = pauli2vector(pauli)
    else: 
        bvec = pauli
    if type(stabs[0]) is str:
        return np.array([twisted_product(pauli2vector(s), bvec) for s in stabs])
    else:
        return np.array([twisted_product(s, bvec) for s in stabs])
    
def pauli_group(qubits):
    """
    generate Pauli group in n-qubits in strings
    """
    return [''.join(p) for p in itt.product('IXYZ', repeat=qubits)]

def paulis_weighted(qubits, weight):
    """
    generate all Pauli operators of weight <= weight
    """
    gen = [('I',)*qubits]
    for i in range(1, weight+1):
        gen = itt.chain(gen, itt.product('XYZ', repeat=i))
    basic_strings = [g+('I',)*(qubits-len(g)) for g in gen]
    paulis = set()
    for b in basic_strings:
        for p in itt.permutations(b):
            paulis.add(''.join(p))
    return paulis

def pauli_band(qubits, weight):
    """
    generated all Pauli operators of weight == weight
    """
    assert weight < qubits, "invalid band"
    basic_strings = [g+('I',)*(qubits-weight) for g in itt.combinations_with_replacement('XYZ', weight)]
    paulis = set()
    for b in basic_strings:
        for p in itt.permutations(b):
            paulis.add(''.join(p))
    return paulis

def pauli2tableau(pstr):
    tout = stim.Tableau(len(pstr))
    for idx, p in enumerate(pstr):
        tout.append(stim.Tableau.from_named_gate(p), [idx])
    return tout

def sample_globalbias(qubits, rx, ry, rz, N):
    """
    sample error distribution with rx, ry, rz dephasing
    """
    assert rx+ry+rz <= 1, "dephasing rates > 1"
    hist = dict()
    for n in range(N):
        e = []
        for q in range(qubits):
            s = random.uniform(0, 1)
            if 0 <= s < rx:
                e.append('X')
            elif rx <= s < rx+ry:
                e.append('Y')
            elif rx+ry <= s < rx+ry+rz:
                e.append('Z')
            else:
                e.append('I')
        estr = ''.join(e)
        if estr not in hist:
            hist[estr] = 1/N
        else:
            hist[estr] += 1/N
    return hist

def check_distribution_globalbias(qubits, wt_min, wt_max, rx, ry, rz, N):
    """
    sample from global biased noise model and
    partition into a group of error checks along with rates below and above wt_min and wt_max
    """
    assert rx+ry+rz <= 1, "dephasing rates > 1"
    hist = dict()
    amp_below = 0
    amp_above = 0
    for n in range(N):
        e = []
        for q in range(qubits):
            s = random.uniform(0, 1)
            if 0 <= s < rx:
                e.append('X')
            elif rx <= s < rx+ry:
                e.append('Y')
            elif rx+ry <= s < rx+ry+rz:
                e.append('Z')
            else:
                e.append('I')
        estr = ''.join(e)
        wt = pauli_weight(estr)
        if (wt_min <= wt) and (wt <= wt_max):
            if estr not in hist:
                hist[estr] = 1/N
            else:
                hist[estr] += 1/N
        elif wt < wt_min:
            amp_below += 1/N
        elif wt > wt_max: 
            amp_above += 1/N
    return [hist, amp_below, amp_above]
