import numpy as np
import pymatching, stim
# from lib.tools import binary2pauli

def binary2pauli(mat: np.ndarray):
    '''
    n = # of qubits
    r x 2n binary matrix -> list of r pauli strings
    '''
    out = []
    n = mat.shape[1] // 2
    for i in range(mat.shape[0]):
        ps = ''
        for j in range(n):
            k = 2 * mat[i, j + n] + mat[i, j]
            ps += '_XZY'[k]
        out.append(ps)
    return out

def checkmatrix(stabilizers,single_type=False):
    if single_type:
        matrix = np.zeros([len(stabilizers),len(stabilizers[0])],dtype=int)
        for i,stab in enumerate(stabilizers):
            for j,pauli in enumerate(stab):
                if pauli != '_':
                    matrix[i,j] = 1
    else:
        raise NotImplementedError('To add for non-CSS code')
    return matrix

def pL_from_checkmatrix(l_op,checkmatrix,rounds,syndromes,observables,CSS_error=None):
    M = pymatching.Matching(checkmatrix,repetitions=rounds)
    recovery_logical_acommutes = []
    for synd in syndromes:
        c = M.decode(synd)
        if CSS_error:
            R = stim.PauliString(''.join([CSS_error if i == 1 else '_' for i in c]))
        else:
            c = np.hstack([c[c.size//2:],c[:c.size//2]])
            R = stim.PauliString(binary2pauli(c[None,:])[0])
        recovery_logical_acommutes.append(1-int(R.commutes(l_op)))
    pL = ((recovery_logical_acommutes + observables.flatten()) % 2).mean()
    return pL