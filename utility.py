import numpy as np
from numpy import sqrt

def SigmaX(): 
    return  np.array([[0,1],[1,0]])
def SigmaY(): 
    return np.array([[0,-1j],[1j,0]])
def SigmaZ(): 
    return np.array([[1,0],[0,-1]])

def UMEvec(dim):
    """unormalized, maximally entanged ket
    
    The form of UMEvec is ones at |i>|i> and zero elsewhere
    """
    psi = np.zeros((dim**2), complex)
    for n in range(0, dim):
        psi[n*(dim+1)] = 1
    #xfor
    return psi
#!UMEvec

def UMEmat(dim):
    """unormalized, maximally entanged density matrix"""
    psi = UMEvec(dim)
    return np.outer(psi, psi)
#!UMEmat

def pstr2pmat(pstr):
    """
    Convert Pauli string to Pauli matrix
    """
    pmat = 1
    for p in pstr:
        if p == 'I':
            pmat = np.kron(pmat, np.eye(2))
        elif p == 'X':
            pmat = np.kron(pmat, SigmaX())
        elif p == 'Y':
            pmat = np.kron(pmat, SigmaY())
        elif p == 'Z':
            pmat = np.kron(pmat, SigmaZ())
    return pmat
#!pstr2pmat

def matrize(vec, R=-1, C=-1):
    """
    convention: stack rows
    """
    if R == -1 and C == -1:
        # default: square matrix
        dim = np.sqrt(vec.size)
        assert abs(dim - int(dim)) < 1e-9, "ERROR::matrize(vec); Vector not of square length."
        dim = int(dim)
    
        arry = []
        for d1 in range(dim):
            arry.append([])
            for d2 in range(dim):
                arry[d1].append(vec[d1*dim+d2])
        return np.array(arry)
    else:
        # rectangular matrix
        assert vec.size == R*C
        arry = []
        for dr in range(R):
            arry.append([])
            for dc in range(C):
                arry[dr].append(vec[dr*C + dc])
            #xfor
        #xfor 
        return np.array(arry)
#!matrize

def vectorize(mat):
    """
    convention: stack columns 
    """
    dims = mat.shape
    assert len(dims) == 2, "input not a matrix"
    Dr = dims[0]
    Dc = dims[1]
    
    vecout = np.zeros((Dc*Dr), dtype=complex)
    for dr in range(Dr):
        for dc in range(Dc):
            vecout[dr*Dc + dc] = mat[dr, dc]
    return vecout
#!matrize

def num2base(n, b, k):
    """
    convert n to base b with k places
    """
    if n == 0:
        return [0 for _ in range(k)]
    digits = []
    for ki in range(k):
        if n > 0:
            digits.append(int(n % b))
            n //= b
        else:
            digits.append(0)
    return digits[::-1]
#!num2base

def dagger(mat):
    """
    Dagger operation = complex conjugate & transpose
    """
    return np.conj(np.transpose(mat))
#!dagger

def TrX(p, sys, dim):
    """
    my partial trace function
    
    p: input density matrix
    sys: systems to be traced over, reffering to dim's indices
    dim: dimensions of the subsystems
    
    cf: http://www.dr-qubit.org/matlab/TrX.m
    """
    sys = np.asarray(sys)
    dim = np.asarray(dim)
    assert np.prod(dim) == p.shape[0]
    Ndim = dim.size
    keep = [d for d in range(Ndim)]
    for s in sys:
        keep.remove(s)
    Nkeep = np.prod(dim[keep])
    
    # reshape into tensor with one row & col index for each subsystem
    rho = p.reshape(np.tile(dim, 2))
    
    # make permutation that takes traced subsystems to first positions
    perm = [i for i in range(Ndim)]
    for idx, s in enumerate(sys):
        if perm[idx] != s:
            perm[perm.index(s)] = perm[idx]
            perm[idx] = s
    permp = [p+Ndim for p in perm]
    rho = np.transpose(rho, perm+permp)
    
    # trace over the first subsytems 
    for s in sys:
        rho = np.trace(rho, axis1=0, axis2=int(len(rho.shape)/2) )
    return rho.reshape(Nkeep, Nkeep)
#!TrX

### Printing Functions #############################################################

def cnum2str(cnum, prec=3):
    """Convert cnum to string with given precision"""
    t1 = f'{cnum.real:.{prec}f}'
    t2 = f'{cnum.imag:.{prec}f}'
    return '(' + t1 + ', ' + t2 + ')'
#!cnum2str

def mat2str(cmat, prec=3):
    strout = ""
    for row in cmat:
        strout += '['
        for c in row:
            strout += cnum2str(c, prec)
            strout += ' '
        strout += ']\n'
    return strout
#!cmat2str

def mat2html(cmat, prec=3):
    strout = ""
    for row in cmat:
        strout += '['
        for c in row:
            strout += cnum2str(c, prec)
            strout += ' '
        strout += ']<br>'
    return strout
#!cmat2str