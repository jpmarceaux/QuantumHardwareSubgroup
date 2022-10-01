import numpy as np
from numpy import sqrt
from math import log
from .utility import *
from .states_operators import *

class Channel:
    """
    Class object to represent a quantum channel 
    
    Stores only the Choi tensor of the channel in the standard basis 
    
    Internal members:
    _cten, 
    """
    def __init__(self, *args, **kwargs):
        if kwargs.get('rep') == "cten":
            self._cten = args[0]
        elif kwargs.get('rep') == "cmat":
            self._cten = cmat2cten(args[0], args[1], args[2])
        elif kwargs.get('rep') == "lop":
            self._cten = lop2cten(args[0])
        elif kwargs.get('rep') == "kraus":
            self._cten = kraus2cten(args[0])
        elif kwargs.get('rep') == "pvec":
            self._cten = pvec2cten(args[0])
    
    def __str__(self):
        return "Channel with choi dimensions (%i, %i, %i, %i)" % self._cten.shape
    
    def apply(self, X):
        """
        use ChoiTensor and sum based on amplitudes of input X in standard basis 
        """
        assert(
            (type(X) is State) or 
            (type(X) is Operator) or
            (type(X) is np.array) or
            (type(X) is np.ndarray) 
        ), "X not state, operator, or numpy array"
        if type(X) is State or type(X) is Operator:
            Xmat = X.mat()
        else:
            Xmat = X
        assert Xmat.shape == self._cten[:,:,0,0].shape, "dimensional mismatch"
        dims = Xmat.shape
        xout = np.zeros(self._cten[0,0,:,:].shape, dtype=complex)
        kset = cten2kraus(self._cten)
        for K in kset:
            xout += K@Xmat@dagger(K)
        if type(X) is State:
            return State(xout)
        elif type(X) is Operator:
            return Operator(xout)
        else:
            return xout
        
    def rep(self, crep):
        if crep == "cten":
            return self._cten
        elif crep == "cmat":
            return cten2cmat(self._cten)
        elif crep == "lop":
            return cten2lop(self._cten)
        elif crep == "kraus":
            return cten2kraus(self._cten)
        elif crep == "pvec":
            return cten2pvec(self._cten)
    
    def html(self, prec=3):
        """make html representation of channel with given precision"""
        file = ""
        file += "<h1> Quantum channel with dimensions(%i, %i, %i, %i) </h1>" % self._cten.shape
        # print representations
        file += "<h2> Transfer Matrix: </h2>"
        file += mat2html(self.rep("lop"), prec)
        file += "<h2> Choi Tensor: </h2>"
        file += cten2html(self._cten, prec)
        file += "<h2> Pauli vector: </h2>"
        file += pvec2html(self.rep("pvec"), prec)
        return file
    
    def shape(self):
        return self._cten.shape
#@Channel



def cten2html(cten, prec=3):
    return cmat2html(cten2cmat(cten), prec) 
#xcten2html

def cmat2cten(cmat, R, C):
    """
    convert a dim**2 x dim**2 matrix into 4d numpy tensor
    of shape (C, C, R, R)
    """
    dims = cmat.shape
    assert dims[0] == R*C, "invalid dims"
    assert dims[0] == dims[1], "cmat not square"
    
    cten = np.zeros((C, C, R, R), dtype=complex)
    for i in range(C):
        for j in range(C):
            cten[i, j, :, :] = (
                cmat[(i*R):((i+1)*R), (j*R):(j+1)*R]
            )
    return cten

def cten2cmat(cten):
    """
    convert choi tensor into 2D matrix
    """
    assert len(cten.shape) == 4, "not a proper choi tensor"
    C = int(cten.shape[0])
    R = int(cten.shape[2])
    cmat = np.zeros((R*C, R*C), dtype=complex)
    for i in range(C):
        for j in range(C):
            cmat[(i*R):((i+1)*R), (j*R):(j+1)*R] = (
                cten[i, j, :, :]
            )
    return cmat

def cten2kraus(cten):
    C = cten.shape[0]
    R = cten.shape[2]
    cmat = cten2cmat(cten)
    Lambda, Smat = np.linalg.eig(cmat)
    # take transpose of Smat so we are getting eigenvectors; 
    # (transpose of unitary matrix is still unitary)
    # see https://numpy.org/doc/stable/reference/generated/numpy.linalg.eig.html
    Smat = Smat.transpose()
    
    KrausSet = []
    for idx, l in enumerate(Lambda):
        svec = Smat[idx]
        lam = np.sqrt(l)
        kvec = lam*svec
        KrausSet.append(matrize(kvec, R, C))
    return KrausSet

def cmat2kraus(cmat, R, C):
    cten = cmat2cten(cmat, R, C)
    return ctenkraus(cten)

def kraus2cten(kset):
    """
    convert kset into choi matrix
    """
    dims = kset[0].shape
    for K in kset:
        assert dims == K.shape, "Kraus ops not of equal dims"
    
    cmat = np.zeros((dims[0]*dims[1], dims[0]*dims[1]), dtype=complex)
    for K in kset:
        cmat += np.outer(vectorize(K), vectorize(dagger(K)))
    return cmat2cten(cmat, kset[0].shape[0], kset[0].shape[1])

def kraus2cmat(kset):
    return cten2cmat(kraus2cten(kset), kset[0].shape[0], kset[0].shape[1])

def cten2lop(cten):
    """
    convert choi tensor to linear operator 
    
    calculate by vectorizing blocks 
        using row-stacking convention
    """
    
    dims = cten.shape
    
    dx = dims[0]*dims[2]
    dy = dims[1]*dims[3]
    assert dx == dy, "choi matrix not square"
    
    lop = np.zeros((dims[0]*dims[1], dims[2]*dims[3]), dtype=complex)
    
    L = dims[0]
    for i in range(L):
        for j in range(L):
            lop[i*L+j, :] = vectorize(cten[i,j,:,:])
    return lop
#xcmat2lop

def lop2cten(lop):
    """
    convert linear operator to choi matrix
    
    assume input lop of shape (R^2, C^2), 
        where a kraus operator is of shape (R,C)
    
    calculate by calculating K(Phi)|i>|j> and matrizing, i.e., 
        matrizing the row i*C+j 
    """
    dims = lop.shape
    R = sqrt(dims[0])
    C = sqrt(dims[1])
    assert abs(R-int(R)) < 1e-9, "Lop not of square row dimension"
    R = int(R)
    assert abs(C-int(C)) < 1e-9, "Lop not of square column dimension"
    C = int(C)
    
    cten = np.zeros((C,C,R,R), dtype=complex)
    
    cset = []
    for i in range(R):
        for j in range(R):
            cij = matrize(lop[i*R+j, :])
            cset.append(cij)
    for i in range(C):
        for j in range(C):
            cten[i,j,:,:] = cset[0]
            cset.pop(0)
    return cten
#xlop2cten


def lop2cmat(lop):
    return cten2cmat(lop2cten(lop))
#xlop2cmat




def cmat2html(cmat, prec=3):
    strout = ""
    dims = cmat.shape
    dr = int(sqrt(dims[0]))
    dc = int(sqrt(dims[1]))
    for rdx, row in enumerate(cmat):
        cnt = 0
        strout += '|'
        for idx, c in enumerate(row): 
            strout += cnum2str(c, prec)
            cnt += 1
            if cnt%dc == 0 and idx != len(row) -1:
                strout += '| |'
        strout += "|</br>"
        if rdx%dr == 1:
            strout += "</br>"
    return strout
#!cmat2htmal
    

def cten2html(cten, prec=3):
    return cmat2html(cten2cmat(cten), prec) 
#xcten2html

def pvec2html(pvec, prec=3):
    strout = ""
    L = int(pvec.shape[0])
    k = int(log(L, 4))
    for l in range(L):
        strout += cnum2str(pvec[l], prec) + ' | '
        nb = num2base(l, 4, k)
        for j in nb:
            if j == 0:
                strout += 'I'
            elif j == 1:
                strout += 'X'
            elif j == 2:
                strout += 'Y'
            elif j == 3:
                strout += 'Z'
        strout += "<br/>"
    return strout
#!pvec2html

def lop2pvec(lop):
    if lop.shape[0] != lop.shape[1]:
        return np.array([-1])
    N = np.log2(lop.shape[0])
    if abs(N - int(N)) > 1e-9:
        return np.array([-2])
    N = int(N)
    Qs = gpauli(N)
    qvec = np.zeros(4**N, dtype=complex)
    for idx, Q in enumerate(Qs):
        qvec[idx] = np.trace(Q@lop)
    return (1/2**N)*qvec
#!lop2pvec

def pvec2lop(pvec):
    D = sqrt(pvec.shape[0])
    assert abs(D - int(D)) < 1e-9, "pvec not a square length"
    D = int(D)
    N = np.log2(D)
    assert abs(N - int(N)) < 1e-9, "not of qubit dimensions"
    N = int(N)
    
    lopout = np.zeros((D,D), dtype=complex)
    G = gpauli(N)
    assert pvec.shape[0] == len(G), "dimension mismatch"
    
    for idx,g in enumerate(G):
        lopout += pvec[idx]*g
    return lopout
#!pvec2lop

def cten2pvec(cten):
    return lop2pvec(cten2lop(cten))
#!cten2pvec

def pvec2cten(pvec):
    return lop2cten(pvec2lop(pvec))
#!cten2pvec

