import numpy as np
from numpy import sqrt
from .utility import *

class State:
    """
    Type to store quantum states
    
    State is stored as a density matrix always 
    
    contains: 
    _dmat <-- density matrix
    """
    def __init__(self, arrin):
        # check type of input array and then make density matrix
        assert type(arrin) is type(np.array([])), "input not a numpy array"
        assert len(arrin.shape) <= 2, "input array of dim > 2"
        if len(arrin.shape) == 1:
            self._dmat = np.outer(arrin, np.conj(arrin))
        else:
            self._dmat = arrin
    #!__init__
            
    def purity(self):
        return np.trace(self._dmat@self._dmat)
    
    def mat(self):
        return self._dmat
    
    def normalize(self):
        t = np.trace(self._dmat)
        if abs(1 - t) > 1e-9:
            self._dmat = (1/t)*self._dmat

    def html(self, title="", prec=3):
        fstr = "<h1>" + title + "</h1>\n"
        fstr += f"<h2> Quantum state with dimensions ({self._dmat.shape[0]}, {self._dmat.shape[1]}) </h2>"
        
        # print density matrix 
        fstr += "<h3> density matrix </h3>"
        fstr += mat2html(self._dmat, prec)
        
        return fstr
    
    def __str__(self):
        return mat2str(self._dmat)
#@State

class Register(State):
    """
    n-qubit register
    
    initializes state to logical 0
    """
    def __init__(self, qbits):
        self._dmat = np.zeros((2**qbits, 2**qbits), dtype=complex)
        self._dmat[0,0] = 1
#@Register

class Operator:
    """
    Type to represent quantum operator 
    """
    def __init__(self, matin):
        # assure the input is either a matrix or 2d numpy array
        assert (type(matin) is type(np.array([]))), "input not a numpy array"
        assert len(matin.shape) == 2, "input not a 2d array"
        assert matin.shape[0] == matin.shape[1], "input array not square"
        self._mat = matin
        
    def html(self, title="", prec=3):
        fstr = "<h1>" + title + "</h1>"
        fstr += f"<h2> Quantum operator with dimensions ({self._mat.shape[0]}, {self._mat.shape[1]})</h2>"
        
        # print density matrix 
        fstr += "<h3> matrix </h3>"
        fstr += mat2html(self._mat, prec)
        
        return fstr
    
    def mat(self):
        return self._mat
    
    def __str__(self):
        return mat2str(self._mat)
#xOperator

def gpauli(D):
    """
    Generate list of all Pauli operators on D qubits
    
    convert [0, 4^D-1] to base-4 and use that
    """
    L = 4**D
    Qlst = []
    for i in range(L):
        nb = num2base(i, 4, D)
        Qi = 1
        for j in nb:
            assert j < 4
            if j == 0:
                Qi = np.kron(Qi, np.eye(2))
            elif j == 1:
                Qi = np.kron(Qi, SigmaX)
            elif j == 2:
                Qi = np.kron(Qi, SigmaY)
            elif j == 3:
                Qi = np.kron(Qi, SigmaZ)
        Qlst.append(Qi)
    return Qlst
#!gpauli

