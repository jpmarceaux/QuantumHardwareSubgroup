import numpy as np
from numpy import sqrt
from .utility import *
from .states_operators import *

class Measurement:
    """
    Type to represent a quantum measurement
    """
    def __init__(self, mset):
        self._mset = []
        # check mset a proper POVM; this does not check for positivity
        dims = mset[0].shape
        assert dims[0] == dims[1], "measurement set not of square operators"
        # check closure to the identity 
        P = np.zeros(dims, dtype=complex)
        for mu in mset:
            P += mu
        assert np.all(np.isclose(P, np.eye(dims[0]))), "measurement set does not close to I"
        for m in mset:
            self._mset.append(m)
    
    def apply(self, rho):
        """apply measurement and output probability vector"""
        if type(rho) is State:
            dmat = rho.mat()
        else:
            dmat = rho
            
        pout = []
        for mu in self._mset:
            pout.append(np.trace(dmat@mu))
        return pout
    
    def html(self, title="", prec=3):
        fstr = "<h1>" + title + "</h1>\n"
        fstr += "<h1> Measurement Set </br>"
        
        # print pvec
        for m in self._mset: 
            fstr += mat2html(m, prec) + "</br>"
        return fstr
#@Measurement

class PauliMeasurement(Measurement):
    """
    Construct measurement on eigenspace of given Pauli operator 
    """
    def __init__(self, pstr):
        pmat = pstr2pmat(pstr)
        N = len(pstr)
        assert 2**N == pmat.shape[0], "dimension mismatch"
        self._mset = [
            (np.eye(2**N) + pmat)/2, 
            (np.eye(2**N) - pmat)/2
        ]