### lib/stabilizer.py ###
import numpy as np
import stim, stimcirq, cirq
from typing import Dict, List, Tuple

def stabilizer_circuits(lat, encoding, after_encode_unitary, noise, rounds=1, meas_logical_via_ancilla=False):
    '''
    
    Output full stabilizer circuits starting from difference 
    Implemented for k=1 logical qubit
    
    Input:
        lat: compass code lattice 
        
        encoding: starting by just resetting or performing full encoding, choose one from {reset_X, reset_Z, encode_Z}
        
        after_encode_unitary: to go from 0 -> 1 or + -> -
        
        noise: circuit level noise (gate_noise1, gate_noise2, meas_noise)
        
        meas_logical_via_ancilla: keep this option for testing only, to be removed once no longer needed
    '''
    
    Sx, Sz, ncol = lat.getSx(), lat.getSz(), lat.dimZ
    logical, logical_type = (lat.logicals[0],'X') if 'X' in encoding else (lat.logicals[1],'Z')
    _, _, meas_noise = noise
    
    ### manual stabilizer check ###
    # stabilizer_checks = stim.Circuit(f'''
    #     H 9 11 10 12
    #     {gate_noise1} 9 11 10 12
    #     CX 9 3 10 5 11 7 2 13 4 15 8 14
    #     {gate_noise2} 9 3 10 5 11 7 2 13 4 15 8 14
    #     CX 9 0 10 2 11 4 1 13 3 15 7 14 
    #     {gate_noise2} 9 0 10 2 11 4 1 13 3 15 7 14
    #     CX 10 4 11 6 12 8 1 15 5 14 7 16
    #     {gate_noise2} 10 4 11 6 12 8 1 15 5 14 7 16
    #     CX 10 1 11 3 12 5 0 15 4 14 6 16
    #     {gate_noise2} 10 1 11 3 12 5 0 15 4 14 6 16
    #     H 9 11 10 12
    #     {gate_noise1} 9 11 10 12
    #     {meas_noise} 9 10 11 12 13 14 15 16
    #     MR 9 10 11 12 13 14 15 16
    # ''')
    
    ### SET UP ###
    n_data = len(Sx[0])
    n_stab_x = len(Sx)
    n_stab_z = len(Sz)
    n_stab = len(Sx) + len(Sz)

    stabilizer_checks = zigzag_stabilizer_checks(Sx, Sz, ncol, *noise)
    logical_gadget = measurement_gadgets([logical],'cnot',*noise)[0]

    if logical_type == 'X':
        detector_1st_round = [stim.Circuit(f'DETECTOR({n_data+i}, 0) rec[{i-n_stab}]') for i in range(n_stab_x)]
    else:
        detector_1st_round = [stim.Circuit(f'DETECTOR({n_data+i}, 0) rec[{i-n_stab}]') for i in range(n_stab_x,n_stab)]
    detector_nth_round = [stim.Circuit(f'DETECTOR({n_data+i}, 0) rec[{i-n_stab}] rec[{i-2*n_stab}]') for i in range(n_stab)]

    ### INITIALIZATION ###
    circuit = stim.Circuit(f'R ' + ' '.join(str(i) for i in range(n_data,n_data+n_stab)))
    if encoding == 'reset_X':
        circuit += stim.Circuit(f'RX ' + ' '.join(str(i) for i in range(n_data)))
    else:
        circuit += stim.Circuit(f'R  ' + ' '.join(str(i) for i in range(n_data)))
        if encoding == 'reset_Z':
            pass
        elif encoding == 'encode_Z':
            circuit += StabilizerCode(Sx+Sz).encoding_circuit(stim=True)
        else:
            raise NotImplementedError(f'`{encoding}` is not implemented')
    
    circuit += gatestring_to_stimcircuit(after_encode_unitary)

    ### 1st logical measurement via ancilla ###
    if meas_logical_via_ancilla:
        circuit += logical_gadget
        circuit.append('OBSERVABLE_INCLUDE', [stim.target_rec(-1)], 0)

    ### STABILIZER CHECKS ###
    # 1st round
    circuit.append('TICK')
    circuit += stabilizer_checks

    for detector in detector_1st_round:
        circuit += detector

    # nth round
    if rounds > 1:
        repeat_circuit = stabilizer_checks.copy()
        repeat_circuit += stim.Circuit('SHIFT_COORDS(0, 1)')
        for detector in detector_nth_round:
            repeat_circuit += detector

        circuit += repeat_circuit*(rounds-1)

    ### MEASURE LOGICAL OPERATOR ###     
    ### 2nd logical measurement ###
    if meas_logical_via_ancilla:
        circuit += logical_gadget
        circuit.append('OBSERVABLE_INCLUDE', [stim.target_rec(-1)], 0)
    # OR measure logical directly from data
    else:
        if logical_type == 'X':
            
            if meas_noise: circuit += stim.Circuit(f'Z{meas_noise[1:]} '+' '.join(str(i) for i in range(n_data)))
            circuit.append('MX', range(n_data))
            
            for j,sx in enumerate(Sx):
                targets = [f'rec[-{n_stab+n_data-j}]']+[f'rec[-{i}]' for i in n_data-char_locations(sx,'X')]
                circuit += stim.Circuit(f'DETECTOR({n_data+j}, 1) '+' '.join(targets))
                
        else:
            
            if meas_noise: circuit += stim.Circuit(f'{meas_noise} '+' '.join(str(i) for i in range(n_data)))
            circuit.append('M', range(n_data))
            
            for j,sz in enumerate(Sz):
                targets = [f'rec[-{n_stab_z+n_data-j}]']+[f'rec[-{i}]' for i in n_data-char_locations(sz,'Z')]
                circuit += stim.Circuit(f'DETECTOR({n_data+n_stab_x+j}, 1) '+' '.join(targets))

        obs_targets = [stim.target_rec(-i) for i in n_data-char_locations(logical,logical_type)]
        circuit.append('OBSERVABLE_INCLUDE', obs_targets, 0)
    
    return circuit

def measurement_gadgets(stabilizers_in, construction='cnot', gate_noise1=None, gate_noise2=None, meas_noise=None):
    """
    Input:
        stabilizers: a list of stabilizers written in terms of {I/_,X,Y,Z}
        construction: direct or hadamard:
            1) `cnot` using only CNOTs from data to ancilla along with single qubit gates
                - H then S    : rotates Z basis -> Y basis
                - S_dag then H: rotates Y basis -> Z basis
                verifiable via checking that: Y stabilizer == kron(S@H,I) @ CNOT @ kron(H@S_dag)
            2) `hadamard` using H gates on ancilla and C-Pauli from ancilla to data
    Output:
        Measurement gadgets
    
    first N qubits as data qubits
    remaining as ancilla qubits
    """
    
    # allow both '_' and 'I' in stabilizers
    stabilizers = [s.replace('_','I') for s in stabilizers_in] if '_' in stabilizers_in[0] else stabilizers_in.copy()
    
    N = len(stabilizers[0])
    gadgets = []
    for ancilla, stab in enumerate(stabilizers):
        if construction == 'cnot':
            bS, bH, mid, aH, aS = '','','','',''
            for i, pauli in enumerate(stab):
                s = f'{i} '
                if pauli != 'I':
                    mid += s + f'{ancilla+N} '
                    if pauli == 'X':
                        bH += s
                        aH += s
                    elif pauli == 'Y':
                        bS += s
                        bH += s
                        aH += s
                        aS += s
            # gate noise
            if gate_noise1:
                bS = f'S_DAG {bS} \n{gate_noise1} {bS}' if len(bS) > 0 else ''
                bH = f'H {bH}     \n{gate_noise1} {bH}' if len(bH) > 0 else ''
                aH = f'H {aH}     \n{gate_noise1} {aH}' if len(aH) > 0 else ''
                aS = f'S {aS}     \n{gate_noise1} {aS}' if len(aS) > 0 else ''
            else:
                bS = f'S_DAG {bS}' if len(bS) > 0 else ''
                bH = f'H {bH}    ' if len(bH) > 0 else ''
                aH = f'H {aH}    ' if len(aH) > 0 else ''
                aS = f'S {aS}    ' if len(aS) > 0 else ''
                
            if gate_noise2:
                mid = f'CX {mid} \n{gate_noise2} {mid}' if len(mid) > 0 else ''
            else:
                mid = f'CX {mid}' if len(mid) > 0 else ''
            
            # measurement noise
            if meas_noise:
                aS += f'\n{meas_noise} {ancilla + N}'
            aS += f'\nMR {ancilla + N}'
            gadgets.append(stim.Circuit('\n'.join([bS,bH,mid,aH,aS])))

        elif construction == 'hadamard':
            before, mX, mY, mZ, after = '','','','',''
            s1 = f'{ancilla+N} '
            before += s1
            for i, pauli in enumerate(stab):
                s2 = f'{ancilla+N} {i} '
                if pauli == 'X': 
                    mX += s2
                elif pauli == 'Y': 
                    mY += s2
                elif pauli == 'Z': 
                    mZ += s2
            after += s1 
            
            # gate noise
            if gate_noise1:
                before = f'H {before} \n{gate_noise1} {before}' if len(before) > 0 else ''
                after  = f'H {after}  \n{gate_noise1} {after}' if len(after) > 0 else ''
            else:
                before = f'H {before}' if len(before) > 0 else ''
                after  = f'H {after}' if len(after) > 0 else ''
                
            if gate_noise2:
                mX = f'CX {mX} \n{gate_noise2} {mX}' if len(mX) > 0 else ''
                mY = f'CY {mY} \n{gate_noise2} {mY}' if len(mY) > 0 else ''
                mZ = f'CZ {mZ} \n{gate_noise2} {mZ}' if len(mZ) > 0 else ''
            else:
                mX = f'CX {mX}' if len(mX) > 0 else ''
                mY = f'CY {mY}' if len(mY) > 0 else ''
                mZ = f'CZ {mZ}' if len(mZ) > 0 else ''
           
            # measurement noise
            if meas_noise:
                after += f'\n{meas_noise} {ancilla + N}'
            after += f'\nMR {ancilla + N}'
            gadgets.append(stim.Circuit('\n'.join([before,mX,mY,mZ,after])))
            
    return gadgets

def zigzag_stabilizer_checks(Sx, Sz, ncol, gate_noise1=None, gate_noise2=None, meas_noise=None):
    """
    Input:
        Sx: a list of X-type stabilizers
        Sz: a list of Z-type stabilizers
        ncol: number of columns in lattice
        gate_noise1: 1-qubit gate noise channel
        gate_noise2: 2-qubit gate noise channel
        meas_noise: measurement noise channel
    Output:
        Measurement gadgets in stim following the zigzag pattern
    
    How it works:
        - For CSS code only
        - Start measuring from the upper left corner of all stabilizers (X then Z)
        - CNOT on gauge operators one after the other in zigzag pattern (X | and Z --)
        - X-type stabilizers measured via H on ancilla and CX from ancilla to data
        - Z-type stabilizers measured via CX from data to ancilla
        
    first N qubits as data qubits
    remaining as ancilla qubits
    """
    # starting points for all stabilizers
    x_start = np.array([stab.index('X') for stab in Sx])
    z_start = np.array([stab.index('Z') for stab in Sz])
    n_stab_x = len(Sx)
    n_stab = len(Sx)+len(Sz)
    n_data = len(Sx[0])
    max_length_x = max([sx.count('X') for sx in Sx])//2
    max_length_z = max([sz.count('Z') for sz in Sz])//2
    
    ### X-type
    
    # Hadamards
    H_targets = ' '.join(str(i) for i in range(n_data,n_data+n_stab_x))
    circ_str = 'H ' + H_targets
    if gate_noise1: circ_str += f'\n{gate_noise1} ' + H_targets     
    circ_str += '\nTICK'
    
    for _ in range(max_length_x):
        targets = ['','']
        for i in range(len(x_start)):
            i1, i2 = x_start[i], x_start[i]+ncol
            try:
                if Sx[i][i1] + Sx[i][i2] == 'XX':
                    targets[0] += f' {n_data+i} {i1}'
                    targets[1] += f' {n_data+i} {i2}'
            except:
                pass
        for target in targets:
            circ_str += '\nCX' + target
            if gate_noise2: circ_str += f'\n{gate_noise2}' + target
            circ_str += '\nTICK'
        x_start += 1
        
    # Hadamards again
    circ_str += '\nH ' + H_targets
    if gate_noise1: circ_str += f'\n{gate_noise1} ' + H_targets
    circ_str += '\nTICK'
    
    ### Z-type
    for _ in range(max_length_z):
        targets = ['','']
        for i in range(len(z_start)):
            i1, i2 = z_start[i], z_start[i]+1
            try:
                if Sz[i][i1] + Sz[i][i2] == 'ZZ':
                    targets[0] += f' {i1} {n_data+n_stab_x+i}'
                    targets[1] += f' {i2} {n_data+n_stab_x+i}'
            except:
                pass
        for target in targets:
            circ_str += '\nCX' + target
            if gate_noise2: circ_str += f'\n{gate_noise2}' + target
            circ_str += '\nTICK'
        z_start += ncol
        
    # Measure all ancillas
    M_targets = ' '.join(str(i) for i in range(n_data,n_data+n_stab))
    # M_targets = '9 15 11 16 13 10 14 12'
    if meas_noise: circ_str += f'\n{meas_noise} ' + M_targets
    circ_str += '\nMR ' + M_targets
    return stim.Circuit(circ_str)


'''
/////
Adapted from Cirq/examples/stabilizer_code.py
JP's modification
/////
'''
class StabilizerCode(object):
    def __init__(self, group_generators_in: List[str]):
        # allow both '_' and 'I' in group_generators
        group_generators = [g.replace('_','I') for g in group_generators_in] if '_' in group_generators_in[0] else group_generators_in.copy()        
        n = len(group_generators[0])
        k = n - len(group_generators)

        # Build the matrix defined in section 3.4. Each row corresponds to one generator of the
        # code, which is a vector of dimension n. The elements of the vectors are Pauli matrices
        # encoded as I, X, Y, or Z. However, as described in the thesis, we encode the Pauli
        # vector of 2*n Booleans.
        M = np.zeros((n - k, 2 * n), np.int8)
        for i, group_generator in enumerate(group_generators):
            for j, c in enumerate(group_generator):
                if c == 'X' or c == 'Y':
                    M[i, j] = 1
                elif c == 'Z' or c == 'Y':
                    M[i, n + j] = 1

        M, X, Z, r, permutation = _transfer_to_standard_form(M, n, k)

        self.n: int = n
        self.k: int = k
        self.r: int = r
        self.M: List[str] = _build_by_code(M)
        self.logical_Xs: List[str] = _build_by_code(X)
        self.logical_Zs: List[str] = _build_by_code(Z)
        self.permutation = permutation # register permutation mapping physical qubit indices to encoded indices
                
    def encoding_circuit(self,stim=False):
        """
        produces an encoding circuit that maps the all 0's register to logical 0

            
        Returns: 
            A circuit that maps the register to the logical 0 state
            
        tracks permutations of columns when reducing to standard form 
        """
        register = [cirq.NamedQubit(str(i)) for i in range(self.n)]
        gate_dict = {'X': cirq.X, 'Y': cirq.Y, 'Z': cirq.Z}
        
        circuit = cirq.Circuit()
                
        for r in range(self.r):
            physical_r = self.permutation.index(r)
            circuit.append(cirq.H(register[physical_r]))

            # Let's consider the first stabilizer:
            # The reason for adding S gate is Y gate we used is the complex format (i.e. to
            # make it Hermitian). It has following four cases: (ignore the phase factor)
            # (I+X@P_2...P_k)|0...0> = |0...0> + |1>|\psi>
            # (I+Y@P_2...P_k)|0...0> = |0...0> + i|1>|\psi>
            # The other forms are not possible in the standard form, by construction.

            # The first case means we need [1,1] vector and controlled gates and in the
            # second case we need [1, i] vector and controlled gates. Corresponding, it is
            # the first column of H and the first column of SH respectively.

            # For the other stabilizers, the process can be repeated, as by definition they
            # commute.

            if self.M[r][r] == 'Y' or self.M[r][r] == 'Z':
                circuit.append(cirq.S(register[physical_r]))

            for n in range(self.n):
                physical_n = self.permutation.index(n)
                if n == r:
                    continue
                if self.M[r][n] == 'I':
                    continue
                op = gate_dict[self.M[r][n]]
                # assert op == cirq.X
                circuit.append(cirq.CNOT(register[physical_r], register[physical_n]))

        return stimcirq.cirq_circuit_to_stim_circuit(circuit) if stim else circuit
    
def _build_by_code(mat: np.ndarray) -> List[str]:
    """Transforms a matrix of Booleans into a list of Pauli strings.
    Takes into input a matrix of Boolean interpreted as row-vectors, each having dimension 2 * n.
    The matrix is converted into another matrix with as many rows, but this time the vectors
    contain the letters I, X, Y, and Z representing Pauli operators.
    Args:
        mat: The input matrix of Booleans.
    Returns:
        A list of Pauli strings.
    """
    out = []
    n = mat.shape[1] // 2
    for i in range(mat.shape[0]):
        ps = ''
        for j in range(n):
            k = 2 * mat[i, j + n] + mat[i, j]
            ps += "IXZY"[k]
        out.append(ps)
    return out


# It was considered to use scipy.linalg.lu but it seems to be only for real numbers and does
# not allow to restrict only on a section of the matrix.
def _gaussian_elimination(
    M: np.ndarray, min_row: int, max_row: int, min_col: int, max_col: int, permutation: list
) -> int:
    """Gaussian elimination for standard form.
    Performs a Gaussian elemination of the input matrix and transforms it into its reduced row
    echelon form. The elimination is done only on a sub-section of the matrix (specified) by
    ranges of rows and columns. The matrix elements are integers {0, 1} interpreted as elements
    of GF(2).
    In short, this is the implementation of section 4.1 of the thesis.
    Args:
        M: The input/output matrix
        min_row: The minimum row (inclusive) where the perform the elimination.
        max_row: The maximum row (exclusive) where the perform the elimination.
        min_col: The minimum column (inclusive) where the perform the elimination.
        max_col: The maximum column (exclusive) where the perform the elimination.
    Returns:
        The rank of the matrix.
    """
    assert M.shape[1] % 2 == 0
    n = M.shape[1] // 2

    max_rank = min(max_row - min_row, max_col - min_col)

    rank = 0
    for r in range(max_rank):
        
        i = min_row + r
        j = min_col + r
        
        pivot_rows, pivot_cols = np.nonzero(M[i:max_row, j:max_col])

        if pivot_rows.size == 0:
            break

        pi = pivot_rows[0]
        pj = pivot_cols[0]

        # Swap the rows:
        M[[i, i + pi]] = M[[i + pi, i]]

        # Swap the columns:
        M[:, [(j + pj), j]] = M[:, [j, (j + pj)]]
        if j > n: 
            # if you're in the 2nd column space
            j_other_half = (j + n) % (2 * n)
            i1 = permutation.index(j_other_half)
            i2 = permutation.index(j_other_half+pj)
            temp = permutation[i1]
            permutation[i1] = permutation[i2]
            permutation[i2] = temp
        else:
            i1 = permutation.index(j)
            i2 = permutation.index(j+pj)
            temp = permutation[i1]
            permutation[i1] = permutation[i2]
            permutation[i2] = temp
            
        # Since the columns in the left and right half of the matrix represent the same qubit, we
        # also need to swap the corresponding column in the other half.
        j_other_half = (j + n) % (2 * n)
        M[:, [(j_other_half + pj), j_other_half]] = M[:, [j_other_half, (j_other_half + pj)]]

        # Do the elimination.
        for k in range(i + 1, max_row):
            if M[k, j] == 1:
                M[k, :] = np.mod(M[i, :] + M[k, :], 2)

        rank += 1

    # Backward replacing to get identity
    for r in reversed(range(rank)):
        i = min_row + r
        j = min_col + r

        # Do the elimination in reverse.
        for k in range(i - 1, min_row - 1, -1):
            if M[k, j] == 1:
                M[k, :] = np.mod(M[i, :] + M[k, :], 2)

    return rank, permutation


def _transfer_to_standard_form(
    M: np.array, n: int, k: int
) -> Tuple[np.array, np.array, np.array, int]:
    """Puts the stabilizer matrix in its standardized form, as in section 4.1 of the thesis.
    Args:
        M: The stabilizier matrix, to be standardized.
        n: Dimension of the code words.
        k: Dimension of the message words.
    Returns:
        The standardized matrix.
        The logical Xs.
        The logical Zs.
        The rank of the matrix.
        
    Keeps track of the permutation of columns in gaussian elimination 
    """
    permutation = [i for i in range(n)]

    # Performing the Gaussian elimination as in section 4.1
    
    r, permutation = _gaussian_elimination(M, 0, n - k, 0, n, permutation)
    _, permutation = _gaussian_elimination(M, r, n - k, n + r, 2 * n, permutation)
    
    # Get matrix sub-components, as per equation 4.3:
    A2 = M[0:r, (n - k) : n]
    C1 = M[0:r, (n + r) : (2 * n - k)]
    C2 = M[0:r, (2 * n - k) : (2 * n)]
    E = M[r : (n - k), (2 * n - k) : (2 * n)]

    X = np.concatenate(
        [
            np.zeros((k, r), dtype=np.int8),
            E.T,
            np.eye(k, dtype=np.int8),
            np.mod(E.T @ C1.T + C2.T, 2),
            np.zeros((k, n - r), np.int8),
        ],
        axis=1,
    )

    Z = np.concatenate(
        [
            np.zeros((k, n), dtype=np.int8),
            A2.T,
            np.zeros((k, n - k - r), dtype=np.int8),
            np.eye(k, dtype=np.int8),
        ],
        axis=1,
    )
    return M, X, Z, r, permutation

'''
/////
---------------------------
/////
'''

def gatestring_to_stimcircuit(gate_string):
    loc = {}
    for i,gate in enumerate(gate_string):
        if gate == 'I' or gate == '_':
            continue
        if gate not in loc:
            loc[gate] = ''
        loc[gate] += f' {i}'
    return stim.Circuit('\n'.join([g+l for g,l in zip(loc.keys(),loc.values())]))

def char_locations(string,char):
    '''Return locations where `char` appears in `string` '''
    locs = []
    for loc,c in enumerate(string):
        if c == char:
            locs.append(loc)
    return np.array(locs)