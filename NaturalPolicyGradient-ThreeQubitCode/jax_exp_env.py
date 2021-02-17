import numpy as np
import itertools

zero = np.array([[1.],[0.]])
one = np.array([[0.],[1.]])
sigmax = np.array([[0.,1.],[1.,0.]])
sigmay = np.array([[0.,-1j],[1j,0.]])
sigmaz = np.array([[1.,0.],[0.,-1.]])
sigma = {'x':sigmax,'y':sigmay,'z':sigmaz}

def jtensor(arr):
    prod = arr[0]
    for i in range(1,len(arr)):
        prod = jnp.kron(prod,arr[i])
    return prod

def tensor(arr):
    prod = arr[0]
    for i in range(1,len(arr)):
        prod = np.kron(prod,arr[i])
    return prod

def projectors(num_qubit,imeas):
    proj = [[np.eye(2) for i in range(num_qubit)],
            [np.eye(2) for i in range(num_qubit)]]
    proj[0][imeas] = zero@zero.T
    proj[1][imeas] = one@one.T
    return [tensor(p) for p in proj]

def cnot(num_qubit,control,target):
    on = [np.eye(2) for i in range(num_qubit)]
    on[control] = one@one.T
    on[target] = sigmax
    off = [np.eye(2) for i in range(num_qubit)]
    off[control] = zero@zero.T
    return tensor(on) + tensor(off)
    
def flip(num_qubit,imeas,kind='x'):
    gate = [np.eye(2) for i in range(num_qubit)]
    gate[imeas] = sigma[kind]
    return tensor(gate)

def get_actions(num_qubit):
    projs = {f'M{i}':projectors(num_qubit,i) for i in range(num_qubit)} # meas dict
    cnots = {f'CNOT{s[0]}{s[1]}':cnot(num_qubit,s[0],s[1]) 
             for s in itertools.permutations(range(num_qubit),2)} # cnot dict
    bitflips = {f'X{i}':flip(num_qubit,i,'x') for i in range(num_qubit)}

    return {**projs, **cnots, **bitflips, 'I':np.eye(2**num_qubit)}

def rho_n(n,num_qubit,rest=0):
    rho_logical = 0.5*(np.eye(2) + n[0]*sigmax + n[1]*sigmay + n[2]*sigmaz)
    if rest==0:
        ket_rest = tensor([zero for i in range(num_qubit-1)])
    else:
        ket_rest = tensor([one for i in range(num_qubit-1)])
    rho_rest = ket_rest @ ket_rest.T
    return tensor([rho_logical,rho_rest])

from jax.config import config
config.update("jax_enable_x64", True)
import jax
import jax.numpy as jnp
from scipy.integrate import ode
from functools import partial

@partial(jax.jit)
@partial(jax.vmap, in_axes=(0,None))
def core(rho, expD):
#     return expD.dot(rho)
    return jnp.dot(expD,rho)

def evolve_jax(rho0_batch, expD_jit):
    # Reshape
    init_shape = rho0_batch.shape
    rho0_batch = rho0_batch.reshape(*init_shape[:-2],-1) # (batch...,2^N,2^N)->(batch...,4^N)
    rho0_batch = rho0_batch.reshape(-1,rho0_batch.shape[-1]) # -> (batch,4^N)
    
    res = np.array(expD_jit(jnp.array(rho0_batch))).T
    res = res.reshape(*init_shape[:-2],*init_shape[-2:])
    
    return res

def update_rho(rho,U):
    return jnp.dot(jnp.dot(U,rho),U.T.conj())

class jax_exp_env():
    '''
    Apply the action at t0
    Evolve the density matrix from t0 to t1
    Input:
        rho0: [batch_size, 4, dim, dim]
    Return: rho at t1
    '''
    def __init__(self, num_qubit, rho0, expD, actions, tmax, 
                 num_substep=2, hparam={'P':0.1}, pparam={'Tsingle':50}, 
                 solver_args={'atol':1e-8,'rtol':1e-8,'method':'adam','order':12}):
        # Parameters
        self.num_qubit = num_qubit
        self.rho0 = rho0
        self.size = rho0.shape[0]
        self.expD_jit = partial(core, expD)
        self.update_rho = jax.jit(jax.vmap(update_rho,in_axes=(0,None)))
        self.actions = actions
        self.tmax = tmax
        self.num_substep = num_substep
        self.hparam = hparam # hyperparameters
        self.pparam = pparam # physical parameters
        self.solver_args= solver_args
        
    def reset(self):
        # Initialization
        self.rho = self.rho0.copy()
        self.tnow = 0
        self.norm_factors = np.ones((self.size,4))
        self.rq = np.ones(self.size)
        self.prob = np.ones(self.size)
        self.revealed = np.zeros(self.size,dtype=bool)
        # initial state: do nothing (I), no measurement outcome (2)           
        return (self.rho, -1*np.ones(self.size), 2*np.ones(self.size), np.zeros((self.size,3)), self.tnow) 

    def step(self,a,gate_first=True,meas_types=''):
        if isinstance(meas_types,str):
            meas_types = ['']*self.size
        acts = np.array(list(self.actions))[a]
        if gate_first == -1:
            meas_out = self.gate_operation(acts,meas_types)
        elif gate_first:
            meas_out = self.gate_operation(acts,meas_types)
            self.dissipative_dynamics()
        else:
            self.dissipative_dynamics()        
            meas_out = self.gate_operation(acts,meas_types)
        self.normalization() 

        ### Update everything else
        self.tnow += 1
        reward = self.reward_bias(acts)

        new_state = np.einsum('ijkl,ij->ijkl',self.rho,self.norm_factors)
        new_state = (new_state, a, meas_out, self.meas_bias, self.tnow)
        
        eps = 1e-5
        biased = abs(self.meas_bias.max(axis=1)) > 1e-3
        zero_rq = self.rq_weighted < eps
        biased_or_zero_rq = np.any([biased,zero_rq],axis=0)
        not_revealed = np.logical_not(self.revealed)
        tdone = [np.isclose(self.tnow,self.tmax)]*self.size
        

        self.revealed = np.where( np.all([biased_or_zero_rq,not_revealed],axis=0), True, self.revealed)
        done = np.any([tdone,self.revealed],axis=0)
           
        return new_state, reward, done
    
    def gate_operation(self, acts, meas_types):
        self.rho_other = np.zeros(self.rho.shape,dtype=np.complex128)
        self.meas_out = 2*np.ones(self.size)
        self.p = np.ones(self.size)
        ### Apply the action ###
        for ib,act in enumerate(acts):
            if 'M' in act:
                # Get projectors and calculate probability Trace(proj @ normalized_rho0)
                proj = self.actions[act]
                p = ( proj[1] @ self.rho[ib,0]/self.rho[ib,0].trace() ).trace()
                assert not p.imag.any()
                self.p[ib] = p.real
                meas_out = self.measurement_outcome(self.p[ib],meas_types[ib])
                self.meas_out[ib] = meas_out
    #             print(act,self.p,self.meas_out)
                # Update the state
                rho = [np.array(self.update_rho(self.rho[ib],proj[i])) for i in range(2)]
                self.rho[ib] = rho[meas_out]
                self.rho_other[ib] = rho[1-meas_out]
            elif act != 'I':
                # Apply unitary
                U = self.actions[act]
                self.rho[ib] = np.array(self.update_rho(self.rho[ib],U))
        return self.meas_out
    
#     def gate_operation(self, acts, meas_types):
#         self.rho_other = np.zeros(self.rho.shape,dtype=np.complex128)
#         self.meas_out = 2*np.ones(self.size)
#         self.p = np.ones(self.size)
#         ### Apply the action ###
#         for ib,act in enumerate(acts):
#             if 'M' in act:
#                 # Get projectors and calculate probability Trace(proj @ normalized_rho0)
#                 proj = self.actions[act]
#                 p = ( proj[1] @ self.rho[ib,0]/self.rho[ib,0].trace() ).trace()
#                 assert not p.imag.any()
#                 self.p[ib] = p.real
#                 meas_out = self.measurement_outcome(self.p[ib],meas_types[ib])
#                 self.meas_out[ib] = meas_out
#     #             print(act,self.p,self.meas_out)
#                 # Update the state
#                 rho = [np.einsum('ij,hjk,kl->hil',proj[i],self.rho[ib],proj[i]) for i in range(2)]
#                 self.rho[ib] = rho[meas_out]
#                 self.rho_other[ib] = rho[1-meas_out]
#             elif act != 'I':
#                 # Apply unitary
#                 U = self.actions[act]
#                 self.rho[ib] = np.einsum('ij,hjk,kl->hil',U,self.rho[ib],U.T.conj())
#         return self.meas_out
    
    # For testing against Foesel et al
    def measurement_outcome(self, p, meas_type):
        meas_out = np.random.binomial(1,p)
        if meas_type == 'noflip':
            meas_out = 0
        elif meas_type == 'flip':
            meas_out = 1        
        elif isinstance(meas_type,list):
            meas_out = meas_type.pop(0)
        return meas_out                   

    def dissipative_dynamics(self):
        # Evolve under the Lindblad eq
        times = np.linspace(self.tnow, self.tnow+1, self.num_substep)
        rho,rho_other,expD_jit,solver_args = self.rho,self.rho_other,self.expD_jit,self.solver_args
#         self.rho = evolve_jax(rho, times[0], times, rhs, **solver_args)[-1]
        self.rho = evolve_jax(rho, expD_jit)
        if rho_other.any():
            self.rho_other = evolve_jax(rho_other, expD_jit)
 
    def normalization(self):
        # Find rho_other ready for calculating rq       
        if self.rho_other.any():
            norm0_other = self.rho_other[:,0].trace(axis1=1,axis2=2)
            norm0_other = np.where(np.isclose(norm0_other,0.),1.,norm0_other)
            self.rho_other = np.einsum('ijkl,ij,i->ijkl',self.rho_other,self.norm_factors,1/norm0_other)
        # Normalize rho and update norm factor    
        norm = self.rho.trace(axis1=2,axis2=3)
        norm = np.where(np.isclose(norm,0.),1.,norm) 
        self.norm_factors = np.einsum('ij,ij,i->ij',self.norm_factors,norm,1/norm[:,0])
        self.rho = np.einsum('ijkl,ij->ijkl',self.rho,1/norm)
            
    def reward_bias(self, acts):
        # Recoverable quantum information R_Q
#         rq = np.ones((self.size,1))
        self.meas_bias = np.zeros((self.size,3))
        
        rhoj = np.einsum('ijkl,ij->ijkl',self.rho[:,1:],self.norm_factors[:,1:])
        rho0 = np.expand_dims(np.einsum('ijk,i->ijk',self.rho[:,0],self.norm_factors[:,0]),1)
        drho = rhoj - rho0
#         rq = np.hstack([rq,np.linalg.svd(drho)[1].sum(-1)]).min(-1)
        rq = np.linalg.svd(drho)[1].sum(-1).min(-1)
        self.rq_weighted = rq.copy()

        for ib,act in enumerate(acts):
            if 'M' in act:
                # Bias vector / note matrix multiplication here
                proj = self.actions[act]
                meas_bias = np.einsum('ijk,kl->ijl',drho[ib],proj[0]-proj[1]).trace(axis1=1,axis2=2)
                assert not meas_bias.imag.any()
                self.meas_bias[ib] = meas_bias.real
                drho_other = self.rho_other[ib,1:] - np.expand_dims(self.rho_other[ib,0],0)
                rq_other = np.linalg.svd(drho_other)[1].sum(-1).min()
                # Weighted R_Q and keep track of trajectory probability
                meas_out,p = self.meas_out[ib],self.p[ib]
                if meas_out == 0:
                    self.rq_weighted[ib] = (1-p)*rq[ib] + p*rq_other
                    self.prob *= 1-p
                else:
                    self.rq_weighted[ib] = p*rq[ib] + (1-p)*rq_other
                    self.prob *= p                

        # Immediate reward
        eps = 1e-5
        norm = 2/self.pparam['Tsingle']
        cond1 = self.rq_weighted > eps
        cond2 = np.all([self.rq_weighted < eps, self.rq > eps],axis=0)
        r1 = np.where(cond1,1+(self.rq_weighted-self.rq)/norm,0)
        r2 = np.where(cond2,-self.hparam['P'],0)
        self.rq = rq    
        return [r1,r2]