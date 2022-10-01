import numpy as np
import torch
import torch.nn.functional as F
from torch.nn.utils.rnn import pad_sequence
import matplotlib.pyplot as plt
import pandas as pd

def plot_pg(data):
    
    fig = plt.figure(figsize=(15,10))
    
    fig.add_subplot(4,3,1)
    smoothed_rewards = pd.Series.rolling(pd.Series(data[:,0]), 10).mean()
    smoothed_rewards = [elem for elem in smoothed_rewards]
    plt.plot(data[:,0])
    plt.plot(smoothed_rewards)
    plt.plot()
    plt.xlabel('Episode')
    plt.ylabel('Sum of return')

    fig.add_subplot(4,3,2)
    plt.plot(data[:,1])
    plt.xlabel('Episode')
    plt.ylabel('Episode length')

    fig.add_subplot(4,3,3)
    smoothed_rq = pd.Series.rolling(pd.Series(data[:,2]), 10).mean()
    smoothed_rq = [elem for elem in smoothed_rq]
    plt.plot(data[:,2],'-o')
    plt.plot(smoothed_rq)
#     plt.yscale('log')
    plt.ylabel('$R_Q$')

    fig.add_subplot(4,3,4)
    plt.plot(data[:,3])
    plt.xlabel('Episode')
    plt.ylabel('Loss')
    
    fig.add_subplot(4,3,5)
    plt.plot(data[:,4])
    plt.xlabel('Episode')
    plt.ylabel('Effective Learning Rate')
    
    fig.add_subplot(4,3,6)
    plt.plot(data[:,5])
    plt.xlabel('Episode')
    plt.ylabel('Entropy')
    
    fig.add_subplot(4,3,7)
    plt.semilogy(data[:,6],label='min')
    plt.semilogy(data[:,7],label='max')
    plt.semilogy(data[:,8],label='mean')
    plt.xlabel('Episode')
    plt.ylabel('Measurement Probability')
    plt.legend()
    
    fig.add_subplot(4,3,8)
    plt.plot(data[:,9])
    plt.xlabel('Episode')
    plt.ylabel('Signals')

    fig.add_subplot(4,3,9)
    plt.plot(data[:,10])
    plt.xlabel('Episode')
    plt.ylabel('Max returns')

    fig.add_subplot(4,3,10)
    plt.plot(data[:,11],label='Euclidean norm')
    plt.xlabel('Episode')
    plt.ylabel('$\Delta g_{van}/g_{van}$')
        
    try:        
        plt.plot(data[:,13],label='Fisher norm')
        plt.legend()
        
        fig.add_subplot(4,3,11)
        plt.plot(data[:,12],label='Euclidean norm')        
        plt.plot(data[:,14],label='Fisher norm')        
        plt.xlabel('Episode')
        plt.ylabel('$\Delta g_{van}/g_{nat}$')
        plt.legend()
        
        fig.add_subplot(4,3,12)
        plt.plot(data[:,15])        
        plt.xlabel('Episode')
        plt.ylabel('Baseline')
        
    except:
        pass
    
    plt.show()
    
def plot_ac(data):
    fig = plt.figure(figsize=(15,10))
    
    fig.add_subplot(4,3,1)
    smoothed_rewards = pd.Series.rolling(pd.Series(data[:,0]), 10).mean()
    smoothed_rewards = [elem for elem in smoothed_rewards]
    plt.plot(data[:,0])
    plt.plot(smoothed_rewards)
    plt.plot()
    plt.xlabel('Episode')
    plt.ylabel('Sum of return')

    fig.add_subplot(4,3,2)
    plt.plot(data[:,1])
    plt.xlabel('Episode')
    plt.ylabel('Episode length')

    fig.add_subplot(4,3,3)
    smoothed_rq = pd.Series.rolling(pd.Series(data[:,2]), 10).mean()
    smoothed_rq = [elem for elem in smoothed_rq]
    plt.plot(data[:,2],'-o')
    plt.plot(smoothed_rq)
#     plt.yscale('log')
    plt.ylabel('$R_Q$')

    fig.add_subplot(4,3,4)
    plt.plot(data[:,4],label='critic')
    plt.xlabel('Episode')
    plt.ylabel('Loss')
    plt.legend()

    fig.add_subplot(4,3,5)
    plt.plot(data[:,3],label='actor')
    plt.xlabel('Episode')
    plt.ylabel('Loss')
    plt.legend()

    fig.add_subplot(4,3,6)
    plt.plot(data[:,3],label='actor')
    plt.plot(data[:,4],label='critic')
    plt.xlabel('Episode')
    plt.ylabel('Loss')
    plt.legend()
    
    fig.add_subplot(4,3,7)
    plt.plot(data[:,5])
    plt.xlabel('Episode')
    plt.ylabel('Effective Learning Rate')
    
    fig.add_subplot(4,3,8)
    plt.plot(data[:,6])
    plt.xlabel('Episode')
    plt.ylabel('Entropy')
    
    fig.add_subplot(4,3,9)
    plt.semilogy(data[:,7],label='min')
    plt.semilogy(data[:,8],label='max')
    plt.semilogy(data[:,9],label='mean')
    plt.xlabel('Episode')
    plt.ylabel('Measurement Probability')
    plt.legend()

    try:
        fig.add_subplot(4,3,10)
        plt.plot(data[:,10])
        plt.xlabel('Episode')
        plt.ylabel('Critic baseline')

        fig.add_subplot(4,3,11)
        plt.plot(data[:,11])
        plt.xlabel('Episode')
        plt.ylabel('Signals')

        fig.add_subplot(4,3,12)
        plt.plot(data[:,12])
        plt.xlabel('Episode')
        plt.ylabel('Max returns')
    
    except:
        pass
    
    plt.show()
    
def plot_rq(data,label=''):
    
    smoothed_rq = pd.Series.rolling(pd.Series(data), 10).mean()
    plt.plot(1-data)
    plt.plot(1-smoothed_rq)

    rq_idle = 1-0.71653131
    rq_encode_full = 1-0.88995381 
    rq_detect = 1-0.985
    rq_best = 1-0.9999181  #0.98971822
    text_pos = -len(data)/35
    plt.axhline(rq_idle,color='k',ls='--')
    plt.text(text_pos,rq_idle,'Idle')
    plt.axhline(rq_encode_full,color='k',ls='--')
    plt.text(text_pos,rq_encode_full,'Full encoding')
    plt.axhline(rq_detect,color='k',ls='--')
    plt.text(text_pos,rq_detect,'Non-adaptive detection')
    plt.text(text_pos,rq_best,'Best detection')
    yticks = np.array([0.,0.9,0.99,0.999,0.9999])
    plt.yscale('log')
    plt.gca().invert_yaxis()
    plt.yticks(1-yticks,yticks)

def lighten_color(color, amount=0.5):
    """
    Lightens the given color by multiplying (1-luminosity) by the given amount.
    Input can be matplotlib color string, hex string, or RGB tuple.

    Examples:
    >> lighten_color('g', 0.3)
    >> lighten_color('#F034A3', 0.6)
    >> lighten_color((.3,.55,.1), 0.5)
    """
    import matplotlib.colors as mc
    import colorsys
    try:
        c = mc.cnames[color]
    except:
        c = color
    c = colorsys.rgb_to_hls(*mc.to_rgb(c))
    return colorsys.hls_to_rgb(c[0], 1 - amount * (1 - c[1]), c[2])   
    
def plot_rq_multi(datas,labels=None,avg_only=False):
    colors = [f'C{i}' for i in range(10)]+['b','k']
    if labels is None:
        labels=['']*len(datas)
        
    for i,data in enumerate(datas):
        smoothed_rq = pd.Series.rolling(pd.Series(data), 10).mean()
        if avg_only:
            plt.plot(1-smoothed_rq,color=colors[i],label=labels[i])
        else:
            plt.plot(1-data,color=colors[i],label=labels[i])
            plt.plot(1-smoothed_rq,color=lighten_color(colors[i],1.3))

    rq_idle = 1-0.71653131
    rq_encode_full = 1-0.88995381 
    rq_detect = 1-0.985
    rq_best = 1-0.9999181 #0.98971822
    text_pos = -len(data)/35
    plt.axhline(rq_idle,color='k',ls='--')
    plt.text(text_pos,rq_idle,'Idle')
    plt.axhline(rq_encode_full,color='k',ls='--')
    plt.text(text_pos,rq_encode_full,'Full encoding')
    plt.axhline(rq_detect,color='k',ls='--')
    plt.text(text_pos,rq_detect,'Non-adaptive detection')
    plt.text(text_pos,rq_best,'Best detection')
    yticks = np.array([0.,0.9,0.99,0.999,0.9999])
    plt.yscale('log')
    plt.gca().invert_yaxis()
    plt.yticks(1-yticks,yticks)
    plt.legend()
    plt.show()
    
def plot_policy(policy,actions):
    plt.figure(figsize=(10,3))
    plt.plot(policy)
    xticks = tuple(actions)
    plt.xticks(np.arange(len(actions)),xticks, rotation=60)
    plt.grid()
    plt.show()
    
def preprocess(state, num_actions, actor_dm = False):
    '''
    Note:
    - Eigendecomp for concatenated dms shuffles up the eigenvalues
    - Try MKL?
    '''
    dms, prev_act, prev_meas, meas_bias, tnow = state
    if prev_act == -1:
        prev_act = num_actions-1
    # PCA
    kets = []
    for j in range(4):
        # Faster eigen decomp
        ev,es = np.linalg.eigh(dms[j].data.toarray())
        sort_ind = np.argsort(ev.real)[::-1]
        ev = ev.real[sort_ind]
        es = es[:,sort_ind]
        ket = np.hstack([np.sqrt(ev[k])*es[:,k] for k in range(6)])

        kets.append(ket.real)
        kets.append(ket.imag)
    kets = torch.FloatTensor(np.hstack(kets))

    prev_act = F.one_hot(torch.tensor([prev_act]),num_actions).flatten().float()
    prev_meas = F.one_hot(torch.tensor([prev_meas]),3).flatten().float()
    meas_bias = torch.tensor(meas_bias).float()
    tnow = torch.tensor([tnow]).float()

    critic_state = torch.cat([kets,meas_bias,prev_act,tnow])
    if actor_dm:
        actor_state = torch.cat([kets,prev_meas,prev_act,tnow])       
    else:
        actor_state = torch.cat([prev_meas,prev_act,tnow])

    return actor_state,critic_state

def preprocess_jax(state, num_actions, order=6, test=False, actor_dm=False,bias=True):

    dms, prev_act, prev_meas, meas_bias, tnow = state
    batch_shape = dms.shape[:-2]
    prev_act = np.where(prev_act==-1,num_actions-1,prev_act)

    ev,es = np.linalg.eigh(dms)
    assert not ev.imag.any()
    ev = ev.real[...,::-1]
    es = es[...,::-1]
    ev = np.where(np.abs(ev)<1e-14,0,ev)
    kets = np.einsum('...j,...ij->...ij',np.sqrt(ev),es)
    if test:
        diff = np.linalg.norm(np.einsum('...ji,...ki->...jk',kets,kets.conj())-dms)/np.linalg.norm(dms)
        print(f'Reconstruction relative error: {diff:.3e}')
        diff_order = np.linalg.norm(np.einsum('...ji,...ki->...jk',kets[...,:order],kets[...,:order].conj())-dms)/np.linalg.norm(dms)
        print(f'Reconstruction relative error at order {order}: {diff_order:.3e}')

    kets = kets[...,:order].reshape(*batch_shape,-1)
    kets = np.ascontiguousarray(kets).view(np.float64).reshape(batch_shape[0],-1) # order=1 is not contiguous
#     old_kets = kets.view(np.float64).reshape(batch_shape[0],-1)

    kets = torch.FloatTensor(kets)
    prev_act = F.one_hot(torch.LongTensor(prev_act),num_actions).float()
    prev_meas = F.one_hot(torch.LongTensor(prev_meas),3).float()
    
    # Correct meas_bias and tnow
#     print('correct meas_bias and tnow')
    meas_bias = (np.abs(meas_bias)>1e-14).any(axis=-1).astype(int)
    meas_bias = F.one_hot(torch.LongTensor(meas_bias),2).float()
    
    tnow = int(tnow>190)
    tnow = F.one_hot(torch.LongTensor(tnow*np.ones(batch_shape[0])),2).float()
    if bias:
        critic_state = torch.cat([kets,meas_bias,prev_act,tnow],dim=1)
    else:
        critic_state = torch.cat([kets,prev_act,tnow],dim=1)
    
#     actor_state = torch.cat([prev_meas,prev_act,tnow],dim=1)
    if actor_dm:
        actor_state = torch.cat([kets,prev_meas,prev_act,tnow],dim=1)       
    else:
        actor_state = torch.cat([prev_meas,prev_act,tnow],dim=1)

    return actor_state,critic_state

def random_choice_2d(a, axis=1):
    r = np.expand_dims(np.random.rand(a.shape[1-axis]), axis=axis)
    return (a.cumsum(axis=axis) > r).argmax(axis=axis)

def get_action(policy,eps=1.):
    policy = np.exp(np.log(policy)*eps)
    policy /= policy.sum(-1,keepdims=True)
    if len(policy.shape) == 1:
        return np.random.choice(len(policy), p=policy)
    elif len(policy.shape) == 2:
        return random_choice_2d(policy)

# def get_returns(rewards,gamma=1.):
#     r1,r2 = rewards
#     T = len(r1)
#     returns = torch.zeros(T)
#     running_returns = 0
#     for t in reversed(range(T)):
#         running_returns = r1[t] + gamma*running_returns
#         returns[t] = (1-gamma)*running_returns + r2[t]
#     return returns

def get_returns(rewards,gamma=1.):
    r1,r2 = torch.tensor(rewards)
    T = rewards.shape[-1]
    returns = torch.zeros(r1.shape)
    running_returns = torch.zeros(r1.shape[:-1])
    for t in reversed(range(T)):
        running_returns = r1[...,t] + gamma*running_returns
        returns[...,t] = (1-gamma)*running_returns + r2[...,t]
    return returns

def get_baseline(returns,kappa,prev_baseline):
    mean_return = returns.mean(0)
    return (1-kappa)*mean_return + kappa*prev_baseline

def pad_tensorlist(l,num_step):
    myl = l.copy()
    myl[0] = torch.cat([myl[0],torch.zeros(num_step-len(myl[0]))])
    return pad_sequence(myl,batch_first=True)

def flat_grad(grads):
    grad_flatten = []
    for grad in grads:
        grad_flatten.append(grad.view(-1))
    grad_flatten = torch.cat(grad_flatten)
    return grad_flatten

def flat_hessian(hessians):
    hessians_flatten = []
    for hessian in hessians:
        hessians_flatten.append(hessian.contiguous().view(-1))
    hessians_flatten = torch.cat(hessians_flatten).data
    return hessians_flatten

def flat_params(model):
    params = []
    for param in model.parameters():
        params.append(param.data.view(-1))
    params_flatten = torch.cat(params)
    return params_flatten

def update_model(model, new_params):
    index = 0
    for params in model.parameters():
        params_length = len(params.view(-1))
        new_param = new_params[index: index + params_length]
        new_param = new_param.view(params.size())
        params.data.copy_(new_param)
        index += params_length
        
def update_grad(model, new_grads):
    index = 0
    for params in model.parameters():
        grads_length = len(params.grad.view(-1))
        new_grad = new_grads[index: index + grads_length]
        new_grad = new_grad.view(params.grad.size())
        params.grad.copy_(new_grad)
        index += grads_length
        
def kl_divergence(p, q):
    return torch.where(p != 0, p * torch.log(p / q), p).sum(-1)

class adam_opt():
    def __init__(self, lr, beta1=0.9, beta2=0.999):
        self.fix = (lr,beta1,beta2)
        self.var = (0,0)
    def step(self, grad):
        lr,beta1,beta2 = self.fix
        m,v = self.var
        m = beta1*m + (1-beta1)*grad
        v = beta2*v + (1-beta2)*grad**2
        self.var = (m,v)
        return lr*m/np.sqrt(v)
    
def get_entropies(p):
    return -(p*torch.log(p)).sum(-1)

def meas_prob(p):
    prob = p.cpu().detach().numpy()
    prob = prob[...,:4]
    return prob.min(),prob.max(),prob.mean()