import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
    
class Actor_RNN(nn.Module):
    def __init__(self, param, num_layer=2, drop_out=0.5):
        super(Actor_RNN, self).__init__()

        nin, nout, hid = param
        self.num_actions = nout        
        self.lstm = nn.LSTM(nin, hid, num_layer, dropout=drop_out)
        self.linear = nn.Linear(hid, nout)
    
    def reset_hidden(self,batch_size=1,device='cpu'):
        num_layer,hid = self.lstm.num_layers,self.lstm.hidden_size
        self.h_0 = torch.zeros(num_layer, batch_size, hid, device=device)
        self.c_0 = torch.zeros(num_layer, batch_size, hid, device=device)
    
    def forward(self, state):
        policy,(h_n,c_n) = self.lstm(state,(self.h_0,self.c_0))
        self.h_0,self.c_0 = h_n,c_n
        policy = self.linear(policy.squeeze(1))
        return F.softmax(policy,dim=-1) + 1e-8

class Actor_FNN(nn.Module):
    def __init__(self, param):
        super(Actor_FNN, self).__init__()

        nin, nout, hid = param
        self.actor = nn.Sequential(nn.Linear(nin, hid),nn.ReLU(),
                           nn.Linear(hid, hid),nn.ReLU(),
                           nn.Linear(hid, nout))
    
    def forward(self, state):
        policy = self.actor(state)
        policy = F.softmax(policy,dim=-1) + 1e-8
        return policy
        
    
class Critic_FNN(nn.Module):
    def __init__(self, param):
        super(Critic_FNN, self).__init__()

        nin, nout, hid = param
        self.critic = nn.Sequential(nn.Linear(nin, hid),nn.ReLU(),
                                   nn.Linear(hid, hid),nn.ReLU(),
                                   nn.Linear(hid, nout))
    def forward(self, state):
        value = self.critic(state)        
        return value
    
def FNN_init(model,scale=1.,kind='uniform'):
    if kind == 'uniform':
        for m in model.modules():
            if isinstance(m, nn.Linear):
                fan_in, fan_out = nn.init._calculate_fan_in_and_fan_out(m.weight)
                bound = scale / np.sqrt(fan_in)
                nn.init.uniform_(m.weight, -bound, bound)            
                nn.init.uniform_(m.bias, -bound, bound)
    elif kind == 'xavieruniform':
        for m in model.modules():
            if isinstance(m, nn.Linear):
                fan_in, fan_out = nn.init._calculate_fan_in_and_fan_out(m.weight)
                bound = scale * np.sqrt(6/(fan_in+fan_out))
                nn.init.uniform_(m.weight, -bound, bound)            
                nn.init.zeros_(m.bias)
    elif kind == 'xaviernormal':
        for m in model.modules():
            if isinstance(m, nn.Linear):
                fan_in, fan_out = nn.init._calculate_fan_in_and_fan_out(m.weight)
                std = scale * np.sqrt(2.0 / float(fan_in + fan_out))
                nn.init.normal_(m.weight,std=std)            
                nn.init.normal_(m.bias,std=std)