import numpy as np
import torch
from torch import nn,optim
import random
import collections




class Actor(nn.Module):
    def __init__(self,input_size,output_size):
        super(Actor,self).__init__()
        self.net = nn.Sequential(
            nn.Linear(input_size,128),
            nn.ReLU(),
            nn.Linear(128,output_size)
        )
    
    def forward(self, inputs):
        inputs = torch.tensor(inputs,dtype=torch.float32)
        output = self.net(inputs)
        return output

    
class Critic(nn.Module):
    def __init__(self,input_size,output_size):
        super(Critic,self).__init__()
        self.net = nn.Sequential(
            nn.Linear(input_size,128),
            nn.ReLU(),
            nn.Linear(128,output_size)
        )

    def forward(self, inputs):
        inputs = torch.tensor(inputs,dtype=torch.float32)
        output = self.net(inputs)
        return output


def train(net,learning_rate):
    return 0





