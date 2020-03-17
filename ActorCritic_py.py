import numpy as np
import torch
from torch import nn,optim
import random
import collections




class ActorCritic(nn.Module):
    def __init__(self,input_size,output_size):
        super(Actor,self).__init__()
        self.actor = nn.Sequential(
            nn.Linear(input_size,128),
            nn.ReLU(),
            nn.Linear(128,output_size)
        )
        self.critic = nn.Sequential(
            nn.Linear(input_size,128),
            nn.ReLU(),
            nn.Linear(128,1)
        )
        self.memory = []

    def forward(self, inputs):
        inputs = torch.tensor(inputs,dtype=torch.float32)
        a_prob = nn.Softmax(self.actor(inputs),dim=1)
        v = self.critic(inputs)
        return a_prob, v

    def save_memory(self,transition):
        self.memory.append(transition)


def train(net,optimizer,memory,gamma,loss_list):
    for prob,reward,s,s_next in memory:
        A = reward + gamma*net.critic(s_next) - net.critic(s)
        loss = -torch.log(prob)*A 
        loss_list.append(loss)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()


def plot_curse(self,target_list,loss_list):
    figure1 = plt.figure()
    plt.grid()
    X = []
    for i in range(len(target_list)):
        X.append(i)
    plt.plot(X,target_list,'-r')
    plt.xlabel('epoch')
    plt.ylabel('score')

    figure2 = plt.figure()
    plt.grid()
    X = []
    for i in range(len(loss_list)):
        X.append(i)
    plt.plot(X,loss_list,'-b')
    plt.xlabel('train step')
    plt.ylabel('loss')
    plt.show()


