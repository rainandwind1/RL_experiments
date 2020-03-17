import gym
import torch
from torch import nn,optim
import numpy as np
from DDQN_py import DDQN,train,plot_curse
import copy

env = gym.make("CartPole-v1")
# env = gym.make("Acrobot-v1")
# env = gym.make("Breakout-ram-v0")
obversation = env.reset()

print("Obversation space:",env.observation_space)
print("Action space:",env.action_space)



# 超参数设置
gamma = 0.98
learning_rate = 0.002
output_size = 2
state_size = 4
memory_len = 30000
#alpha = 0.6   ???


epoch_num = 600   # 回合数
max_steps = 400   # 最大步数
update_target_interval = 100 # 目标网络更新间隔
batch_size = 100

# 初始化
Q_value = DDQN(input_size = state_size,output_size=output_size,memory_len = memory_len)
Q_target =  DDQN(input_size = state_size,output_size=output_size,memory_len = memory_len)
# Q_value.build(input_shape=(1,state_size))
# Q_target.build(input_shape=(1,state_size))

# optimizer = optim.Adam(Q_value.net.parameters(),lr = learning_rate)
score_list = []
loss_list = []



for i in range(epoch_num):
    epsilon = max(0.01,0.16-0.01*(i)/200)
    s = env.reset()
    score = 0
    for j in range(max_steps):
        env.render()
        a = Q_value.sample_action(s,epsilon=epsilon)
        s_next,reward,done,info = env.step(a)
        done_flag = 0.0 if done else 1.0
        Q_value.save_memory((s,a,reward/100,s_next,done_flag))
        score += reward
        s = s_next
        if done:
            break
    score_list.append(score)
    if len(Q_value.memory_list) >= 2000:
        train(Q_value,Q_target,batch_size,gamma,learning_rate,loss_list)
    # 更新目标网络
    if (i+1) % update_target_interval == 0 and i > 0:
        # for raw,target in zip(Q_value.parameters(),Q_target.parameters()):
        #     target.load
        Q_target.load_state_dict(Q_value.state_dict())
        # p = torch.tensor(s,dtype=torch.float32)
        # p = p.squeeze(0)
        # print(Q_value(p))
        print("%d epochs score: %d \n"%(i+1,score))



plot_curse(score_list,loss_list)
env.close()


