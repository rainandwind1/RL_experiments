import numpy as np
import turtle as t
import torch
from torch import nn,optim
from ActorCritic_py import ActorCritic,train,plot_curse,save_param,load_param
import os
import gym



map_scale = 4

# Hyperparameter
learning_rate = 0.00018
memory_len = 30000
gamma = 0.98
batch_size = 100
output_size = 2
state_size = 4


epoch_num = 1000
max_steps = 400
update_target_interval = 10
path = 'E:\Code\param\AC_params_test.pkl'

# 初始化
AC = ActorCritic(input_size=state_size,output_size=output_size)
optimizer = optim.Adam(AC.parameters(),lr = learning_rate)
# optimizer_v = optim.Adam(AC.critic.parameters(),lr = learning_rate)
score_list = []
loss_list = []

if os.path.exists(path):
    load_param(AC,path)
    #print(AC)



def main():
    env = gym.make("CartPole-v1")
    score_avg = 0.0
    for epo_i in range(epoch_num):
        score = 0.0
        s = env.reset()
        epsilon = 0.01 - 0.001*epo_i/100
        for i in range(max_steps):
            env.render()
            action,prob = AC.sample_action(s,epsilon)
            s_next,reward,done_flag,info = env.step(action)
            AC.save_memory((prob,reward,s,s_next))
            score += reward
            s = s_next
            if done_flag == 1:
                break
        train(AC,optimizer,gamma,loss_list)
        score_list.append(score)
        score_avg += score
        if (epo_i+1) % update_target_interval == 0:
            print("%d epoch: avg score: %f"%(epo_i+1,score_avg/update_target_interval))
            score_avg = 0.0
    plot_curse(score_list,loss_list)
    save_param(AC,path)
    env.close()


if __name__ == "__main__":
    main()


# # 画笔参数
# t.setup(1000,1000)
# t.pensize(3)
# t.speed(1)
# t.pencolor('purple')

# def init_target(name,color,redius):
#     t.begin_poly()
#     t.dot(redius,color)  
#     t.end_poly()
#     shape = t.get_poly()
#     t.register_shape(name,shape)

# t.goto(0,0)
# t.goto(0,500)
# t.goto(500,500)
# t.goto(500,0)
# t.goto(0,0)
# t.penup()
# t.setpos(20,20)


# target = t.Turtle()
# target.penup()
# target.setpos(100,100)
# target.dot(30,"blue")
# target.reset()
# target.setpos(10,20)
# t.mainloop()