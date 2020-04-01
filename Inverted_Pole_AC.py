import turtle as t
import os
import numpy as np
import math
from Inverted_Pole_env import Inverted_Pole
from ActorCritic_py import ActorCritic,train,plot_curse
import torch
from torch import nn,optim


gamma = 0.98
num_epoch = 500
update_interval = 30
max_step = 300
learning_rate = 1e-3

pi_scale = 180/math.pi
pi = math.pi
t.setup(1000,1000)
t.pensize(5)
t.speed(10)
t.pencolor('purple')
map_scale = 4

input_size = 2
output_size = 3
AC = ActorCritic(input_size = input_size,output_size = output_size)
optimizer = optim.Adam(AC.parameters(),lr = learning_rate)
loss_list =[]
score_list = []



def main():
    env = Inverted_Pole()
    for epo_i in range(num_epoch):
        score = 0.0
        s = env.reset()
        for i in range(max_step):
            a_index,prob = AC.sample_action(s)
            s_next,reward,done_flag = env.step(a_index)
            #print(a_index,reward)
            AC.save_memory((prob,reward,s,s_next))
            score += reward
            s = s_next
            if done_flag == 0:
                break
        score_list.append(score)
        train(AC,optimizer,gamma,loss_list)
        print("%d epoch: avg score: %f"%(epo_i+1,score))
    plot_curse(score_list,loss_list)
    t.mainloop()

if __name__ == "__main__":
    main()



# t.goto(0,0)
# t.pendown()
# t.dot(18)
# t.penup()
# t.goto(-250,-250)
# t.pendown()
# t.goto(250,-250)
# t.goto(250,250)
# t.goto(-250,250)
# t.goto(-250,-250)
# t.penup()
# t.goto(0,0)

# def Skip(step):
#     t.penup()
#     t.forward(step)
#     t.pendown()

# Q_rew = np.matrix([[5,0],[0,0.1]])
# R_rew = 1.0

# def bar_init(name, length):
#     # 注册Turtle形状，建立表针Turtle
#     # Skip(-length * 0.1)
#     # 开始记录多边形的顶点。当前的乌龟位置是多边形的第一个顶点。
#     t.begin_poly()
#     t.pensize(10)
#     t.forward(length * 1.1)
#     # 停止记录多边形的顶点。当前的乌龟位置是多边形的最后一个顶点。将与第一个顶点相连。
#     t.end_poly()
#     # 返回最后记录的多边形。
#     handForm = t.get_poly()
#     t.hideturtle()
#     t.register_shape(name, handForm)

# print(int(-np.matrix([1,1])*Q_rew*np.transpose(np.matrix([1,1]))) - R_rew*1**2)
# bar = bar_init("Pole",168)
# Pole = t.Turtle()
# Pole.shape("Pole")
# Pole.shapesize(1,1,9)
# Pole.setheading(60)
# t.mainloop()

