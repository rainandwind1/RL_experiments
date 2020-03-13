import gym

env = gym.make("Breakout-ram-v0")
print(env.action_space)
print(env.observation_space)
obversation = env.reset()
for i in range(1000):
    env.render()
    a = env.action_space.sample()
    s_,r,done,info = env.step(a)
    if done:
        env.reset()
env.close()



'''
Cart-Pole-v1

# 超参数设置
gamma = 0.98
learning_rate = 0.0002
output_size = 4
state_size = 128
memory_len = 30000
#alpha = 0.6   ???


epoch_num = 600   # 回合数
max_steps = 400   # 最大步数
update_target_interval = 60 # 目标网络更新间隔
batch_size = 100
'''



