import tensorflow as tf
import numpy as np
import time

from trainers.ppo_trainer import Trainer
from envs import Pong as Env


env = Env(stacks=4, skips=1)
trainer = Trainer(env.action_space.n)
running_reward=-21
steps=0
for e in range(100000):
    s = env.reset()
    ep_score = 0
    done = False
    
    while not done:
        a, v, l = trainer.action(s)

        n_s, r, done, info = env.step(a)
        trainer.add(s, a, l, v, n_s, r, done)
        trainer.update()
        
        s = n_s
        ep_score+=r
        steps+=1
    
    running_reward = ep_score if running_reward==None else running_reward * 0.99 + ep_score * 0.01
    print(f'PPO Episode: {e} | Steps: {steps} | Episode Reward: {ep_score} | Average Reward: {running_reward}')