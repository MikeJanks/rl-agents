import tensorflow as tf
import numpy as np
import time

from trainers.ppo_rnn_trainer import Trainer
from envs import Pong as Env


env = Env(stacks=4, skips=1)
trainer = Trainer(env.action_space.n)
running_reward=-21
steps=0
for e in range(100000):
    s = env.reset()
    ep_score = 0
    done = False
    start = 0.

    h = trainer.agent.initial_hidden()
    
    while not done:
        a, v, l, h = trainer.action(s, h)

        n_s, r, done, info = env.step(a)
        trainer.add(start, s, a, l, v, n_s, r, done)
        trainer.update(h)
        
        start = 1.
        s = n_s
        ep_score+=r
        steps+=1
    
    running_reward = ep_score if running_reward==None else running_reward * 0.99 + ep_score * 0.01
    print(f'PPO Episode: {e} | Steps: {steps} | Episode Reward: {ep_score} | Average Reward: {running_reward}')