import tensorflow as tf
tf.keras.backend.set_floatx('float64')

import numpy as np
import time

from trainers.sac_trainer import Trainer
from envs import Atari as Env
from envs import Pong

# env_name = "LunarLanderContinuous-v2"
env_name = "LunarLander-v2"
# env_name = "BipedalWalkerHardcore-v2"
# env_name = "BipedalWalker-v2"
# env_name = "Pong-ram-v4"


# env = Env(env_name)
env = Pong(img_size=64)
trainer = Trainer(env.action_space.n, env.observation_space, is_discrete=True)
# trainer = Trainer(env.action_space.shape[0], env.observation_space, is_discrete=False)
# trainer.load_model('saved_models/sac_'+env_name+'/')
running_reward=None
steps=0
for e in range(100000):
    s = env.reset()
    ep_score = 0
    done = False
    
    while not done:
        a, l = trainer.action(s)
        
        n_s, r, done, info = env.step(a)
        trainer.add(s, a, l, r, n_s, done)
        trainer.update()
        
        s = n_s
        ep_score+=r
        steps+=1
    
    # if e%10==0:
    #     trainer.save_model('saved_models/sac_'+env_name+'/')
    
    running_reward = ep_score if running_reward==None else running_reward * 0.99 + ep_score * 0.01
    print(f'PPO Episode: {e} | Steps: {steps} | Episode Reward: {ep_score} | Average Reward: {running_reward}')