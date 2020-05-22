import gym
from utils.Preprocess import Preprocess

from nes_py.wrappers import JoypadSpace
import gym_super_mario_bros
from gym_super_mario_bros.actions import SIMPLE_MOVEMENT
env = gym_super_mario_bros.make('SuperMarioBros-v0')
env = JoypadSpace(env, SIMPLE_MOVEMENT)

def Env(**kargs):
    # return Pong(**kargs)
    return Mario(**kargs)

class Pong():
    def __init__(self, img_size=32, stacks=4, skips=4, return_seq=False):
        self.env = gym.make('Pong-v0')
        self.preprocess = Preprocess(img_size, stacks, return_seq)
        self.skips=skips
        self.pong_action = { 0: 0, 1: 2, 2: 3 }
        self.action_space=self.env.action_space
        self.action_space.n=3
        self.observation_space=(img_size, img_size, stacks)

    def reset(self):
        self.preprocess.reset()
        s = self.env.reset()
        s = self.preprocess(s)
        return s

    def step(self, a):
        total_r=0
        for i in range(self.skips):
            self.env.render()

            n_s, r, done, info = self.env.step(self.pong_action[a])
            n_s = self.preprocess(n_s)
            total_r+=r

            if done: break
            
        return n_s, total_r, done, info
    
    
class Atari():
    def __init__(self, env):
        self.env = gym.make(env)
        self.action_space=self.env.action_space
        self.observation_space=self.env.observation_space.shape

    def reset(self):
        s = self.env.reset()
        return s

    def step(self, a):
        self.env.render()
        n_s, r, done, info = self.env.step(a)
        return n_s, r, done, info



class Mario():
    def __init__(self, img_size=32, stacks=4, skips=4, return_seq=False):
        env = gym_super_mario_bros.make('SuperMarioBros-v2')
        self.env = JoypadSpace(env, SIMPLE_MOVEMENT)
        self.preprocess = Preprocess(img_size, stacks, return_seq)
        self.skips=skips
        self.action_space=self.env.action_space
        self. observation_space=(img_size, img_size, stacks)

    def reset(self):
        self.preprocess.reset()
        s = self.env.reset()
        s = self.preprocess(s)
        return s

    def step(self, a):
        total_r=0
        for i in range(self.skips):
            self.env.render()

            n_s, r, done, info = self.env.step(a)
            n_s = self.preprocess(n_s)
            total_r+=r

            if done: break
            
        return n_s, total_r, done, info


