import gym
from utils.Preprocess import Preprocess

from nes_py.wrappers import JoypadSpace
import gym_super_mario_bros
from gym_super_mario_bros.actions import SIMPLE_MOVEMENT
env = gym_super_mario_bros.make('SuperMarioBros-v0')
env = JoypadSpace(env, SIMPLE_MOVEMENT)

def Env(**kargs):
    return Pong(**kargs)

class Pong():
    def __init__(self, img_size=32, skips=4):
        self.env = gym.make('Pong-v0')
        self.preprocess = Preprocess(img_size)
        self.skips=skips
        self.pong_action = { 0: 0, 1: 2, 2: 3 }
        self.action_space=self.env.action_space
        self.action_space.n=3

    def reset(self):
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



class Mario():
    def __init__(self, img_size=32, skips=4):
        env = gym_super_mario_bros.make('SuperMarioBros-v0')
        self.env = JoypadSpace(env, SIMPLE_MOVEMENT)
        self.preprocess = Preprocess(img_size)
        self.skips=skips
        self.action_space=self.env.action_space
        self.action_space.n=3

    def reset(self):
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


