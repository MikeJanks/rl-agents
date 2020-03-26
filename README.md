# rl-agents
A moduler libarary for creating and training reinforcement learning agents. The ability to be moduler and easy to edit is something I have struggle to find in other rl libraries. The importance come from the fact that modern reinfocerment learning agents are a combination of algorithns.

This library decouples the training prosses from the agents allowing you to train multiple agent architectures together without rewritting the training and agents process. This works by having the agents architecture and loss functions defined and having a trainer be the mediem between the environment and the agent. The agent classes define their netowrk structure and loss functions. The trainer will call the agents to interact with the enviroment and will collect all whats neccesary from the enviroment needed for training the agent. Having the trainer be the mediem means you dont rewrite your agents when you want to combine multiple agents in one training process.

An example that inspired this library is having a Proximal policy Optimization Agent be trained with the Random Network Distilation achitecture and found myself rewritting alot of code. Now both the architectures are modules that the trainer can train with together.


# Quick Start
To setup training there are 3 stages
1. environment loop
1. Trainer Class
1. Agent Class

I have a sample called run.py. The environment loop will initialize the trainer and the trainer will be placed before and after the environment step to provide the action need to step and collect any information from the environment after and train if neccesary.

I have a sample trainer called Trainer.py. The Trainer class defines initializes the agent and makes sure it collects and preprocesses whats necessary for the agents it initialized. The trainer then defines the update function by combining all the loss functions from its agents.

I have a sample Agent called agents/ppo.py. Here is the the actuall agent architecture that defines the network then creates function to get predictions and defines the agents loss function. The trainer is expected to know what the agent needs for every function and provides what it needs.
