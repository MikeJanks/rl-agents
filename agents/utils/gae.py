import numpy as np


def get_gaes(rewards, values, next_value, masks, gamma, lambda_):
    values  = np.append(values, next_value)
    returns = []
    gae = 0
    for i in reversed(range(len(rewards))):
        delta = rewards[i] + gamma * values[i + 1] * masks[i] - values[i]
        gae = delta + gamma * lambda_ * masks[i] * gae
        returns.insert(0, gae + values[i])
    returns = np.array(returns)
    adv = returns - values[:-1]
    return returns, adv