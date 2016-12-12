# TORCS API documentation from Loiacono et al.:
# https://arxiv.org/abs/1304.1672

import numpy as np

def lng(obs):
    """
    Reward longitudal velocity projected on track axis,
    """
    speedX = np.array(obs.speedX)
    angle = obs.angle
    reward = speedX * np.cos(angle)
    return reward 

def lng_trans(obs):
    """
    Reward longitudal velocity projected on track axis,
    with a penality for transverse velocity.
    """
    speedX = np.array(obs.speedX)
    # Track distance
    trackPos = np.array(obs.trackPos)
    angle = obs.angle
    reward = speedX * np.cos(angle) - np.abs(speedX * np.sin(angle)) - np.abs(speedX * trackPos)
    return reward

def lng_trans_heavy_penalty(obs):
    """
    Reward longitudal velocity projected on track axis,
    with a penality for transverse velocity.
    """
    speedX = np.array(obs.speedX)
    # Track distance
    trackPos = np.array(obs.trackPos)
    angle = obs.angle
    reward = speedX * np.cos(angle) - np.abs(speedX * np.sin(angle)) - np.abs(speedX * np.power(trackPos,8))
    return reward


def lng_trans_prime(obs):
    """
    Reward longitudal velocity projected on track axis,
    with a penality for transverse velocity.
    """
    speedX = np.array(obs.speedX)
    # Track distance
    trackPos = np.array(obs.trackPos)
    angle = obs.angle
    reward = speedX * np.cos(angle) - np.abs(speedX * np.sin(angle)) - np.abs(speedX * angle)
    return reward

