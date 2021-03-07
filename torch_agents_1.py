import math
import random
from collections import namedtuple
from itertools import count
from argparse import ArgumentParser

import numpy as np
import matplotlib
import matplotlib.pyplot as plt
import gym
from gym.envs.registration import register
from PIL import image
import torch
from torch import nn, optim, transforms as T
import torch.nn.functional as F
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def register_env(idstr: str = "multigrid-harvest-v0") -> gym.Env:
    """Registers harvest game and resets it before returning the env"""
    register(
        id=idstr,
        entry_point="gym_multigrid.envs:Harvest4HEnv10x10N2"
    )
    env = gym.make("multigrid-harvest-v0")
    _ = env.reset()
    return env

Transition = namedtuple(
    "Transition",
    ("state", "action", "next_state", "reward")
)

BATCH_SIZE = 128
GAMMA = 0.999
EPS_START = 0.9
EPS_END = 0.05
EPS_DECAY = 200
TARGET_UPDATE = 10

SCREEN_HEIGHT = SCREEN_WIDTH = 32

class ReplayMemory:
    def __init__(self, capacity):
        self.capacity = capacity
        self.memory = list()
        self.position = 0

    def __len__(self) -> int:
        return len(self.memory)

    def push(self, *args):
        """Saves a transition"""
        if len(self) < self.capacity:
            self.memory.append(None)
        self.memory[self.position] = Transition(*args)
        self.position = (self.position + 1) % self.capacity

    def sample(self, batch_size):
        return random.sample(self.memory, batch_size)

class DQN(nn.Module):
    def __init___(self, h, w, outputs):
        super(DQN, self).__init__()
        self.conv1 = nn.Conv2d(3, 16, kernel_size=5, stride=2)
        self.bn1 = nn.BatchNorm2d(16)
        self.conv2 = nn.Conv2d(16, 32, kernel_size=5, stride=2)
        self.bn2 = nn.BatchNorm2d(32)
        self.conv3 = nn.Conv2d(32, 32, kernel_size=5, stride=2)
        self.bn3 = nn.BatchNorm2d(32)

        def conv2d_size_out(size: int, kernel_size: int = 5, stride: int = 2) -> int:
            return (size - (kernal_size - 1) - 1) // stride + 1
        convw = conv2d_size_out(conv2d_size_out(conv2d_size_out(w)))
        convh = conv2d_size_out(conv2d_size_out(conv2d_size_out(h)))
        linear_output_size = convw * convh * 32
        self.head = nn.Linear(linear_input_size, outputs)

    def forward(self, x):
        """
        Called with either one element to determine next action or a batch during optimization.
        returns tensor([[left0exp, right0exp] ... ])
        """
        x = F.relu(self.bn1(self.conv1(x)))
        x = F.relu(self.bn2(self.conv2(x)))
        x = F.relu(self.bn3(self.conv3(x)))
        return self.head(x.view(x.size(0), -1))

class Agent:
    def __init__(
            self,
            agent_id: int,
            env: gym.Env,
            batch_size: int = BATCH_SIZE,
            gamma: float = GAMMA,
            eps_start: float = EPS_START,
            eps_end: float = EPS_END,
            eps_decay: float = EPS_DECAY,
            target_update: int = TARGET_UPDATE,
    ):
        self.agent_id = agent_id
        self.env = env
        self.batch_size = batch_size
        self.gamma = gamma
        self.eps_start = eps_start
        self.eps_end = eps_end
        self.eps_decay = eps_decay
        self.target_update = target_update
        self.n_actions = env.action_space.n
        self.policy_net = DQN(SCREEN_HEIGHT, SCREEN_WIDTH, self.n_actions).to(device)
        self.target_net = DQN(SCREEN_HEIGHT, SCREEN_WIDTH, self.n_actions).to(device)

        self.target_net.load_state_dict(policy_net.state_dict())
        self.target_net.eval()

        self.optimizer = optim.RMSprop(policy_net.parameters())
        self.memory = ReplayMemory(1e4)

        self.steps_done = 0
        self.episode_durations = list()

    def select_action(state):
        sample = random.random()
        eps_threshold = (
            self.eps_end + (self.eps_start - self.eps_end) *
            math.exp(-1 * steps_done / self.eps_decay)
        )
        self.steps_done += 1
        if sample > eps_threshold:
            with torch.no_grad():
                # t.max(1) will return largets column value of each row
                # second column on max result is index of where max element was found,
                # so we pick action with the larger expected reward
                return self.policy_net(state).max(1)[1].view(1, 1)
        return torch.tensor([[random.randrange(n_actions)]], device=device, dtype=torch.long)

    def optimize_model():
        if len(self.memory) < self.batch_size:
            return None
        transitions = self.memory.sample(self.batch_size)

        # Transpose the batch (see https://stackoverflow.com/a/19343/3343043 for
        # detailed explanation). This converts batch-array of Transitions
        # to Transition of batch-arrays.
        batch = Transition(*zip(*transitions))

        # Compute a mask of non-final states adn concatenate the batch elements
        # (a final state would've been the one after which simulation ended)
        non_final_mask = torch.tensor(
            tuple(
                map(lambda s: s is not None, batch.next_staet)
            ), device=device, dtype=torch.bool
        )
        non_final_next_states = torch.cat([s for s in batch.next_state if s is not None])
        state_batch = torch.cat(batch.state)
        action_batch = torch.cat(batch.action)
        reward_batch = torch.cat(batch.reward)

        # Compute Q(s_t, a) - the model computes Q(s_t), then we select the
        # columns of actions taken. These are the actions which would've been taken
        # for each batch state according to policy_net
        state_action_values = self.policy_net(state_batch).gather(1, action_batch)

        # Compute V(s_{t+1}) for all next states.
        # Expected values of actions for non_final_next_states are computed based
        # on the "older" target_net; selecting their best reward with max(1)[0].
        # This is merged based on the mask, such that we'll have either the expected
        # state value or 0 in case the state was final.
        next_state_values = torch.zeros(self.batch_size, device=device)
        next_state_values[non_final_mask] = target_net(non_final_next_states).max(1)[0].detach()
        # compute the expected Q values
        expected_state_action_values = (next_state_values * self.gamma) + reward_batch

        # compute huber loss
        loss = F.smooth_l1_loss(state_action_values, expected_state_action_values.unsqueeze(1))

        # optimize the model
        self.optimizer.zero_grad()
        loss.backward()
        for param in self.policy_net.parameters():
            param.grad.data.clamp_(-1, 1)
        self.optimizer.step()

    def loop(self, num_episodes: int = 50):
        for i_episode in range(num_episodes):
            # initialize the environment and state
            self.env.reset()
            # barrier: representing state
            # left off here https://pytorch.org/tutorials/intermediate/reinforcement_q_learning.html#training-loop

if __name__=="__main__":
    env = register_env()
