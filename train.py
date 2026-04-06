import os
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"
import torch
import torch.optim as optim
import numpy as np
import random
from collections import deque

from env import DroneEnv
from dqn import DQN

env = DroneEnv()

model = DQN()
optimizer = optim.Adam(model.parameters(), lr=1e-3)

memory = deque(maxlen=10000)

gamma = 0.99
batch_size = 64
epsilon = 1.0

def select_action(state):
    if random.random() < epsilon:
        return env.action_space.sample()
    else:
        with torch.no_grad():
            return model(torch.FloatTensor(state)).argmax().item()

def train_step():
    if len(memory) < batch_size:
        return

    batch = random.sample(memory, batch_size)

    states, actions, rewards, next_states, dones = zip(*batch)

    states = torch.FloatTensor(states)
    actions = torch.LongTensor(actions)
    rewards = torch.FloatTensor(rewards)
    next_states = torch.FloatTensor(next_states)
    dones = torch.FloatTensor(dones)

    q_values = model(states).gather(1, actions.unsqueeze(1)).squeeze()

    next_q = model(next_states).max(1)[0]

    target = rewards + gamma * next_q * (1 - dones)

    loss = ((q_values - target.detach()) ** 2).mean()

    optimizer.zero_grad()
    loss.backward()
    optimizer.step()


episodes = 200

for ep in range(episodes):
    state, _ = env.reset()
    total_reward = 0

    while True:
        action = select_action(state)

        next_state, reward, done, _, _ = env.step(action)

        memory.append((state, action, reward, next_state, done))

        state = next_state
        total_reward += reward

        train_step()

        if done:
            break

    epsilon *= 0.995
    print(f"Episode {ep}, reward: {total_reward:.2f}")

torch.save(model.state_dict(), "model.pth")