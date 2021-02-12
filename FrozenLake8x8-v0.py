import gym
import numpy as np
import random

environment = gym.make('FrozenLake8x8-v0')
Q = np.zeros((environment.observation_space.n, environment.action_space.n))
num_episodes = 1000000
learning_rate = 0.01
gamma = 0.95

epsilon = 1.0
max_epsilon = 1.0
min_epsilon = 0.1
epsilon_decay = 0.1

won = 0
lost = 0

for episode in range(num_episodes):
  observation = environment.reset()
  done = False

  while not done:
    if random.uniform(0, 1) > epsilon:
      action = np.argmax(Q[observation, :])
    else:
      action = environment.action_space.sample()

    new_observation, reward, done, info = environment.step(action)

    if done:
      if reward == 1:
        reward = 100
        won = won + 1
      else:
        reward = -100
        lost = lost + 1
    else:
      reward = -0.1

    #Q(s,a):= Q(s,a) + lr [R(s,a) + gamma * max Q(s',a') - Q(s,a)]
    old_q_value = Q[observation, action]
    new_q_value = old_q_value + learning_rate * (reward + gamma * np.max(Q[new_observation, :]) - old_q_value)
    Q[observation, action] = new_q_value

    observation = new_observation

    epsilon = min_epsilon + (max_epsilon - min_epsilon)*np.exp(-epsilon_decay*episode)

    if done:
      print("Ep.", episode, ":", won/lost)
      break

print(Q)

for episode in range(int(input('Examples? '))):
  observation = environment.reset()
  
  while True:
    environment.render()
    action = np.argmax(Q[observation, :])
    observation, reward, done, info = environment.step(action)

    if done:
      break