import numpy as np
import gym
import random as rn

environment = gym.make("FrozenLake-v0")
training_episodes = 1000000
learning_rate = 0.8
gamma = 0.95

epsilon = 1.0
max_epsilon = 1.0
min_epsilon = 0.1
epsilon_decay = 0.001

Q = np.zeros((environment.observation_space.n, environment.action_space.n))

for episode in range(training_episodes):
  observation = environment.reset()
  score = 0

  while True:
    if rn.uniform(0, 1) > epsilon:
      action = np.argmax(Q[observation, :])
    else:
      action = environment.action_space.sample()

    new_observation, reward, done, info = environment.step(action)

    if reward == 1:
      reward = 10
    else:
      reward = -0.1

    old_q_value = Q[observation, action]
    new_q_value = old_q_value + learning_rate * (reward + gamma * np.max(Q[new_observation, :]) - old_q_value)

    Q[observation, action] = new_q_value

    observation = new_observation

    epsilon = min_epsilon + (max_epsilon - min_epsilon)*np.exp(-epsilon_decay*episode)

    score += reward

    if done:
      print(score)
      break

observation = environment.reset()
while True:
  environment.render()
  action = np.argmax(Q[observation, :])
  print(action)
  observation, reward, done, info = environment.step(action)

  if done:
    break

environment.close()
print(Q)

#Q(s,a):= Q(s,a) + lr [R(s,a) + gamma * max Q(s',a') - Q(s,a)]