import gym
import matplotlib.pyplot as plt

from stable_baselines3 import PPO

from gym_pool_test.envs.pool_env import Pool_test

# TODO: cuando definamos seed hay que definirlo para las tres librerias debido a func internas:
# numpa, random y torch

# Create environment
env = gym.make('pool-v0')

# Instantiate the agent
model = PPO('MlpPolicy', env) #, learning_rate=1e-3, verbose=1)
# Train the agent
model.learn(total_timesteps=int(1e6)) #5e5))
# Save the agent
model.save("test_long")
# del model  # delete trained model to demonstrate loading

# Load the trained agent
# model = PPO.load("listo_1")

# Evaluate the agent
# mean_reward, std_reward = evaluate_policy(model, model.get_env(), n_eval_episodes=10)

# Enjoy trained agent
obs = env.reset()
obs_list = []
axis = []
act = []
for i in range(1000):
    action, _states = model.predict(obs)
    obs, rewards, done, info = env.step(action)
    # env.render()

    obs_list.append(rewards)
    act.append(action)
    axis.append(i)
    if done:
        obs = env.reset()

# plt.scatter(axis, act)
# plt.show()
plt.plot(obs_list)
plt.show()

env.close()