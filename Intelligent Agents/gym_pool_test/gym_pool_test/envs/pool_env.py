
import numpy as np
import gym
from gym import error, spaces, utils
from gym.utils import seeding
import math
import random

class Pool_test(gym.Env):
    '''
        Agent learning to buy and sell in an scenario where price follows a sin function

    '''
    metadata = {'render.modes': ['console']}

    def __init__(self): # , arg1, arg2, ...):
        super(Pool_test, self).__init__()
        # Define action and observation space
        # They must be gym.spaces objects
        # Example when using discrete actions:
        n_actions = 3
        self.action_space = spaces.Discrete(n_actions)
        self.observation_space = spaces.Box(low=0, high=np.inf,
                                        shape=(1,), dtype=np.float32)


    def step(self, action):
        # precio_anterior = 0.2*math.sin((self.timesteps-5)*math.pi/180) + 1
        # self.precio = 0.2*math.sin(self.timesteps*math.pi/180) + 1
        # reward = self.reward
        # precio = math.sin(self.timesteps*math.pi/180)
        precio = self.precio
        # print('1')
        # print(precio)
        # precio = 1 if precio > 0 else -1

        valor = 0
        if action == 0:
            # Vende
            valor += 1*precio
            self.stocks += 1
            self.cash -= 1*precio
        elif action == 1:
            valor += 0
        elif action == 2:
            # Compra
            self.cash += self.stocks*precio
            self.stocks = 0 #50/precio
            
        else:
            raise ValueError("Received invalid action={} which is not part of the action space".format(action))

        done = 0



        # self.reward = self.reward + valor #- 0.05*self.timesteps
        # if self.reward > 100:
        #     self.reward = 200
        #     done = 1
        
        # if self.reward < -100:
        #     self.reward = -60
        #     done = 1
        reward_anterior = self.reward
        rew_nueva = self.cash + self.stocks*precio

        if rew_nueva < reward_anterior: self.reward -= 1

        elif rew_nueva > reward_anterior: self.reward += 1

        # self.reward -= 1
        # if self.timesteps > 120:
        #     self.reward = -60
        #     done = 1

        # Optionally we can pass additional info, we are not using that for now
        self.timesteps += 1
        # return observation, reward, done, info
        # state = [precio, precio_anterior]

        # precio = random.randint(-10,10)
        self.precio = 1+(0.5*math.sin(self.timesteps*math.pi/180) + 1)
        self.info['buffer'].append(self.precio)
        

        # print('2')
        # print(self.precio)
        # state = [self.precio]

        mov_avg = (self.info['buffer'][-1] - self.info['buffer'][-5])/5
        state = [mov_avg]
        return np.array(state).astype(np.float32), self.reward, done, self.info

    def reset(self):
        self.reward = 0
        self.timesteps = 0
        self.precio = 0
        self.cash = 100
        self.stocks = 0
        self.info = {'buffer': [0,0,0,0,0,0]}

        # return observation  # reward, done, info can't be included
        # state = [self.precio, self.precio_anterior]
        state = [self.precio]
        return np.array(state).astype(np.float32)

    def close (self):
        pass


# ****************************************************************************


# class Pool_test(gym.Env):
#     '''
#         Agent learning to buy and sell in an scenario where price follows a sin function

#     '''
#     metadata = {'render.modes': ['console']}

#     def __init__(self): # , arg1, arg2, ...):
#         super(Pool_test, self).__init__()
#         # Define action and observation space
#         # They must be gym.spaces objects
#         # Example when using discrete actions:
#         n_actions = 3
#         self.action_space = spaces.Discrete(n_actions)
#         self.observation_space = spaces.Box(low=0, high=np.inf,
#                                         shape=(1,), dtype=np.float32)


#     def step(self, action):
#         # precio_anterior = 0.2*math.sin((self.timesteps-5)*math.pi/180) + 1
#         # self.precio = 0.2*math.sin(self.timesteps*math.pi/180) + 1
#         # reward = self.reward
#         # precio = math.sin(self.timesteps*math.pi/180)
#         precio = self.precio
#         # print('1')
#         # print(precio)
#         precio = 1 if precio > 0 else -1

#         valor = 0
#         if action == 0:
#             # Vende
#             valor += 5*precio
#         elif action == 1:
#             valor += -1
#         elif action == 2:
#             # Compra
#             valor += -5*precio #50/precio
#         else:
#             raise ValueError("Received invalid action={} which is not part of the action space".format(action))

#         done = 0

#         self.reward = self.reward + valor #- 0.05*self.timesteps
#         if self.reward > 100:
#             self.reward = 200
#             done = 1
        
#         if self.reward < -100:
#             self.reward = -60
#             done = 1

#         self.reward -= 1
#         # if self.timesteps > 120:
#         #     self.reward = -60
#         #     done = 1

#         # Optionally we can pass additional info, we are not using that for now
#         info = {}
#         self.timesteps += 1
#         # return observation, reward, done, info
#         # state = [precio, precio_anterior]

#         precio = random.randint(-10,10)
#         self.precio = precio #0.2*math.sin(self.timesteps*math.pi/180) + 1
#         # print('2')
#         # print(self.precio)
#         state = [self.precio]
#         return np.array(state).astype(np.float32), self.reward, done, self.info

#     def reset(self):
#         self.reward = 0
#         self.timesteps = 0
#         self.precio = 0
#         self.precio_anterior = 0
#         self.info = {'buffer': []}

#         # return observation  # reward, done, info can't be included
#         # state = [self.precio, self.precio_anterior]
#         state = [self.precio]
#         return np.array(state).astype(np.float32)

#     def close (self):
#         pass



# ****************************************************************************


# class Pool_test(gym.Env):
#     # metadata = {'render.modes': ['human']}
#     metadata = {'render.modes': ['console']}

#     def __init__(self): # , arg1, arg2, ...):
#         super(Pool_test, self).__init__()
#         # Define action and observation space
#         # They must be gym.spaces objects
#         # Example when using discrete actions:
#         n_actions = 3
#         self.action_space = spaces.Discrete(n_actions)
#         self.observation_space = spaces.Box(low=0, high=100,
#                                         shape=(1,), dtype=np.float32)

#     def step(self, action):
#         cuenta_old = self.cuenta
#         # print(cuenta_old)
#         if action == 0:
#             self.cuenta -= 1
#         elif action == 1:
#             self.cuenta += 1
#         elif action == 2:
#             self.cuenta += 10
#         else:
#             raise ValueError("Received invalid action={} which is not part of the action space".format(action))

#         # Account for the boundaries of the grid
#         # self.agent_pos = np.clip(self.cuenta, 0, self.grid_size)

#         # Are we at the left of the grid?
#         # print(self.cuenta)
#         done = bool(self.cuenta > 100)

#         # Null reward everywhere except when reaching the goal (left of the grid)
#         # reward = 1 if self.cuenta > cuenta_old else 0
#         if action == 1:
#             extra = 0
#             if self.cuenta > 100:
#                 extra = 0

#             reward = self.reward + 10 + extra - 100*self.timesteps
        
#         else: 
#             reward = self.reward - 1

#         # Optionally we can pass additional info, we are not using that for now
#         info = {}

#         self.timesteps += 1

#         # return observation, reward, done, info
#         return np.array([self.cuenta]).astype(np.float32), reward, done, info

#     def reset(self):
#         self.cuenta = 0
#         self.reward = 0
#         self.timesteps = 0
#         # return observation  # reward, done, info can't be included
#         return np.array([self.cuenta]).astype(np.float32)

#     # Optional: allow to visualize the agent in action
#     # def render(self, mode='human'):

#     def close (self):
#         pass

'''
To check that your environment follows the gym interface, please use:

from stable_baselines.common.env_checker import check_env

env = CustomEnv(arg1, ...)
# It will check your custom environment and output additional warnings if needed
check_env(env)

'''