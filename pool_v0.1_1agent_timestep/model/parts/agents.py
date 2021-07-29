
import numpy as np
import random

import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision

import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision


'''

    Name: agents_1action.py + Intlligence
    Mempool implementation: There is slippage as the action does not happen when it was sent.
    Works - tested

'''

# Function defining the internal structure of the intelligent agent
def get_model(ver=1):

    if ver == 1:
        class Net1(nn.Module):
            def __init__(self):
                super(Net1, self).__init__()
                self.n0 = nn.Linear(1,5)
                self.n1 = nn.Linear(5,1)

            def forward(self, x):              
                x = F.relu(self.n0(x))
                prediction = torch.sigmoid(self.n1(x))

                return prediction

        return Net1()


# Mempool
def selector(params, substep, state_history, prev_state):
    mempool = prev_state['mempool']
    mempool = mempool[1:]

    action = trader(params, substep, state_history, prev_state)

    action = action['action']

    lista_aux = ['trader', action['action'], action['tokens'], action['cash'], action['exp_price'], action['gas']]
    mempool.append(lista_aux)

    action = investor(params, substep, state_history, prev_state)
    action = action['action']
    lista_aux = ['investor', action['action'], action['tokens'], action['cash'], action['exp_price'], action['gas']]
    mempool.append(lista_aux)

    action = buyer(params, substep, state_history, prev_state)
    action = action['action']
    lista_aux = ['buyer', action['action'], action['tokens'], action['cash'], action['exp_price'], action['gas']]
    mempool.append(lista_aux)

    action = Foundation(params, substep, state_history, prev_state)
    action = action['action']
    lista_aux = ['Foundation', action['action'], action['tokens'], action['cash'], action['exp_price'], action['gas']]
    mempool.append(lista_aux)

    mempool.sort(key=lambda x: x[5], reverse=True)

    action = {
                mempool[0][0]:{
                    'action': mempool[0][1],
                    'tokens': mempool[0][2],
                    'cash': mempool[0][3],
                    'exp_price': mempool[0][4],
                    'gas': mempool[0][5],
                },

                'mempool': mempool
    }

    if len(mempool) > 20: mempool = mempool[:20]

    return {agent: action}


def agent(params, substep, state_history, prev_state, policy_input):
    key = list(policy_input.keys())[0]

    return ('agent', policy_input[key])


def mempool(params, substep, state_history, prev_state, policy_input):
    key = list(policy_input.keys())[0]

    mempool = policy_input[key]['mempool']

    return ('mempool', mempool)

###############################################################################

def new_price(action, tA, tB):
    new_tA = tA + action
    new_price = tB/new_tA

    return new_price

def prev_dict_agent(state_history, prev_state, agent):
    key_token = str(agent) + '_tokens'
    key_cash = str(agent) + '_cash'

    dict_agent = {'action': 0,
            'tokens': prev_state[key_token],
            'cash': prev_state[key_cash],
            'exp_price': 0,
            'gas': 0,
            }

    return dict_agent 

def update_dict_agent(state_history, prev_state, agent, dicc):

    dict_agent = {'action': dicc['action'],
            'tokens': dicc['tokens'],
            'cash': dicc['cash'],
            'exp_price': dicc['exp_price'],
            'gas': dicc['gas'],
            }

    return dict_agent 

def calc_gas(action):

    # Gas %: between 2 and 10% depending on the amount of tokens to exchange assuming 10% (10 in the formulas)
    # as maximum for small amounts and 2% (2-10 = -8) for amounts bigger than 10.000 (10000 in the formulas)
    # Random noise with values between -2 and 2% with one decimal are included so real rates running from 0-12%

    action = abs(action)
    if action > 10000: action = 10000
    gas_rate = action*(-8/10000) + 10
    noise = random.randint(-20,20)/10  
    gas_rate = (gas_rate + noise)/100

    return gas_rate

###############################################################################

# Trader test

def trader_calc(state_history, prev_state):
    tA = prev_state['num_token_A']
    tB = prev_state['num_token_B']
    val_tA = prev_state['value_token_A']
    val_tB = prev_state['value_token_B']
    timestep = state_history[-1][-1]['timestep']
    agent_tokens = prev_state['trader_tokens']
    agent_cash = prev_state['trader_cash']
    action = 0

    order = random.randint(-1000, 1000)
    if order > 0:
        if prev_state['trader_cash'] > order*val_tA:
            action = order
            agent_tokens += action
            exp_price = new_price(action, tA, tB)
            gas_rate = calc_gas(action)
            gas =  gas_rate*abs(action)*exp_price
            agent_cash -= action*exp_price - gas
        else:
            action = 0
            exp_price = 0
            gas = 0
    else:
        if prev_state['trader_tokens'] > abs(order):
            action = order
            agent_tokens += action
            exp_price = new_price(action, tA, tB)
            gas_rate = calc_gas(action)
            gas =  gas_rate*abs(action)*exp_price
            agent_cash -= action*exp_price - gas
        else:
            action = 0
            exp_price = 0    
            gas = 0

    dict_agent = {'action': action,
            'tokens': agent_tokens,
            'cash': agent_cash,
            'exp_price': exp_price,
            'gas': gas,
            }

    return dict_agent   


def trader(params, substep, state_history, prev_state):
    dict_agent = trader_calc(state_history, prev_state)

    return {'action': dict_agent}

def trader_tokens(params, substep, state_history, prev_state, policy_input):  
    key = list(policy_input.keys())[0]
    if list(policy_input[key].keys())[0] == 'trader':
        dict_agent = update_dict_agent(state_history, prev_state, 'trader', policy_input[key]['trader'])
    else:
        dict_agent = prev_dict_agent(state_history, prev_state, 'trader')

    return ('trader_tokens', dict_agent['tokens'])

def trader_cash(params, substep, state_history, prev_state, policy_input):
    key = list(policy_input.keys())[0]
    if list(policy_input[key].keys())[0] == 'trader':
        dict_agent = update_dict_agent(state_history, prev_state, 'trader', policy_input[key]['trader'])
    else:
        dict_agent = prev_dict_agent(state_history, prev_state, 'trader')

    return ('trader_cash', dict_agent['cash'])

def trader_action(params, substep, state_history, prev_state, policy_input):   
    key = list(policy_input.keys())[0]
    if list(policy_input[key].keys())[0] == 'trader':
        dict_agent = update_dict_agent(state_history, prev_state, 'trader', policy_input[key]['trader'])
    else:
        dict_agent = prev_dict_agent(state_history, prev_state, 'trader')

    return ('trader_action', dict_agent['action'])

def trader_expected_price(params, substep, state_history, prev_state, policy_input):
    key = list(policy_input.keys())[0]
    if list(policy_input[key].keys())[0] =='trader':
        dict_agent = update_dict_agent(state_history, prev_state, 'trader', policy_input[key]['trader'])
    else:
        dict_agent = prev_dict_agent(state_history, prev_state, 'trader')

    return ('trader_expected_price', dict_agent['exp_price'])

###############################################################################

# investor

def investor_calc(state_history, prev_state):
    tA = prev_state['num_token_A']
    tB = prev_state['num_token_B']
    val_tA = prev_state['value_token_A']
    val_tB = prev_state['value_token_B']
    timestep = state_history[-1][-1]['timestep']
    agent_tokens = prev_state['investor_tokens']
    agent_cash = prev_state['investor_cash']
    action = 0

    order = random.randint(-1000, 500)
    if order > 0:
        if prev_state['investor_cash'] > order*val_tA:
            action = order
            agent_tokens += action
            exp_price = new_price(action, tA, tB)
            gas_rate = calc_gas(action)
            gas =  gas_rate*abs(action)*exp_price
            agent_cash -= action*exp_price - gas
        else:
            action = 0
            exp_price = 0
            gas = 0
    else:
        if prev_state['investor_tokens'] > abs(order):
            action = order
            agent_tokens += action
            exp_price = new_price(action, tA, tB)
            gas_rate = calc_gas(action)
            gas =  gas_rate*abs(action)*exp_price
            agent_cash -= action*exp_price - gas
        else:
            action = 0
            exp_price = 0   
            gas = 0

    dict_agent = {'action': action,
            'tokens': agent_tokens,
            'cash': agent_cash,
            'exp_price': exp_price,
            'gas': gas,
            }

    return dict_agent   


def investor(params, substep, state_history, prev_state):
    dict_agent = investor_calc(state_history, prev_state)

    return {'action': dict_agent}

def investor_tokens(params, substep, state_history, prev_state, policy_input):  
    key = list(policy_input.keys())[0]
    if list(policy_input[key].keys())[0] == 'investor':
        dict_agent = update_dict_agent(state_history, prev_state, 'investor', policy_input[key]['investor'])
    else:
        dict_agent = prev_dict_agent(state_history, prev_state, 'investor')

    return ('investor_tokens', dict_agent['tokens'])

def investor_cash(params, substep, state_history, prev_state, policy_input):
    key = list(policy_input.keys())[0]
    if list(policy_input[key].keys())[0] == 'investor':
        dict_agent = update_dict_agent(state_history, prev_state, 'investor', policy_input[key]['investor'])
    else:
        dict_agent = prev_dict_agent(state_history, prev_state, 'investor')

    return ('investor_cash', dict_agent['cash'])

def investor_action(params, substep, state_history, prev_state, policy_input):   
    key = list(policy_input.keys())[0]
    if list(policy_input[key].keys())[0] == 'investor':
        dict_agent = update_dict_agent(state_history, prev_state, 'investor', policy_input[key]['investor'])
    else:
        dict_agent = prev_dict_agent(state_history, prev_state, 'investor')

    return ('investor_action', dict_agent['action'])

def investor_expected_price(params, substep, state_history, prev_state, policy_input):
    key = list(policy_input.keys())[0]
    if list(policy_input[key].keys())[0] =='investor':
        dict_agent = update_dict_agent(state_history, prev_state, 'investor', policy_input[key]['investor'])
    else:
        dict_agent = prev_dict_agent(state_history, prev_state, 'investor')

    return ('investor_expected_price', dict_agent['exp_price'])

###############################################################################

# buyer

def buyer_calc(state_history, prev_state):
    tA = prev_state['num_token_A']
    tB = prev_state['num_token_B']
    val_tA = prev_state['value_token_A']
    val_tB = prev_state['value_token_B']
    timestep = state_history[-1][-1]['timestep']
    agent_tokens = prev_state['buyer_tokens']
    agent_cash = prev_state['buyer_cash']
    action = 0
    exp_price = 0

    if timestep%1 == 0:
        if prev_state['buyer_cash'] > 0:
            action = int((prev_state['buyer_cash']/2)/val_tA)
            exp_price = new_price(action, tA, tB)
            action = int((prev_state['buyer_cash']/2)/exp_price)
            agent_tokens += action
            exp_price = new_price(action, tA, tB)
            gas_rate = calc_gas(action)
            gas =  gas_rate*abs(action)*exp_price
            agent_cash -= action*exp_price - gas
        else:
            action = 0
            exp_price = 0
            gas = 0


    # Two perturbations at timestep 30 and 60
    if timestep == 30:
        action = 50000
        gas = 2000
    
    elif timestep == 60:
        action = 20000
        gas = 2000
    

    dict_agent = {'action': action,
            'tokens': agent_tokens,
            'cash': agent_cash,
            'exp_price': exp_price,
            'gas': gas,
            }

    return dict_agent   

def buyer(params, substep, state_history, prev_state):
    dict_agent = buyer_calc(state_history, prev_state)

    return {'action': dict_agent}

def buyer_tokens(params, substep, state_history, prev_state, policy_input):  
    key = list(policy_input.keys())[0]
    if list(policy_input[key].keys())[0] == 'buyer':
        dict_agent = update_dict_agent(state_history, prev_state, 'buyer', policy_input[key]['buyer'])
    else:
        dict_agent = prev_dict_agent(state_history, prev_state, 'buyer')

    return ('buyer_tokens', dict_agent['tokens'])

def buyer_cash(params, substep, state_history, prev_state, policy_input):
    key = list(policy_input.keys())[0]
    if list(policy_input[key].keys())[0] == 'buyer':
        dict_agent = update_dict_agent(state_history, prev_state, 'buyer', policy_input[key]['buyer'])
    else:
        dict_agent = prev_dict_agent(state_history, prev_state, 'buyer')

    return ('buyer_cash', dict_agent['cash'])

def buyer_action(params, substep, state_history, prev_state, policy_input):   
    key = list(policy_input.keys())[0]
    if list(policy_input[key].keys())[0] == 'buyer':
        dict_agent = update_dict_agent(state_history, prev_state, 'buyer', policy_input[key]['buyer'])
    else:
        dict_agent = prev_dict_agent(state_history, prev_state, 'buyer')

    return ('buyer_action', dict_agent['action'])

def buyer_expected_price(params, substep, state_history, prev_state, policy_input):
    key = list(policy_input.keys())[0]
    if list(policy_input[key].keys())[0] =='buyer':
        dict_agent = update_dict_agent(state_history, prev_state, 'buyer', policy_input[key]['buyer'])
    else:
        dict_agent = prev_dict_agent(state_history, prev_state, 'buyer')

    return ('buyer_expected_price', dict_agent['exp_price'])


# ###############################################################################

# Foundation

def Foundation_calc(state_history, prev_state):
    tA = prev_state['num_token_A']
    tB = prev_state['num_token_B']
    val_tA = prev_state['value_token_A']
    val_tB = prev_state['value_token_B']
    timestep = state_history[-1][-1]['timestep']
    action = 0
    exp_price = 0

    gas = 0
    exp_price = 0
    action = 0

    if timestep > 1:

        precio = prev_state['value_token_A']
        precio2 = (precio - 0.68)*100

        PATH = '/home/d10a/Projects/model_agent_logic_test'
        modelo = 1
        model = get_model(modelo)
        model.load_state_dict(torch.load(PATH))

        # Training loop - Simulations slows down significativelly

        # criterion = torch.nn.MSELoss()
        # optimizer = torch.optim.Adam(model.parameters())
        # optimizer.zero_grad()

        # input_vector = (torch.FloatTensor([precio2]))
        # label = 0
        # if precio2 > 0: label = 1

        # label = torch.FloatTensor([label])
        # prediction = model(input_vector)
        # loss = criterion(prediction, label)
        # loss.backward(retain_graph=True)
        # optimizer.step()

        # torch.save(model.state_dict(), PATH)

        # End training
        
        value = (torch.FloatTensor([precio2]))
        prediction = model(value)
        action = 0

        if prediction < 0.5:
            action = -20000
            gas = 100000
    
    dict_agent = {'action': action,
            'tokens': 0,
            'cash': 0,
            'exp_price': 0,
            'gas': gas,
            }

    return dict_agent   

def Foundation(params, substep, state_history, prev_state):
    dict_agent = Foundation_calc(state_history, prev_state)

    return {'action': dict_agent}

def Foundation_tokens(params, substep, state_history, prev_state, policy_input):  
    key = list(policy_input.keys())[0]
    if list(policy_input[key].keys())[0] == 'Foundation':
        dict_agent = update_dict_agent(state_history, prev_state, 'Foundation', policy_input[key]['Foundation'])
    else:
        dict_agent = prev_dict_agent(state_history, prev_state, 'Foundation')

    return ('Foundation_tokens', dict_agent['tokens'])

def Foundation_cash(params, substep, state_history, prev_state, policy_input):
    key = list(policy_input.keys())[0]
    if list(policy_input[key].keys())[0] == 'Foundation':
        dict_agent = update_dict_agent(state_history, prev_state, 'Foundation', policy_input[key]['Foundation'])
    else:
        dict_agent = prev_dict_agent(state_history, prev_state, 'Foundation')

    return ('Foundation_cash', dict_agent['cash'])

def Foundation_action(params, substep, state_history, prev_state, policy_input):   
    key = list(policy_input.keys())[0]
    if list(policy_input[key].keys())[0] == 'Foundation':
        dict_agent = update_dict_agent(state_history, prev_state, 'Foundation', policy_input[key]['Foundation'])
    else:
        dict_agent = prev_dict_agent(state_history, prev_state, 'Foundation')

    return ('Foundation_action', dict_agent['action'])

def Foundation_expected_price(params, substep, state_history, prev_state, policy_input):
    key = list(policy_input.keys())[0]
    if list(policy_input[key].keys())[0] =='Foundation':
        dict_agent = update_dict_agent(state_history, prev_state, 'Foundation', policy_input[key]['Foundation'])
    else:
        dict_agent = prev_dict_agent(state_history, prev_state, 'Foundation')

    return ('Foundation_expected_price', dict_agent['exp_price'])


# ###############################################################################

# Mechanisms / Variables
def num_token_A(params, substep, state_history, prev_state, policy_input):
    #print(state_history[-1])
    tA = prev_state['num_token_A']
    tB = prev_state['num_token_B']
    val_tA = prev_state['value_token_A']
    val_tB = prev_state['value_token_B']

    key = list(policy_input.keys())[0]
    key2 = list(policy_input[key].keys())[0]
    #for _ in policy_input.keys():
    tA += policy_input[key][key2]['action']
    
    return ('num_token_A', tA)


def num_token_B(params, substep, state_history, prev_state, policy_input):
    tA = prev_state['num_token_A']
    tB = prev_state['num_token_B']
    val_tA = prev_state['value_token_A']
    val_tB = prev_state['value_token_B']
    cte = prev_state['liquidity']

    key = list(policy_input.keys())[0]
    key2 = list(policy_input[key].keys())[0]
    #for _ in policy_input.keys():
    tA += policy_input[key][key2]['action']

    new_tB = cte/tA

    return ('num_token_B', new_tB)


def value_token_A(params, substep, state_history, prev_state, policy_input):
    tA = prev_state['num_token_A']
    tB = prev_state['num_token_B']
    val_tA = prev_state['value_token_A']
    val_tB = prev_state['value_token_B']
    cte = prev_state['liquidity']
    action = 0

    key = list(policy_input.keys())[0]
    key2 = list(policy_input[key].keys())[0]
    #for _ in policy_input.keys():
    tA += policy_input[key][key2]['action']

    new_value_tA = new_price(action, tA, tB)

    return ('value_token_A', new_value_tA)


# Ancore, we do not modify its value
def value_token_B(params, substep, state_history, prev_state, policy_input):
    value_token_B = prev_state['value_token_B']

    return ('value_token_B', value_token_B)


def gas(params, substep, state_history, prev_state, policy_input):
    key = list(policy_input.keys())[0]
    key2 = list(policy_input[key].keys())[0]
    
    new_gas = prev_state['gas'] + policy_input[key][key2]['gas']

    return ('gas', new_gas)
