
from os import initgroups
import random
import matplotlib.pyplot as plt
import numpy as np
from random import randint
from scipy.special import comb
import csv
import pandas as pd

import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision

import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision

import matplotlib.pyplot as plt


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

        print('Model: Net version 1')
        return Net1()


def load_data(items):
    count = 0 
    data_list = []

    while count < items+1:
        value = randint(0,100) - 95
        label = 0
        if value > 0: label = 1

        data_list.append([value, label])
        count += 1
    
    return data_list

PATH = './model_agent_logic_test'

data = load_data(200)

modelo = 1
model = get_model(modelo)

Load = False
if Load == True:
    model.load_state_dict(torch.load(PATH))

print('Start training')
epochs = 2
criterion = torch.nn.MSELoss()
epoch = 0

while(epoch<epochs):
    optimizer = torch.optim.Adam(model.parameters())
    loss_list = []
    lista_print = []
    lista_print2 = []
    count = 0

    for input_vector in data:
        label = input_vector[-1]
        input_vector = input_vector[:-1]

        optimizer.zero_grad()

        input_vector = (torch.FloatTensor(input_vector))
        label = torch.FloatTensor([label])
        prediction = model(input_vector)
        loss = criterion(prediction, label)
        loss.backward(retain_graph=True)
        optimizer.step()
        loss_list.append(loss.item())
        lista_print.append(prediction.item())
        count += 1
        lista_print2.append(count)

    if epoch%1 == 0:

        correct = 0
        for input_vector in data:
            label = input_vector[-1]
            input_vector = input_vector[:-1]
            input_vector = (torch.FloatTensor(input_vector))
            label = torch.FloatTensor([label])
            prediction = model(input_vector)
            if prediction > 0.5 and label == 1: correct += 1
            elif prediction < 0.5 and label == 0: correct += 1
            else: correct = correct

            lista_print.append(prediction.item())
            count += 1
            lista_print2.append(count)

    epoch += 1

# Uncomment to save the weights of the new model
# torch.save(model.state_dict(), PATH)
plt.scatter(lista_print2, lista_print)

title = str('Training loss')
plt.title(title)
plt.ylabel('Units')
plt.xlabel('Steps') 

plt.show()