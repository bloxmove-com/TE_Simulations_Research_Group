import numpy as np
import random
import pandas as pd
import matplotlib.pyplot as plt

# # Define a dictionary containing Students data
# data = {'Name': ['Jai', 'Princi', 'Gaurav', 'Anuj'],
#         'Height': [5.1, 6.2, 5.1, 5.2],
#         'Qualification': ['Msc', 'MA', 'Msc', 'Msc']}
  
# # Define a dictionary with key values of
# # an existing column and their respective
# # value pairs as the # values for our new column.
# address = {'Delhi': 'Jai', 'Bangalore': 'Princi',
#            'Patna': 'Gaurav', 'Chennai': 'Anuj'}
  
# # Convert the dictionary into DataFrame
# df = pd.DataFrame(data)
  
# # Provide 'Address' as the column name
# print(df.keys())

# for _ in df.keys():
#     print(_)


# Slippage:

# initial_ta = 1000
# initial_tb = 1000
# initial_p_ta = 1
# initial_p_tb = 1
# change = -400

# ta = initial_ta
# tb= initial_tb
# p_ta = initial_p_ta
# p_tb = initial_p_tb
# cte = ta*tb

# add_ta = change
# new_ta = ta + add_ta
# new_tb = cte/new_ta
# new_pta = (p_tb*new_tb)/new_ta

# print('***************')
# print('1. 200 a precio inicial')
# print(new_ta)
# print(new_pta)
# print(new_pta*new_ta)
# print(p_tb*new_tb)
# print('Te llevas X tokens:')
# print(ta-new_ta)
# print(new_tb-tb)


# print('***************')
# print('2. 200 a precio actualizado')

# ta = initial_ta
# tb= initial_tb
# p_ta = initial_p_ta
# p_tb = initial_p_tb

# cte = ta*tb
# new_ta = ta
# lista =  []
# token = 0
# while token < abs(change):

#         new_ta -= 1
#         new_tb = cte/new_ta
#         new_pta = (p_tb*new_tb)/new_ta
#         lista.append(new_pta)
#         token += 1

# print(new_ta)
# print(new_pta)
# print(new_pta*new_ta)
# print(p_tb*new_tb)
# print('Te llevas X tokens:')
# print(ta-new_ta)
# print(new_tb-tb)

# plt.plot(lista)
# plt.show()


# class agent():
#         def __init__(self):
#                 self.count = 0
#                 self.price = 0

# dimi = agent()
# def test(dimi):
#         dimi.count +=  2
        
#         return 

# test(dimi)
# print(dimi.count)


df = pd.DataFrame({
    'col1': ['A', 'A', 'B', np.nan, 'D', 'C'],
    'col2': [2, 1, 9, 8, 7, 4],
    'col3': [0, 1, 9, 4, 2, 3],
    'col4': ['a', 'B', 'c', 'D', 'e', 'F']
})

print(df.iloc[[2]])