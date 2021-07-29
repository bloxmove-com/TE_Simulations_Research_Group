from numpy.core.fromnumeric import var
import pandas as pd
import matplotlib.pyplot as plt

def montecarlo_plot(df, variables, runs):
    
    timesteps = df.timestep
    max_time = max(df.timestep)
    timesteps = df.timestep[:max_time+1]
    data = (pd.DataFrame({}))

    # if len(variables) == 1:
    #     for i in range(runs):
    #         data = df[variables[0]]
            
    #         fig, axes = plt.subplots(figsize=(12, 8))
    #         data.plot(ax=axes)

    #         axes.set_title(str(variables[0]))
    #         axes.set_ylabel('Units')
    #         axes.set_xlabel('Timestps')

    # elif len(variables) > 1:
    
    for i in range(runs):
        for ii in range(len(variables)):
            lista_aux = []
            for iii in range(len(df[variables[ii]])):              
                if df['run'][iii] == i+1:
                    lista_aux.append(df[variables[ii]][iii])
            
            name_aux = str(variables[ii]) + '_Run_' + str(i+1)
            data[name_aux] = lista_aux

    fig, axes = plt.subplots(figsize=(12, 8))
    data.plot(ax=axes)


    title = str(variables[0])
    
    if len(variables) > 1:
        for _ in variables[1:]:
            title += '& ' + str(_)

    axes.set_title(title)
    axes.set_ylabel('Units')
    axes.set_xlabel('Timestps')

