"""
Model parameters.
"""

from .state_variables import initial_state

cte = initial_state['num_token_A']*initial_state['num_token_B']

sys_params = {
    'cte': [cte]        # OJO! si hay cambios de L no es constante e igual tiene
                        # mas sentido definirlo en state_variable
}


