from .parts.agents import *
import random


state_update_blocks = [
    {
        # agents.py // Behaviors or Policies
        'policies': {
            'selector': selector
            
        },

        # agents.py // Mechanisms or Variables
        'variables': {
            'agent': agent,
            'mempool': mempool,
            
            'num_token_A': num_token_A,
            'num_token_B': num_token_B,
            'value_token_A': value_token_A,
            'value_token_B': value_token_B,

            #'liquidity': liquidity,                             # Total liquidity in pool


            'investor_tokens': investor_tokens,
            'investor_cash': investor_cash,
            'investor_action': investor_action,
            'investor_expected_price': investor_expected_price,

            'trader_tokens': trader_tokens,
            'trader_cash': trader_cash,
            'trader_action': trader_action,
            'trader_expected_price': trader_expected_price,

            'buyer_tokens': buyer_tokens,
            'buyer_cash': buyer_cash,
            'buyer_action': buyer_action,
            'buyer_expected_price': buyer_expected_price,

            'Foundation_tokens': Foundation_tokens,
            'Foundation_cash': Foundation_cash,
            'Foundation_action': Foundation_action,
            'Foundation_expected_price': Foundation_expected_price,

            'gas': gas,

        }
    }
]