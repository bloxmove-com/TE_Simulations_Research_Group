
num_token_A = 20000000
num_token_B = 20000000

initial_state = {
    'agent': 0,
    'mempool': [],

    'num_token_A': num_token_A,
    'num_token_B': num_token_B,
    'value_token_A': 1,
    'value_token_B': 1,

    'liquidity': num_token_A*num_token_B,   # Total liquidity in pool

    'investor_tokens': 5000,
    'investor_cash': 10000,
    'investor_action': 0,
    'investor_expected_price': 0,

    'trader_tokens': 0,
    'trader_cash': 5000,
    'trader_action': 0,
    'trader_expected_price': 0,

    'buyer_tokens': 0,
    'buyer_cash': 10000,
    'buyer_action': 0,
    'buyer_expected_price': 0,

    'Foundation_tokens': 0,
    'Foundation_cash': 0,
    'Foundation_action': 0,
    'Foundation_expected_price': 0,

    'gas': 0,

}

