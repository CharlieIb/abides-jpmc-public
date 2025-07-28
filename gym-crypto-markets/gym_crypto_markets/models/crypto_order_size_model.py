import json

import numpy as np
from pomegranate import GeneralMixtureModel
# --- Order size model definition ---
# Based on the analysis of BTCUSDT data from Binance:
# - The vast majority of trades are very small in actual BTC terms
# - When scaled by 100000 (notional amounts), the quantities will range from 1 (for 0.00001 BTC) up to ~1034891 (for 10.34891 BTC).
# - The mean notional quantity is ~ 491.78
# - The standard deviation notional quantity is ~4043.65
# - The distribution is highly skewed towards the lower end of the notional values

# - We will primarily use a LogNormalDistribution for the bulk of the trades
# as it naturally handles skewed data and non-negative values
# I include a few Normal distributions for larger, less frequent 'block' trades

NOISE_LOG_NORMAL_MEAN_SCALED = 2.8871
NOISE_LOG_NORMAL_STD_SCALED = 1.9939

VM_LOG_NORMAL_MEAN_SCALED = 4.8073
VM_LOG_NORMAL_STD_SCALED = 2.050




noise_order_size = {
    "class": "GeneralMixtureModel",
    "distributions": [
        {
            "class": "Distribution",
            "name": "LogNormalDistribution",
            "parameters": [NOISE_LOG_NORMAL_MEAN_SCALED, NOISE_LOG_NORMAL_STD_SCALED],
            "frozen": False,
        },
        {
            "class": "Distribution",
            "name": "NormalDistribution",
            "parameters": [100000.0, 0.15],
            "frozen": True,
        },
        {
            "class": "Distribution",
            "name": "NormalDistribution",
            "parameters": [500000.0, 0.15],
            "frozen": True,
        },
        {
            "class": "Distribution",
            "name": "NormalDistribution",
            "parameters": [1000000.0, 0.15],
            "frozen": True,
        },
    ],
    "weights": [
        0.99992946, # High weight for the dominant LogNormal component
        0.00007,    # Very small weight for 100000 notional (corresponds to 0.00002% for 1-5 BTC original)
        0.0000005,  # Even smaller weight for 500000 notional (corresponds to 0.0000005% for 5-10 BTC original)
        0.00000004, # Tiny weight for 1000000 notional (corresponds to 0.00000004% for 10-50 BTC original)
        ],
}
value_order_size = {
    "class": "GeneralMixtureModel",
    "distributions": [
        {
            "class": "Distribution",
            "name": "LogNormalDistribution",
            "parameters": [VM_LOG_NORMAL_MEAN_SCALED, VM_LOG_NORMAL_STD_SCALED],
            "frozen": False,
        },
        {
            "class": "Distribution",
            "name": "NormalDistribution",
            "parameters": [100.0, 5.00],
            "frozen": True,
        },
        {
            "class": "Distribution",
            "name": "NormalDistribution",
            "parameters": [250.0, 5.00],
            "frozen": True,
        },
{
            "class": "Distribution",
            "name": "NormalDistribution",
            "parameters": [500.0, 5.00],
            "frozen": True,
        },
        {
            "class": "Distribution",
            "name": "NormalDistribution",
            "parameters": [1000.0, 10.00],
            "frozen": True,
        },
    ],
    "weights": [
        0.20,
        0.20,
        0.20,
        0.20,
        0.20,
        ],
}
momentum_order_size = {
    "class": "GeneralMixtureModel",
    "distributions": [
        {
            "class": "Distribution",
            "name": "LogNormalDistribution",
            "parameters": [VM_LOG_NORMAL_MEAN_SCALED, VM_LOG_NORMAL_STD_SCALED],
            "frozen": False,
        },
        {
            "class": "Distribution",
            "name": "NormalDistribution",
            "parameters": [50.0, 5.00],
            "frozen": True,
        },
        {
            "class": "Distribution",
            "name": "NormalDistribution",
            "parameters": [100.0, 5.00],
            "frozen": True,
        },
        {
            "class": "Distribution",
            "name": "NormalDistribution",
            "parameters": [250.0, 5.0],
            "frozen": True,
        },
        {
            "class": "Distribution",
            "name": "NormalDistribution",
            "parameters": [500.0, 10.0],
            "frozen": True,
        },

    ],
    "weights": [
        0.20,
        0.20,
        0.20,
        0.20,
        0.20,
        ],
}


class OrderSizeModel:
    def __init__(self, agent_type: str) -> None:
        if agent_type == "noise":
            self.model_config = noise_order_size
        elif agent_type == "value":
            self.model_config = value_order_size
        elif agent_type == "momentum":
            self.model_config = momentum_order_size
        else:
            raise ValueError(f"Unknown agent type: {agent_type}")

        self.model = GeneralMixtureModel.from_json(json.dumps(self.model_config))

    def sample(self, random_state: np.random.RandomState) -> int:
        sampled_value = self.model.sample(random_state=random_state)
        return int(round(max(0.0, sampled_value)))
