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

# - We will primarily use a LogNormalDistribution for the noise of the model
# as it naturally handles skewed data and non-negative values
# I include a few Normal distributions for larger, less frequent 'block' trades
# All agents will most of the time block trade 100 and will occationally trade higher amounts



NOISE_LOG_NORMAL_MEAN_SCALED = 2.8871
NOISE_LOG_NORMAL_STD_SCALED = 1.9939
#
# VM_LOG_NORMAL_MEAN_SCALED = 4.8073
# VM_LOG_NORMAL_STD_SCALED = 2.050
#



order_size = {
    "class": "GeneralMixtureModel",
    "distributions": [
        {
            "class": "Distribution",
            "name": "LogNormalDistribution",
            "parameters": [NOISE_LOG_NORMAL_MEAN_SCALED, NOISE_LOG_NORMAL_MEAN_SCALED],
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
        0.10,
        0.60,
        0.10,
        0.10,
        0.10,
        ],
}


class OrderSizeModelSimple:
    def __init__(self, agent_type: str) -> None:
        if agent_type == "noise":
            self.model_config = order_size
        elif agent_type == "value":
            self.model_config = order_size
        elif agent_type == "momentum":
            self.model_config = order_size
        else:
            raise ValueError(f"Unknown agent type: {agent_type}")

        self.model = GeneralMixtureModel.from_json(json.dumps(self.model_config))

    def sample(self, random_state: np.random.RandomState) -> int:
        sampled_value = self.model.sample(random_state=random_state)
        return int(round(max(0.0, sampled_value)))
