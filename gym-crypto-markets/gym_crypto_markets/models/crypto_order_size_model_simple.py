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



NOISE_LOG_NORMAL_MEAN_SCALED = 3.1
NOISE_LOG_NORMAL_STD_SCALED = 2.1
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
        "parameters": [NOISE_LOG_NORMAL_MEAN_SCALED, NOISE_LOG_NORMAL_STD_SCALED],
        "frozen": False
      },
      { "class": "Distribution", "name": "NormalDistribution", "parameters": [50.0, 2.5], "frozen": True},
      { "class": "Distribution", "name": "NormalDistribution", "parameters": [200.0, 10.0], "frozen": True },
      { "class": "Distribution", "name": "NormalDistribution", "parameters": [250.0, 12.5], "frozen": True },
      { "class": "Distribution", "name": "NormalDistribution", "parameters": [400.0, 20.0], "frozen": True },
      { "class": "Distribution", "name": "NormalDistribution", "parameters": [100.0, 5.0], "frozen": True },
      { "class": "Distribution", "name": "NormalDistribution", "parameters": [1000.0, 50.0], "frozen": True },
      { "class": "Distribution", "name": "NormalDistribution", "parameters": [300.0, 15.0], "frozen": True },
      { "class": "Distribution", "name": "NormalDistribution", "parameters": [700.0, 35.0], "frozen": True },
      { "class": "Distribution", "name": "NormalDistribution", "parameters": [2000.0, 100.0], "frozen": True },
      { "class": "Distribution", "name": "NormalDistribution", "parameters": [5000.0, 250.0], "frozen": True },
      { "class": "Distribution", "name": "NormalDistribution", "parameters": [800.0, 40.0], "frozen": True },
      { "class": "Distribution", "name": "NormalDistribution", "parameters": [500.0, 25.0], "frozen": True },
      { "class": "Distribution", "name": "NormalDistribution", "parameters": [600.0, 30.0], "frozen": True },
      { "class": "Distribution", "name": "NormalDistribution", "parameters": [900.0, 45.0], "frozen": True },
      { "class": "Distribution", "name": "NormalDistribution", "parameters": [10000.0, 500.0], "frozen": True }
    ],
    "weights": [
      0.2000,
      0.4048,
      0.0651,
      0.0559,
      0.0449,
      0.0376,
      0.0364,
      0.0357,
      0.0305,
      0.0248,
      0.0229,
      0.0191,
      0.0185,
      0.0176,
      0.0161,
      0.0078
    ]
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
