# In gym_crypto_markets/setup.py
from setuptools import setup, find_packages

setup(
    name='gym-crypto-markets',  # The name pip will use
    version='0.1.0',
    packages=find_packages(),  # Automatically find all packages in this directory
    install_requires=[
        'gym==0.18.0',
        'numpy',
        # Add any other dependencies your environment needs, e.g., 'pandas'
    ],
    entry_points={
        'gym.envs': [
            'crypto_env-v01 = gym_crypto_markets.envs.crypto_env_v01:SubGymMarketsCryptoDailyInvestorEnv_v01',
        ]
    }
)