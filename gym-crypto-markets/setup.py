# In gym_crypto_markets/setup.py
from setuptools import setup, find_packages

setup(
    name='gym-crypto-markets',  # The name pip will use
    version='0.1.0',
    packages=find_packages(),  # Automatically find all packages in this directory
    install_requires=[
        'gym==0.18.0',
        'numpy',
        'abides-core',
        'abides-gym',
        'abides-markets',
        'abides-multi-exchange',
    ]
)