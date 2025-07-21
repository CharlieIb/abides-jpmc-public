from gym.envs.registration import register

register(
    id='CryptoEnv-v1',
    entry_point='gym_crypto_markets.envs.crypto_env_v01:SubGymMarketsCryptoDailyInvestorEnv_v01',
)
register(
    id='CryptoEnv-v2',
    entry_point='gym_crypto_markets.envs.crypto_env_v02:SubGymMarketsCryptoDailyInvestorEnv_v02',
)