from gym.envs.registration import register

register(
    id='CryptoEnv-v1',
    # NOTE: The entry point path starts from the project root.
    entry_point='gym_crypto_markets.envs.crypto_env_v01:SubGymMarketsCryptoDailyInvestorEnv_v01',
)