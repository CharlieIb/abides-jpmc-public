ABIDES-GYM Reinforcement Learning Experiments

Introduction

This repository contains a set of experiments for training and evaluating Reinforcement Learning (RL) agents in simulated financial markets. The simulation environment is built upon the ABIDES (Agent-Based Interactive Discrete Event Simulation) framework and extended with OpenAI Gym compatibility for RL tasks, with a specific focus on cryptocurrency markets.

The primary script for running simulations is run_gym_simulation.py, which allows for flexible configuration through YAML files and command-line arguments.

Setup and Installation

Follow these steps to set up the environment and run the simulations.

Prerequisites

    Conda: This project uses Conda for managing dependencies and creating an isolated environment. An installation of Anaconda or Miniconda is required.

Installation Procedure

    Clone the Repository:
    First, clone the repository to a local directory.
    Bash

git clone <your-repository-url>
cd ABIDES_GYM_EXT

Create and Activate the Conda Environment:
Use the provided environment.yml file to create a Conda environment with all the necessary packages.
Bash

conda env create -f environment.yml
conda activate abides

This command creates an environment named abides. This environment must be activated in the terminal session before executing any scripts.

Install Local Packages:
The project is structured into several local Python packages (abides-core, abides-gym, etc.). The provided installation script installs these in editable mode.
Bash

    bash install.sh

The environment is now fully configured.

Running a Simulation

All simulations are launched using the run_gym_simulation.py script located in the my-experiments/my_experiments/exec/ directory. It is necessary to first navigate to this directory.
Bash

cd my-experiments/my_experiments/exec/

The script is executed with the following structure:
Bash

python run_gym_simulation.py <path_to_config.yaml> --mode <simulation_mode> [options]

Script Arguments

Core Arguments

    config_path (Required): The first argument must be the path to the YAML configuration file (e.g., base_config.yaml). This file defines the parameters for the market simulation, the gym environment, and the agents.

    --mode (Required): Specifies the simulation's purpose. The available modes include:

        train-historical: Train an agent using historical market data.

        test-historical: Test a pre-trained agent on historical data.

        train-abides: Train an agent within a fully generated ABIDES market simulation.

        *-se: A suffix for "single-exchange" mode (e.g., train-historical-se), which configures the simulation to run with only one exchange.

Optional Arguments

    --agent <AgentName>: Specifies which agent configuration to use from the YAML file (e.g., DQNAgent, SETripleBarrier). This overrides the default agent specified in the configuration file.

    --load_weights_path <path/to/weights.pth>: Specifies the path to pre-trained model weights. This argument is required when using any test-* mode.

    --date <YYYY-MM-DD>: Executes the simulation using data from a specific date, provided that the corresponding data file exists.

Configuration File (base_config.yaml)

The base_config.yaml file serves as the central location for simulation configuration. It is organized into the following primary sections:

    background_config_params: Defines the market structure, including the number and type of exchanges, agent populations (e.g., noise traders, arbitrageurs), and paths to historical data files.

    gym_environment: Contains settings for the OpenAI Gym wrapper, such as reward function definitions and observation space features.

    agent_configurations: A dictionary where multiple agent models can be defined. Each agent entry specifies its class, learning parameters (e.g., learning rate, discount factor), and neural network architecture.

    simulation_runner: Controls the execution loop of the simulation, including the number of episodes, steps per episode, and settings for logging and saving model weights.

Modifying this file allows for experimentation with different market conditions and agent parameters without altering the Python source code.