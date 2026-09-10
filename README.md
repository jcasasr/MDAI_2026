# Cross-Domain Single Lesion Localization in Medical Images via Deep Reinforcement Learning

- Martina Carretta-Brugueras, Jorge Bernal, Jordi Casas Roma. "Cross-Domain Single Lesion Localization in Medical Images via Deep Reinforcement Learning”. In: Torra, V., Narukawa, Y., Reig, R. (eds) Modeling Decisions for Artificial Intelligence. MDAI 2026. Lecture Notes in Artificial Intelligence, vol 16924. Springer, Cham. [DOI](https://doi.org/xxx)

## Data
In this project we used MRI scans from BraTS 2020 and colonoscopies from PolypSegm_ASH. See statistics in: 
- [Glioblastoma](glio.ipynb)
- [polyps](polyp.ipynb)

## Environment
[env.py](env.py) has the class to convert the input to the gridworld environment

## Usage
Use [general](general.py) to get the pairs of images and masks depending on the dataset and the modality (train, validate, test) perform testing. 

DQN:
- [Training main file](training_dqn.py)
- [DQN architecture](training_dqnpos.py)
- [Agent](training_agents.py)
- [Buffer](training_buffers.py)
- [Training](training_dqn.py)
- [Testing](testing_dqn.py)

PPO:
- [Training](training_ppo.py)
- [testing](testing_ppo.py)

REINFORCE:
- [Training](training_reinforce.py)
- [testing](testing_reinforce.py)
