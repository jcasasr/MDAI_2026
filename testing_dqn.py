import numpy as np
import random
import torch
from gymnasium import spaces

from general import prepare, testing
from env import GlioblastomaPositionalEncoding
from training_dqnpos import DQNPositionalEncoding
from training_agents import DQNAgent
from training_buffers import ReplayBuffer

SEED = 42
random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)

if torch.cuda.is_available():
    torch.cuda.manual_seed(SEED)
    torch.cuda.manual_seed_all(SEED)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

CURRENT_CONFIG = {
    'dataset': "glio",
    'mode': "test",
    'grid_size': 8,
    'action_space': spaces.Discrete(5), 
    'rewards': [
        120.0,   # Correct Stay (Goal)
        -60.0,  # Wrong Stay (False Positive penalty)
        3.0,    # Move into tumor (The "Warm" hint)
        -1.5,   # Exit tumor (The "Cold" hint)
        1.0,    # Move within tumor (Encourage staying on target)
        -0.1    # Step penalty (Urgency)
    ],
    'max_steps': 100
}
    
start = True  # Whether to start on zero or random position
best = True
MODEL_NAME = "GLIO_grid_shaping_008"
LR = 5e-5

test_pairs = prepare(dataset=CURRENT_CONFIG['dataset'], mode="test")
device = "cpu"
env = GlioblastomaPositionalEncoding(*test_pairs[0], **CURRENT_CONFIG)
model = DQNPositionalEncoding(env, learning_rate=LR, device=device, dataset=CURRENT_CONFIG['dataset'])

if best:
    MODEL_NAME = MODEL_NAME + "_BEST_VAL"
    model.load_state_dict(
        torch.load(f"models_DQN/{MODEL_NAME}.dat", map_location=device)
    )
else:
    model.load_state_dict(
        torch.load(f"models_DQN/{MODEL_NAME}.dat", map_location=device)
    )

model.eval()  # important

agent = DQNAgent(
    env_config=CURRENT_CONFIG,
    dnnetwork=model,
    buffer_class=ReplayBuffer,
    train_pairs=test_pairs,
    validation_pairs=None,
    env_class=GlioblastomaPositionalEncoding,
    epsilon=0.0                           
)

if start:
    results2 = testing(
        agent=agent,
        test_pairs=test_pairs,
        agent_type="dqn",
        num_episodes=len(test_pairs),
        env_config=CURRENT_CONFIG,
        save_gifs=True,
        gif_folder=f"PRESENTATION/GIFS_DQN/SOZ_GIFs_Testing_{MODEL_NAME}",
        start_on_zero=True, 
        print_all=False
    )
else:
    results = testing(
        agent=agent,
        test_pairs=test_pairs,
        agent_type="dqn",
        num_episodes=len(test_pairs),
        env_config=CURRENT_CONFIG,
        save_gifs=True,
        gif_folder=f"PRESENTATION/GIFS_DQN/GIFs_Testing_{MODEL_NAME}",
        print_all=False
    )


print("Testing completed.")