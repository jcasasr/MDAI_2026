import numpy as np
import random
import torch
torch.set_num_threads(torch.get_num_interop_threads())
torch.set_num_threads(8)

from gymnasium import spaces
from general import prepare
from env import GlioblastomaPositionalEncoding
from training_dqnpos import DQNPositionalEncoding
from training_agents import DQNAgent
from training_buffers import ReplayBuffer
import wandb

SEED = 42
# Python RNG
random.seed(SEED)
# NumPy RNG
np.random.seed(SEED)
torch.manual_seed(SEED)


RUN_NAME = "POLYP_reward_shaping_008"

CURRENT_CONFIG = {
    'dataset': "polyp",
    'mode': "train",
    'grid_size': 6,
    'action_space': spaces.Discrete(5), 
    'rewards': [
        25.0,   # Correct Stay (Goal)
        -12.0,  # Wrong Stay (False Positive penalty)
        0.5,    # Move into tumor (The "Warm" hint)
        -0.5,   # Exit tumor (The "Cold" hint)
        0.07,    # Move within tumor (Encourage staying on target)
        -0.15    # Step penalty (Urgency)
    ],
    'max_steps': 75
}
    
ENVIRONMENT = GlioblastomaPositionalEncoding
NET = DQNPositionalEncoding
AGENT = DQNAgent
BUFFER = ReplayBuffer

LR = 5e-5
# LR = 0.0001 #From paper
MEMORY_SIZE = 15000 #From paper
MAX_EPISODES = 5000 #From paper

EPSILON = 1.0 #From paper
EPSILON_MIN = 0.1 #From paper
DECAY_TYPE = "exponential"
# DECAY_TYPE = "subtractive"
if DECAY_TYPE == "exponential":
    EPSILON_DECAY = 0.9995 #Let's try exponential decay
    print(f"Starting at {EPSILON}, decaying {EPSILON_DECAY} each episode, will reach {EPSILON_MIN} after {int(np.log(EPSILON_MIN/EPSILON)/np.log(EPSILON_DECAY))} episodes")
else:
    EPSILON_DECAY = (EPSILON - EPSILON_MIN) / MAX_EPISODES
    print(f"Starting at {EPSILON}, decaying {EPSILON_DECAY}, will reach {EPSILON_MIN} after {MAX_EPISODES} episodes")

GAMMA = 0.99 #0.99
BATCH_SIZE = 128 #From paper
BURN_IN = 500 # 500
DNN_UPD = 4
DNN_SYNC = 200
VAL_FREQ = 150

# train_pairs = prepare()
train_pairs = prepare(dataset=CURRENT_CONFIG['dataset'], mode="train")
validation_pairs = prepare(dataset=CURRENT_CONFIG['dataset'], mode="val")

env=ENVIRONMENT(*train_pairs[0], **CURRENT_CONFIG)
print(env.observation_space.shape)
print(env.action_space.n)
print(np.arange(env.action_space.n))
if env.observation_space.shape[1] * CURRENT_CONFIG['grid_size'] == 240:
    print(f"Using correct patch size {env.observation_space.shape[1]} given grid size {CURRENT_CONFIG['grid_size']}")

device = 'mps' if torch.backends.mps.is_available() else 'cpu'
net = NET(env, learning_rate=LR, device=device, dataset=CURRENT_CONFIG['dataset'])

buffer = BUFFER(capacity=MEMORY_SIZE)
agent = AGENT(env_config=CURRENT_CONFIG, dnnetwork=net, buffer_class=BUFFER, train_pairs=train_pairs, validation_pairs=validation_pairs, env_class=ENVIRONMENT,
                 epsilon=EPSILON, eps_decay=EPSILON_DECAY, eps_decay_type=DECAY_TYPE, epsilon_min=EPSILON_MIN,
                 batch_size=BATCH_SIZE, gamma=GAMMA, 
                 memory_size=MEMORY_SIZE, buffer_initial=BURN_IN,
                 save_name=RUN_NAME)

print(f"Using Glioblastoma class {ENVIRONMENT}, DQN class {NET}, Agent class {AGENT}, Buffer class {BUFFER}")

wandb.init(
    project="new",
    name=RUN_NAME,
    id=RUN_NAME,
    config={
        "environment": ENVIRONMENT,
        "configuration": CURRENT_CONFIG,
        "model": NET,
        "agent": AGENT,
        "buffer": BUFFER,
        "lr": LR,
        "MEMORY_SIZE": MEMORY_SIZE,
        "MAX_EPISODES": MAX_EPISODES,
        "EPSILON": EPSILON,
        "EPSILON_DECAY": EPSILON_DECAY,
        "Decay type": DECAY_TYPE,
        "EPSILON_MIN": EPSILON_MIN,
        "GAMMA": GAMMA,
        "BATCH_SIZE": BATCH_SIZE,
        "BURN_IN": BURN_IN,
        "DNN_UPD": DNN_UPD,
        "DNN_SYNC": DNN_SYNC, 
        "VAL_FREQ": VAL_FREQ
    },
    save_code=True)

wandb.save("general.py")
wandb.save("env.py")
wandb.save('training_dqn.py')
wandb.save('training_agents.py')
wandb.save('training_buffers.py')

agent.train(
    train_pairs=train_pairs,
    validation_pairs=validation_pairs,
    gamma=GAMMA,
    max_episodes=MAX_EPISODES,
    dnn_update_frequency=DNN_UPD,
    dnn_sync_frequency=DNN_SYNC, 
    val_frequency=VAL_FREQ
)
wandb.finish()