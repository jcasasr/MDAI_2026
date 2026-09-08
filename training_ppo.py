from stable_baselines3.common.vec_env import DummyVecEnv
from stable_baselines3 import PPO
import torch
import wandb
import numpy as np
import gymnasium as gym
from gymnasium import spaces

from wandb.integration.sb3 import WandbCallback
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.callbacks import EvalCallback
from stable_baselines3.common.callbacks import CallbackList
from stable_baselines3.common.torch_layers import CombinedExtractor

from general import prepare
from env import GlioblastomaPositionalEncoding

class DatasetWrapper(gym.Wrapper):
    def __init__(self, image_paths, mask_paths, **env_kwargs):
        # image_paths and mask_paths are lists of length N
        self.image_paths = image_paths
        self.mask_paths = mask_paths
        self.n = len(image_paths)
        self.env_kwargs = env_kwargs  # arguments to pass to inner env

        # TEMP env to inherit observation/action space
        tmp_env = GlioblastomaPositionalEncoding(image_paths[0], mask_paths[0], **env_kwargs)
        super().__init__(tmp_env)
        
        if self.env_kwargs.get('dataset') == 'polyp':
            # old_shape = self.observation_space.shape
            # new_shape = old_shape + (3,)
            # self.observation_space = spaces.Box(
            #     low=0,
            #     high=255,
            #     shape=new_shape,
            #     dtype=np.uint8
            # )
            self.observation_space = spaces.Box(
                low=0,
                high=255,
                shape=(15, tmp_env.block_size, tmp_env.block_size),
                dtype=np.uint8
            )

    def reset(self, **kwargs):
        idx = np.random.randint(0, self.n)
        if hasattr(self, 'env'):
            self.env.close()
        self.env = GlioblastomaPositionalEncoding(
            self.image_paths[idx],
            self.mask_paths[idx],
            **self.env_kwargs
        )
        if self.env_kwargs.get('dataset') == 'polyp':
            start_on_zero = np.random.random() < 0.5
            obs, info = self.env.reset(**kwargs, start_on_zero=start_on_zero)
            obs = (obs * 255).astype(np.uint8)
            
            if obs.ndim == 4 and 3 in obs.shape:
                channel_axis = list(obs.shape).index(3)
                obs = np.moveaxis(obs, channel_axis, 1) # Move RGB to second position
                # This makes it (5, 3, 40, 40). Now reshape to (15, 40, 40)
                obs = obs.reshape(15, self.env.block_size, self.env.block_size)
            
            # CASE 2: 3D input (Slices, H, W) -> (5, 40, 40)
            elif obs.ndim == 3 and obs.shape[0] == 5:
                # We need 15 channels for the observation_space, but we only have 5.
                # We must repeat the channels to fill the 15-channel requirement.
                obs = np.repeat(obs, 3, axis=0) # (5, 40, 40) -> (15, 40, 40)

            return obs, info
        else:
            if np.random.random() < 0.5:
                return self.env.reset(**kwargs, start_on_zero=True)
            else:
                return self.env.reset(**kwargs, start_on_zero=False)

    def step(self, action):
        if self.env_kwargs.get('dataset') == 'polyp':
            obs, reward, terminated, truncated, info = self.env.step(action)  
            obs = (obs * 255).astype(np.uint8)

            if obs.ndim == 4 and 3 in obs.shape:
                channel_axis = list(obs.shape).index(3)
                obs = np.moveaxis(obs, channel_axis, 1) # Move RGB to second position
                # This makes it (5, 3, 40, 40). Now reshape to (15, 40, 40)
                obs = obs.reshape(15, self.env.block_size, self.env.block_size)
            
            # CASE 2: 3D input (Slices, H, W) -> (5, 40, 40)
            elif obs.ndim == 3 and obs.shape[0] == 5:
                # We need 15 channels for the observation_space, but we only have 5.
                # We must repeat the channels to fill the 15-channel requirement.
                obs = np.repeat(obs, 3, axis=0) # (5, 40, 40) -> (15, 40, 40)

            return obs, reward, terminated, truncated, info   
        else:     
            return self.env.step(action)

def make_env(grid_size, rewards, action_space, max_steps, dataset_name):
    def _init():
        env = DatasetWrapper(
            image_paths=image_paths,
            mask_paths=mask_paths,
            grid_size=grid_size,
            tumor_threshold=0.01,
            rewards=rewards,
            action_space=action_space,
            max_steps=max_steps,
            dataset=dataset_name
        )
        return Monitor(env)
    return _init


CURRENT_CONFIG = {
    'dataset': "polyp",
    'mode': "train",               # IMPORTANT
    'grid_size': 8,
    'action_space': spaces.Discrete(5),
    'rewards': [
        100.0,   # Correct Stay
        -50.0,  # Wrong Stay
        2.5,    # Move into tumor
        -2.5,   # Exit tumor
        0.5,    # Move within tumor
        -0.2    # Step penalty
    ],
    'max_steps': 100
}
MODEL_NAME = "POLYP_ppo_011"

CON2 = {
    'policy': 'MlpPolicy',
    'verbose': 2,
    'n_steps': 2048,
    'batch_size': 128,
    'ent_coef': 0.1,
    'learning_rate': 3e-4,
    'gae_lambda': 0.95,
    'clip_range': 0.2,
    'total_timesteps': 150_000,
    'gradient_save_freq': 100,
    'model_save_freq': 10000,
}

train_pairs = prepare(dataset=CURRENT_CONFIG['dataset'], mode="train")
val_pairs = prepare(dataset=CURRENT_CONFIG['dataset'], mode="val")

image_paths = [p[0] for p in train_pairs]
mask_paths  = [p[1] for p in train_pairs]

val_image_paths = [p[0] for p in val_pairs]
val_mask_paths  = [p[1] for p in val_pairs]

env_fns = [make_env(
    grid_size=CURRENT_CONFIG['grid_size'],
    rewards=CURRENT_CONFIG['rewards'],
    action_space=CURRENT_CONFIG['action_space'],
    max_steps=CURRENT_CONFIG['max_steps'], 
    dataset_name=CURRENT_CONFIG['dataset']
) for _ in range(8)]

env = DummyVecEnv(env_fns)
val_env = Monitor(DatasetWrapper(
    image_paths=val_image_paths,
    mask_paths=val_mask_paths,
    grid_size=CURRENT_CONFIG['grid_size'],
    rewards=CURRENT_CONFIG['rewards'],
    action_space=CURRENT_CONFIG['action_space'],
    max_steps=CURRENT_CONFIG['max_steps'], 
    dataset=CURRENT_CONFIG['dataset']
))

# The Callback
eval_callback = EvalCallback(
    val_env, 
    best_model_save_path=f"./models_PPO/{MODEL_NAME}/best",
    log_path=f"./logs/{MODEL_NAME}/eval", 
    eval_freq=5000, # How often to validate (in timesteps)
    deterministic=True, 
    render=False
)

wandb.init(
    project="new",
    name=MODEL_NAME,
    id=MODEL_NAME,
    config={
        "configuration": CURRENT_CONFIG,
        "model": 'ppo',
        "policy": CON2['policy'],
        "verbose": CON2['verbose'],
        "n_steps": CON2['n_steps'],
        "batch_size": CON2['batch_size'],
        "ent_coef": CON2['ent_coef'],
        "learning_rate": CON2['learning_rate'],
        "gae_lambda": CON2['gae_lambda'],
        "clip_range": CON2['clip_range'],
        "total_timesteps": CON2['total_timesteps'],
        "gradient_save_freq": CON2['gradient_save_freq'],
        "model_save_freq": CON2['model_save_freq'],
    },
    save_code=True)

# model = PPO(
#     CON2['policy'],
#     env,
#     verbose=CON2['verbose'],
#     n_steps=CON2['n_steps'],
#     batch_size=CON2['batch_size'],
#     ent_coef=CON2['ent_coef'],
#     learning_rate=CON2['learning_rate'],
#     gae_lambda=CON2['gae_lambda'],
#     clip_range=CON2['clip_range'],
#     tensorboard_log=f"runs/{MODEL_NAME}"
# )

policy_kwargs = dict(
    activation_fn=torch.nn.ELU, 
    net_arch=dict(pi=[512], vf=[512]), # Use 'vf' (Value Function) instead of 'qf' for PPO
)

model = PPO(
    "CnnPolicy",  # Changed from CnnPolicy to handle the (5, 40, 40, 3) shape
    env,
    verbose=CON2['verbose'],
    n_steps=CON2['n_steps'],
    batch_size=CON2['batch_size'],
    ent_coef=CON2['ent_coef'],
    learning_rate=CON2['learning_rate'],
    gae_lambda=CON2['gae_lambda'],
    clip_range=CON2['clip_range'],
    policy_kwargs=policy_kwargs,
    tensorboard_log=f"runs/{MODEL_NAME}"
)

callbacks = CallbackList([
    WandbCallback(model_save_path=f"models_PPO/{MODEL_NAME}", verbose=2),
    eval_callback
])

model.learn(
    total_timesteps=CON2['total_timesteps'],
    callback=callbacks,
)
