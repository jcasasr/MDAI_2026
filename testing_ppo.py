import numpy as np
import random
import torch
from gymnasium import spaces
import gymnasium as gym

from stable_baselines3 import PPO

from general import prepare, testing
from env import GlioblastomaPositionalEncoding
from stable_baselines3.common.monitor import Monitor


SEED = 42
random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)

if torch.cuda.is_available():
    torch.cuda.manual_seed(SEED)
    torch.cuda.manual_seed_all(SEED)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

class DatasetWrapper(gym.Wrapper):
    def __init__(self, image_paths, mask_paths, **env_kwargs):
        # image_paths and mask_paths are lists of length N
        if isinstance(image_paths, str):
            self.image_paths = [image_paths]
            self.mask_paths = [mask_paths]
        else:
            self.image_paths = image_paths
            self.mask_paths = mask_paths
            
        self.n = len(self.image_paths)
        self.env_kwargs = env_kwargs  # arguments to pass to inner env

        # TEMP env to inherit observation/action space
        tmp_env = GlioblastomaPositionalEncoding(self.image_paths[0], self.mask_paths[0], **env_kwargs)
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
    def __getattr__(self, name):
        """Forward all unknown attributes to the inner environment."""
        return getattr(self.env, name)
    def render(self, show=True):
        # Explicitly forward the 'show' argument to the internal environment
        return self.env.render(show=show)
    def reset(self, **kwargs):
        idx = np.random.randint(0, self.n)
        if hasattr(self, 'env'):
            self.env.close()
        self.env = GlioblastomaPositionalEncoding(
            self.image_paths[idx],
            self.mask_paths[idx],
            **self.env_kwargs
        )
        if 'start_on_zero' in kwargs:
            start_on_zero = kwargs.pop('start_on_zero')
        else:
            start_on_zero = np.random.random() < 0.5
            
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
    'mode': "test",               # IMPORTANT
    'grid_size': 4,
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
MODEL_NAME = "POLYP_ppo_010"
start = True
best = True

test_pairs = prepare(dataset=CURRENT_CONFIG['dataset'], mode="test")

env = DatasetWrapper(
    image_paths=[p[0] for p in test_pairs],
    mask_paths=[p[1] for p in test_pairs],
    **CURRENT_CONFIG
)
device = "cpu"


if best:
    model_path = f"/Users/martina/code/4year/new/models_PPO/{MODEL_NAME}/best/best_model.zip"
    MODEL_NAME = MODEL_NAME + "_BEST"
else:
    model_path = f"/Users/martina/code/4year/new/models_PPO/{MODEL_NAME}/model.zip"

loaded_model = PPO.load(model_path)

loaded_model.env_class = DatasetWrapper

if start:
    results2 = testing(
        agent=loaded_model,
        test_pairs=test_pairs,
        agent_type="ppo",
        num_episodes=len(test_pairs),
        env_config=CURRENT_CONFIG,
        save_gifs=True,
        gif_folder=f"GIFS_PPO/SOZ_{MODEL_NAME}",
        start_on_zero=True, 
        print_all=False
    )
else:
    test_results = testing(
        agent=loaded_model, 
        test_pairs=test_pairs, 
        agent_type="ppo", 
        num_episodes=len(test_pairs), 
        env_config=CURRENT_CONFIG,
        save_gifs=True,
        gif_folder=f"GIFS_PPO/{MODEL_NAME}", 
        start_on_zero=False, 
        print_all=False
    )


print("Testing completed.")