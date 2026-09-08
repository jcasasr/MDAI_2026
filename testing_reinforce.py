import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
from gymnasium import spaces
from torch.distributions import Categorical
import wandb

from general import prepare, testing
from env import GlioblastomaPositionalEncoding

class CNNPolicy(nn.Module):
    def __init__(self, obs_shape, action_dim):
        """
        obs_shape: (C, H, W)
        action_dim: number of discrete actions
        """
        super().__init__()
        C, H, W = obs_shape

        self.conv = nn.Sequential(
            nn.Conv2d(C, 16, kernel_size=3, stride=1),
            nn.ReLU(),
            nn.Conv2d(16, 32, kernel_size=3, stride=1),
            nn.ReLU(),
            nn.Conv2d(32, 64, kernel_size=3, stride=1),
            nn.ReLU(),
        )

        # Dynamically compute the flatten size instead of hardcoding 64*5*5
        with torch.no_grad():
            dummy = torch.zeros(1, C, H, W)
            conv_out = self.conv(dummy)
            flat_dim = conv_out.view(1, -1).size(1)

        self.fc = nn.Sequential(
            nn.Linear(flat_dim, 256),
            nn.ReLU(),
            nn.Linear(256, action_dim),
        )

    def forward(self, x):
        """
        x: (B, C, H, W)
        returns: probs: (B, action_dim)
        """
        x = self.conv(x)
        x = x.flatten(1)
        logits = self.fc(x)
        probs = torch.softmax(logits, dim=-1)
        return probs

    def act(self, state, device):
        state_t = torch.as_tensor(state, dtype=torch.float32, device=device)

        if state_t.ndim == 2:        # (H, W) -> (1, 1, H, W)
            state_t = state_t.unsqueeze(0).unsqueeze(0)
        elif state_t.ndim == 3:      # (C, H, W) -> (1, C, H, W)
            state_t = state_t.unsqueeze(0)
        else:
            raise ValueError(f"Unexpected state ndim={state_t.ndim}, shape={state_t.shape}")

        probs = self.forward(state_t)           # (1, action_dim)
        dist = Categorical(probs)
        action = dist.sample()                  # (1,)
        log_prob = dist.log_prob(action)  # (1,)
        dist_entropy = dist.entropy()    # (1,)
        return action.item(), log_prob.squeeze(0), dist_entropy.squeeze(0)  # scalar tensor

class REINFORCEAgent:
    def __init__(self, env_class, train_pairs, env_config,
                 gamma=0.99, lr=1e-4, save_path="reinforce_policy.pt"):

        self.env_class = env_class
        self.train_pairs = train_pairs
        self.env_config = env_config
        self.gamma = gamma
        self.save_path = save_path

        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        # --- Infer observation shape & action_dim from a sample env ---
        sample_img, sample_mask = train_pairs[0]
        sample_env = env_class(sample_img, sample_mask, **env_config)
        obs, _ = sample_env.reset()

        if obs.ndim == 2:
            C, H, W = 1, obs.shape[0], obs.shape[1]
        elif obs.ndim == 3:
            C, H, W = obs.shape
        else:
            raise ValueError(f"Unexpected obs ndim={obs.ndim}, shape={obs.shape}")

        obs_shape = (C, H, W)
        self.action_dim = env_config["action_space"].n

        # --- Policy network ---
        self.policy = CNNPolicy(obs_shape, self.action_dim).to(self.device)
        self.model = self.policy  # for compatibility with your testing() function
        self.optim = optim.Adam(self.policy.parameters(), lr=lr)

        self.best_reward = -1e9

    def make_env(self, img_path, mask_path):
        return self.env_class(img_path, mask_path, **self.env_config)


    def compute_returns(self, rewards):
        """
        Compute discounted returns G_t for a single episode (no normalization here).
        """
        G = 0.0
        returns = []
        for r in reversed(rewards):
            G = r + self.gamma * G
            returns.insert(0, G)
        return returns  # plain Python list of floats

    def update_imitation(self, states, actions):
        """
        Phase 1: Behavioral Cloning (Imitation Learning)
        Uses standard Cross-Entropy to make the policy mimic the expert.
        """
        # Convert lists to tensors
        state_t = torch.as_tensor(np.array(states), dtype=torch.float32, device=self.device)
        action_t = torch.as_tensor(np.array(actions), dtype=torch.long, device=self.device)

        # Forward pass
        logits = self.policy.fc(self.policy.conv(state_t).flatten(1)) # Get raw logits
        
        criterion = nn.CrossEntropyLoss()
        loss = criterion(logits, action_t)

        self.optim.zero_grad()
        loss.backward()
        self.optim.step()
        return loss.item()

    def update_rl(self, log_probs, returns, entropies):
        """
        Phase 2: REINFORCE with Entropy Bonus
        """
        log_probs_t = torch.stack(log_probs)
        returns_t = torch.as_tensor(returns, dtype=torch.float32, device=self.device)
        entropies_t = torch.stack(entropies)

        # Standardize returns to reduce variance
        returns_t = (returns_t - returns_t.mean()) / (returns_t.std() + 1e-8)

        # Policy Gradient Loss + Entropy Bonus (0.01 weight)
        policy_loss = -(log_probs_t * returns_t).mean()
        entropy_loss = -0.01 * entropies_t.mean() 
        
        total_loss = policy_loss + entropy_loss
        
        self.optim.zero_grad()
        total_loss.backward()
        torch.nn.utils.clip_grad_norm_(self.policy.parameters(), max_norm=1.0)
        self.optim.step()
        
        return total_loss.item()
    
    
    def expert_policy(self, env):
        # tumor centroid
        y, x = np.where(env.mask > 0)
        cy, cx = np.mean(y), np.mean(x)

        # current cell center
        gy = env.agent_pos[0] * env.block_size + env.block_size/2
        gx = env.agent_pos[1] * env.block_size + env.block_size/2

        # teacher picks the move that reduces distance the most
        moves = {
            1: (env.agent_pos[0] + 1, env.agent_pos[1]),  # down
            2: (env.agent_pos[0], env.agent_pos[1] + 1),  # right
            3: (env.agent_pos[0] - 1, env.agent_pos[1]),  # up
            4: (env.agent_pos[0], env.agent_pos[1] - 1),  # left
            0: (env.agent_pos[0], env.agent_pos[1])       # stay
        }

        best_action = 0
        best_dist = float("inf")

        for a, (ny, nx) in moves.items():
            if 0 <= ny < env.grid_size and 0 <= nx < env.grid_size:
                ncy = ny * env.block_size + env.block_size/2
                ncx = nx * env.block_size + env.block_size/2
                d = (ncx - cx)**2 + (ncy - cy)**2
                if d < best_dist:
                    best_dist = d
                    best_action = a

        # when in tumor → STAY
        if env.current_patch_overlap_with_lesion() > 0:
            return 0

        return best_action

    def run_human_episode(self, img_path, mask_path):
        states, actions, rewards = [], [], []

        env = self.make_env(img_path, mask_path)
        state, _ = env.reset(force_on_target=False, start_on_zero=False) # random start
        done = False

        found = False
        steps_to_tumor = None
        step = 0

        while not done:
            action = self.expert_policy(env)
            next_state, reward, term, trunc, _ = env.step(action)

            if env.current_patch_overlap_with_lesion() > 0 and not found:
                found = True
                steps_to_tumor = step
                
            states.append(state)
            actions.append(action)
            rewards.append(reward)

            state = next_state
            done = term or trunc
            step += 1
        
        return {
            "states": states,
            "actions": actions,
            "rewards": rewards,
            "found": found,
            "steps_to_tumor": steps_to_tumor,
            "episode_return": sum(rewards),
        }

    def run_episode_with_entropy(self, img_path, mask_path):
        env = self.make_env(img_path, mask_path)
        log_probs, rewards, entropies = [], [], []

        state, _ = env.reset()
        done = False
        while not done:
            action, log_prob, entropy = self.policy.act(state, self.device)
            next_state, reward, terminated, truncated, _ = env.step(action)

            log_probs.append(log_prob)
            rewards.append(reward)
            entropies.append(entropy)

            state = next_state
            done = terminated or truncated
        return log_probs, rewards, entropies
    
    def train(self, epochs=200, bc_epochs=10):
        # PHASE 1: BEHAVIORAL CLONING
        print("=== PHASE 1: BEHAVIORAL CLONING ===")
        for e in range(bc_epochs):
            total_bc_loss = 0
            for img, mask in self.train_pairs:
                episode = self.run_human_episode(img, mask)
                if episode["found"]:
                    loss = self.update_imitation(episode["states"], episode["actions"])
                    total_bc_loss += loss
            print(f"BC Epoch {e} | Loss: {total_bc_loss/len(self.train_pairs):.4f}")
            wandb.log({
                "bc_epoch": e,
                "bc_loss": total_bc_loss/len(self.train_pairs)
            })

        # PHASE 2: REINFORCE
        print("=== PHASE 2: REINFORCE TRAINING ===")
        for e in range(1, epochs + 1):
            all_log_probs = []
            all_returns = []
            all_entropies = []
            epoch_episode_rewards = []

            for img, mask in self.train_pairs:
                log_probs, rewards, entropies = self.run_episode_with_entropy(img, mask)
                returns = self.compute_returns(rewards)

                all_log_probs.extend(log_probs)
                all_returns.extend(returns)
                all_entropies.extend(entropies)
                epoch_episode_rewards.append(sum(rewards))
                
            rl_loss = self.update_rl(all_log_probs, all_returns, all_entropies)
            avg_reward = np.mean(epoch_episode_rewards)
            print(f"Epoch {e}/{epochs} | Loss: {rl_loss:.4f} | Avg Reward: {avg_reward:.2f}")

            # Save best model
            if avg_reward > self.best_reward:
                self.best_reward = avg_reward
                torch.save(self.policy.state_dict(), self.save_path)
                print(f"  New best model saved with avg reward {self.best_reward:.2f}")
            
            wandb.log({
                "epoch": e,
                "rl_loss": rl_loss,
                "avg_reward": avg_reward
            })
        
        torch.save(self.policy.state_dict(), f"models_reinforce/{RUN_NAME}_final.pt")
        print(f"Training finished. Final model saved to models_reinforce/{RUN_NAME}_final.pt")

CURRENT_CONFIG = {
    'dataset': "polyp",
    'mode': "test",
    'grid_size': 8,
    'action_space': spaces.Discrete(5), 
    'rewards': [
        100.0,   # Correct Stay (Goal)
        -150.0,  # Wrong Stay (False Positive penalty)
        2.5,    # Move into tumor (The "Warm" hint)
        -2.5,   # Exit tumor (The "Cold" hint)
        0.5,    # Move within tumor (Encourage staying on target)
        -0.1    # Step penalty (Urgency)
    ],
    'max_steps': 75
}
RUN_NAME = "POLYP_reinforce_007"
start = True  # Whether to start on zero or random position

test_pairs = prepare(dataset=CURRENT_CONFIG['dataset'], mode='test')

agent = REINFORCEAgent(
    env_class=GlioblastomaPositionalEncoding,
    train_pairs=test_pairs,
    env_config=CURRENT_CONFIG,
    gamma=0.99,
    lr=1e-4,
    save_path=f"models_reinforce/{RUN_NAME}_best.pt"
)

agent.policy.load_state_dict(torch.load(f"models_reinforce/{RUN_NAME}_best.pt"))


if start:
    results2 = testing(
        agent=agent,
        test_pairs=test_pairs,
        agent_type="reinforce",
        num_episodes=len(test_pairs),
        env_config=CURRENT_CONFIG,
        save_gifs=True,
        gif_folder=f"GIFS_REINFORCE/SOZ_GIFs_Testing_{RUN_NAME}",
        start_on_zero=True, 
        print_all=False
    )
else:
    results = testing(
        agent=agent,
        test_pairs=test_pairs,
        agent_type="reinforce",
        num_episodes=len(test_pairs),
        env_config=CURRENT_CONFIG,
        save_gifs=True,
        gif_folder=f"GIFS_REINFORCE/GIFs_Testing_{RUN_NAME}", 
        print_all=False
    )