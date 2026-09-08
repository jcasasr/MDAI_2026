import numpy as np
import matplotlib.pyplot as plt
import sys
import torch
from copy import deepcopy
import wandb
import random

from tqdm.auto import tqdm

SEED = 42

class DQNAgent:
    
    def __init__(self, env_config, dnnetwork, buffer_class, train_pairs, validation_pairs, env_class, 
                 epsilon=0.1, eps_decay=0.99, eps_decay_type="subtraction", epsilon_min=0.01, batch_size=32, gamma=0.99, 
                 memory_size=1500, buffer_initial=150, save_name="Glioblastoma"):
        self.env_config = env_config
        self.env_class = env_class
        
        self.train_pairs = train_pairs
        self.validation_pairs = validation_pairs
        
        self.dnnetwork = dnnetwork # main network
        self.target_network = deepcopy(dnnetwork) # prevents the target Q-values from changing with every single update
        self.target_network.optimizer = None # paper said target net is only  weights, no optimizer

        self.epsilon = epsilon # initial epsilon for e-greedy
        self.eps_decay = eps_decay # decay of epsilon after each episode to balance exploration and exploitation
        self.eps_decay_type = eps_decay_type

        self.epsilon_min = 0 if self.epsilon == 0 else epsilon_min
            
        self.batch_size = batch_size # size of the mini-batch for training
        self.gamma = gamma
        
        self.buffer_initial = buffer_initial # number of random experiences to fill the buffer before training
        
        self.save_name = save_name
        
        # block of the last X episodes to calculate the average reward 
        self.nblock = 100 
                
        # Buffer
        self.buffer = buffer_class(capacity=memory_size)
        self.best_success = -1
        self.initialize()
        
    
    def initialize(self): # reset variables at the beginning of training
        self.update_loss = []
        self.training_rewards = []
        self.mean_training_rewards = []
        self.sync_eps = []
        self.total_reward = 0
        self.step_count = 0
        self.state0 = None  # Initialize as None since we don't have env yet
        
    ## Take new action
    def take_step(self, eps, mode='train'):
        if mode == 'explore':
            # random action in burn-in and in the exploration phase (epsilon)
            action = self.env.action_space.sample() 
        else:
            # Action based on the Q-value (max Q-value)
            action = self.dnnetwork.get_action(self.state0, eps)
        self.step_count += 1
            
        # Execute action and get reward and new state
        new_state, reward, terminated, truncated, _ = self.env.step(action)
        done = terminated or truncated
        self.total_reward += reward
        
        # save experience in the buffer
        self.buffer.append(self.state0, action, reward, done, new_state)
        self.state0 = new_state.copy()
        
        if done:
            self.state0 = self.env.reset()[0]
        return done, reward # THE REWARD RETURN IS FOR DEBUGGING 

            
    ## Training
    def train(self, train_pairs, validation_pairs, gamma=0.99, max_episodes=50000, 
              dnn_update_frequency=4,
              dnn_sync_frequency=200, 
              val_frequency=100):
        
        self.gamma = gamma

        # Fill the buffer with N random experiences
        print("Filling replay buffer...")
        
        pbar = tqdm(total=self.buffer_initial * len(train_pairs), 
                    desc="Buffer Filling", 
                unit="exp", unit_scale=True, 
                file=sys.stdout,
                dynamic_ncols=True,
                bar_format='{l_bar}{bar}| {n_fmt}/{total_fmt} [{elapsed}<{remaining}, {rate_fmt}]')
        
        first = True # just to detect the input channel number
        
        for img_path, mask_path in train_pairs:
            self.env = self.env_class(img_path, mask_path, **self.env_config)
            self.state0, _ = self.env.reset(seed=SEED)
            if first:
                first = False
                # Detect input channels from first environment so i can use with both envs and dqns
                if self.state0.ndim == 2:
                    self.input_channels = 1 # in case it were (60, 60)
                else:
                    self.input_channels = self.state0.shape[0] # in case it were (3, 60, 60)
            # Run short episode on this image
            for _ in range(self.buffer_initial):
                self.take_step(self.epsilon, mode='explore')
            pbar.update(self.buffer_initial)
        pbar.close()
        print(f"Buffer filled with {len(self.buffer.buffer)} experiences")

            
        # Store metrics locally to plot
        self.episode_rewards = []
        self.mean_rewards = []
        self.epsilon_values = []
        self.loss_values = []
 
        episode = 0
        training = True
        
        print("Training...")
        pbar = tqdm(total=max_episodes, 
                    desc="Progress", 
                unit="ep", unit_scale=True, 
                file=sys.stdout,
                dynamic_ncols=True,
                bar_format='{l_bar}{bar}| {n_fmt}/{total_fmt} [{elapsed}<{remaining}, {rate_fmt}]')

        while training and episode < max_episodes:
            if episode < 500: #curriculum learning
                force_on_target = True
            elif episode < 1000:
                force_on_target = (random.random() < 0.8)
            elif episode < 2000:
                force_on_target = (random.random() < 0.5)
            else:
                force_on_target = (random.random() < 0.2)
                
            img_path, mask_path = random.choice(train_pairs)
            self.env = self.env_class(img_path, mask_path, **self.env_config)
            self.state0, _ = self.env.reset(seed=SEED, force_on_target=force_on_target)
            self.total_reward = 0
            
            # DEBUGGING
            pos_rewards = 0
            neg_rewards = 0
            
            for step in range(self.env.max_steps):
                gamedone, reward = self.take_step(self.epsilon, mode='train') # THE REWARD RETURN IS FOR DEBUGGING

                # DEBUGGING
                if reward > 0: pos_rewards += 1
                else: neg_rewards += 1
                # END OF DEBUGGING
                
                # Upgrade main network
                if self.step_count % dnn_update_frequency == 0:
                    self.update()
                    
                # Synchronize the main network and the target network
                if self.step_count % dnn_sync_frequency == 0:
                    self.target_network.load_state_dict(
                        self.dnnetwork.state_dict())
                    self.sync_eps.append(episode)
                    
                if gamedone:
                    episode += 1
                    pbar.update(1)
          
                    # Save the rewards
                    self.training_rewards.append(self.total_reward)
                    
                    # Calculate the average reward for the last X episodes
                    if len(self.training_rewards) >= self.nblock:
                        mean_rewards = np.mean(self.training_rewards[-self.nblock:])
                    else:
                        mean_rewards = np.mean(self.training_rewards)  # Use all rewards if less than nblock
                    
                    self.mean_training_rewards.append(mean_rewards)

                    episode_loss_mean = (
                        np.mean(self.update_loss) if self.update_loss else 0.0
                    )

                    wandb.log({
                        "train/episode": episode,
                        "train/episode_reward": self.total_reward,
                        "train/mean_reward": mean_rewards,
                        "train/loss_episode_mean": episode_loss_mean,
                        "train/epsilon": self.epsilon
                    }, step=self.step_count)
                    
                    if episode % 15 == 0:
                        pbar.write("Episode {:d} | Episode reward {:.2f} | Mean Rewards {:.2f} | Epsilon {:.4f} | Loss {:.4f}".format(
                            episode, self.total_reward, mean_rewards, self.epsilon, episode_loss_mean))
                        pbar.write(f"      Positive rewards: {pos_rewards}, Negative rewards: {neg_rewards}") # DEBUGGING

                    if self.validation_pairs is not None and episode % val_frequency == 0:
                        pbar.write("="*30)
                        pbar.write(f"  VALIDATING AT EPISODE {episode}")
                        pbar.write("="*30)
                        self.run_validation(self.validation_pairs, episode)
                        
                    # Append metrics to lists for plotting
                    self.episode_rewards.append(self.total_reward)
                    self.mean_rewards.append(mean_rewards)
                    self.epsilon_values.append(self.epsilon)
                    self.loss_values.append(episode_loss_mean)
                    
                    self.update_loss = []
                        
                    # Check if there are still episodes left
                    if episode >= max_episodes:
                        training = False
                        print('\nEpisode limit reached.')
                        break
                    
                    if self.eps_decay_type == "exponential":
                        # Update epsilon according to exponential decay
                        self.epsilon = max(self.epsilon * self.eps_decay, self.epsilon_min)
                    else:
                        self.epsilon = max(self.epsilon - self.eps_decay, self.epsilon_min) 
                        
                    torch.save(self.dnnetwork.state_dict(), "models_DQN/" + self.save_name + ".dat")
                    break
            
        pbar.close()
                    
        # PLOTTING
        # fig, axes = plt.subplots(2, 2, figsize=(10, 8))  # Create a 2x2 grid of subplots
        # axes = axes.ravel()  # Flatten the axes array for easier indexing

        # # Plot episode rewards
        # axes[0].plot(self.episode_rewards)
        # axes[0].set_xlabel('Episode')
        # axes[0].set_ylabel('Episode Reward')
        # axes[0].set_title('Episode Rewards Over Time')

        # # Plot mean rewards
        # axes[1].plot(self.mean_rewards)
        # axes[1].set_xlabel('Episode')
        # axes[1].set_ylabel('Mean Reward')
        # axes[1].set_title('Mean Rewards Over Time')

        # # Plot epsilon values
        # axes[2].plot(self.epsilon_values)
        # axes[2].set_xlabel('Episode')
        # axes[2].set_ylabel('Epsilon Values')
        # axes[2].set_title('Epsilon Values Over Time')

        # # Plot loss values
        # axes[3].plot(self.loss_values)
        # axes[3].set_xlabel('Episode')
        # axes[3].set_ylabel('Loss')
        # axes[3].set_title('Loss Over Time')

        # # Adjust layout
        # # fig.suptitle('Training Performance', fontsize=16)  # Add a main title
        # plt.tight_layout(rect=[0, 0.03, 1, 0.95])  # Adjust spacing between subplots and title
        # plt.show()


    def run_validation(self, val_pairs, current_episode):
        print(f"\n--- Running Validation at Episode {current_episode} ---")
        val_rewards = []
        successes = 0
        
        # Select samples
        num_samples = len(val_pairs)
        test_samples = val_pairs
        
        self.dnnetwork.eval() 
        
        with torch.no_grad(): # no gradients are computed
            for i, (img_p, mask_p) in enumerate(test_samples):
                v_env = self.env_class(img_p, mask_p, **self.env_config)
                state, _ = v_env.reset(seed=SEED)
                ep_reward = 0
                
                # For visualization
                path_coords = [] # To store (y, x) or (row, col)
                episode_success = False
                
                for _ in range(v_env.max_steps):
                    if hasattr(v_env, 'agent_pos'):
                        path_coords.append(deepcopy(v_env.agent_pos))
                    
                    action = self.dnnetwork.get_action(state, epsilon=0.0)
                    state, reward, terminated, truncated, _ = v_env.step(action)
                    ep_reward += reward
                    if terminated and v_env.current_patch_overlap_with_lesion() > 0:
                        episode_success = True

                    
                    if terminated or truncated:
                        break
                
                if episode_success: successes += 1
                val_rewards.append(ep_reward)

        avg_val_reward = np.mean(val_rewards)
        success_rate = (successes / num_samples) * 100
        
        wandb.log({
            'val/episode': current_episode,
            'val/avg_reward': avg_val_reward,
            'val/success_rate': success_rate,
        }, step=self.step_count)
        
        # Save Best Model
        if not hasattr(self, 'best_success'): self.best_success = -1
        if success_rate > self.best_success:
            self.best_success = success_rate
            torch.save(self.dnnetwork.state_dict(), f"models_DQN/{self.save_name}_BEST_VAL.dat")
            print(f"*** New Best Success Rate: {success_rate:.1f}%! Model Saved. ***")

        self.dnnetwork.train()
        print(f"Validation Mean Reward: {avg_val_reward:.2f} | Success Rate: {success_rate:.1f}%")
        print("--------------------------------------------\n")

    # Loss calculation           
    def calculate_loss(self, batch):
        # Separate the variables of the experience and convert them to tensors
        states, actions, rewards, dones, next_states = batch
        device = self.dnnetwork.device

        # Add channel dimension
        states = torch.tensor(states, dtype=torch.float32, device=device)
        next_states = torch.tensor(next_states, dtype=torch.float32, device=device)        # If grayscale (H,W) was stored → convert to (1,H,W)
        if self.input_channels == 1 and states.ndim == 3:
            states = states.unsqueeze(1)
            next_states = next_states.unsqueeze(1)

        rewards_vals = torch.FloatTensor(rewards).to(device=device) 
        actions_vals = torch.LongTensor(np.array(actions)).reshape(-1,1).to(device=device)
        dones_t = torch.BoolTensor(dones).to(device=device)
        
        # Obtain the Q values of the main network
        qvals = torch.gather(self.dnnetwork.get_qvals(states), 1, actions_vals)
        
        # Obtain the target Q values.
        # The detach() parameter prevents these values from updating the target network
        qvals_next_all = self.target_network.get_qvals(next_states)  # Shape: [batch_size, n_actions]
        qvals_next = torch.max(qvals_next_all, dim=1)[0].detach()    # Shape: [batch_size]

        # 0 in terminal states
        qvals_next[dones_t] = 0.0 
        
        # print("qvals_next.shape", qvals_next.shape, "dones_t.shape", dones_t.shape) # debugging
        
        # Calculate the Bellman equation
        expected_qvals = (self.gamma * qvals_next) + rewards_vals
        
        # Calculate the loss
        # loss = torch.nn.MSELoss()(qvals, expected_qvals.reshape(-1,1))
        # Use Huber loss instead of MSELoss
        
        # loss = torch.nn.SmoothL1Loss()(qvals, expected_qvals.reshape(-1,1))
        loss = torch.nn.SmoothL1Loss()(qvals.clamp(-10,10), expected_qvals.reshape(-1,1).clamp(-10,10))
        return loss
    

    def update(self, num_buffers=4):
        # Check if buffer has enough experiences
        if len(self.buffer.buffer) < self.batch_size:
            return
             
        # Remove any gradient
        self.dnnetwork.optimizer.zero_grad()  
        
        batch = self.buffer.sample_batch(self.batch_size)

        # Calculate the loss
        loss = self.calculate_loss(batch) 
        # Difference to get the gradients
        loss.backward() 
        
        # add gradient clipping
        # Clipping uses L2 of gradients - “Don’t let gradients go crazy. Keep training stable.”
        # Regularization uses L2 of weights - “Don’t let weights get too big. Keep the model simple.”
        torch.nn.utils.clip_grad_norm_(self.dnnetwork.parameters(), max_norm=1.0)
        # torch.nn.utils.clip_grad_norm_(self.dnnetwork.parameters(), max_norm=0.1)
        # Apply the gradients to the neural network
        self.dnnetwork.optimizer.step() 
        
        wandb.log(
            {"train/loss_step": loss.item()},
            step=self.step_count
        )
        # Save loss values
        # if self.dnnetwork.device == 'cuda':
        #     self.update_loss.append(loss.detach().cpu().numpy())
        # else:
        #     self.update_loss.append(loss.detach().numpy())
        self.update_loss.append(loss.item())
