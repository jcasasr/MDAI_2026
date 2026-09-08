import numpy as np
import os
import torch
from PIL import Image
import pandas as pd

from env import GlioblastomaPositionalEncoding

SEED = 42

def prepare(dataset = "glio", mode = "train"):
    if dataset == "glio":
        if mode == "train":
            base_dir = "/Users/martina/code/4year/new/data/glio_data/training_set_npy"
            csv_path = "/Users/martina/code/4year/new/data/glio_data/training_set.csv"
        elif mode == "val":
            print("Preparing validation set.")
            base_dir = "/Users/martina/code/4year/new/data/glio_data/validation_set_npy"
            csv_path = "/Users/martina/code/4year/new/data/glio_data/validation_set.csv"
        elif mode == "test":
            print("Preparing testing set.")
            base_dir = "/Users/martina/code/4year/new/data/glio_data/testing_set_npy"
            csv_path = "/Users/martina/code/4year/new/data/glio_data/testing_set.csv"
        
        # Construct image and mask filenames
        df = pd.read_csv(csv_path)
        
        df["image_path"] = df.apply(
            lambda row: os.path.join(base_dir, f"{row['Patient']:03d}_{row['SliceIndex']}.npy"), axis=1
        )
        df["mask_path"] = df.apply(
            lambda row: os.path.join(base_dir, f"{row['Patient']:03d}_{row['SliceIndex']}_mask.npy"), axis=1
        )

    elif dataset == "polyp":
        if mode == "train":
            base_dir = "/Users/martina/code/4year/new/data/polyp_data/po_training_set_npy"
            csv_path = "/Users/martina/code/4year/new/data/polyp_data/po_training_set.csv"
        elif mode == "val":
            print("Preparing validation set.")
            base_dir = "/Users/martina/code/4year/new/data/polyp_data/po_validation_set_npy"
            csv_path = "/Users/martina/code/4year/new/data/polyp_data/po_validation_set.csv"
        else:
            print("Preparing testing set.")
            base_dir = "/Users/martina/code/4year/new/data/polyp_data/po_testing_set_npy"
            csv_path = "/Users/martina/code/4year/new/data/polyp_data/po_testing_set.csv"
        
        # Construct image and mask filenames
        df = pd.read_csv(csv_path)
        
        df["image_path"] = df.apply(
            lambda row: os.path.join(base_dir, f"{row['Patient']}.npy"), axis=1
        )
        df["mask_path"] = df.apply(
            lambda row: os.path.join(base_dir, f"{row['Patient']}_mask.npy"), axis=1
        )

    else:
        print("Dataset not recognized. Please choose 'glio' or 'polyp'.")
        
    # Sanity check (optional)
    pairs = [
        (img, mask)
        for img, mask in zip(df["image_path"], df["mask_path"])
        if os.path.exists(img) and os.path.exists(mask)
    ]

    print(f"✅ Found {len(pairs)} pairs out of {len(df)} listed in CSV.")
    return pairs

def testing(agent, test_pairs, agent_type, num_episodes=None, env_config=None, save_gifs=True, gif_folder="TEST_GIFS", start_on_zero=False, print_all=True):
    if num_episodes is None:
        num_episodes = len(test_pairs)
    
    # Create GIF folder if needed
    if save_gifs and not os.path.exists(gif_folder):
        os.makedirs(gif_folder)
    
    # Set model to evaluation mode
    if agent_type.lower() == "dqn":
        agent.dnnetwork.eval()
    # elif agent_type.lower() == "ppo":
    #     agent.model.eval()
    if agent_type.lower() == "reinforce":
        agent.policy.eval()
    
    results = {
        'hard_success': [], 'hard_failure': [],
        'timeout_success': [], 'timeout_failure': [],
        'average_reward': [],
        'steps_to_find_tumor': [], 'total_tumor_reward': [],
        'tumor_sizes_pixels': [], 'tumor_sizes_percentage': [],
        'episode_details': []
    }
    
    grid_size = env_config.get('grid_size', 4)
    rewards = env_config.get('rewards', [5.0, -1.0, -0.2])
    action_space = env_config.get('action_space', None)
    
    ### OUTER LOOP = PER IMAGE ###
    for i in range(min(num_episodes, len(test_pairs))):
        img_path, mask_path = test_pairs[i]
        
        dataset_type = env_config.get('dataset', 'glio') # Get dataset from config
        
        # Create environment
        if hasattr(agent, 'env_class'):
            env = agent.env_class(img_path, mask_path, grid_size=grid_size, rewards=rewards, action_space=action_space, dataset=dataset_type)
        else:
            env = GlioblastomaPositionalEncoding(img_path, mask_path, grid_size=grid_size, rewards=rewards, action_space=action_space, dataset=dataset_type)
        
        # Get tumor size information for this episode
        tumor_size_pixels = count_tumor_pixels(env)
        total_pixels = env.image.shape[0] * env.image.shape[1]
        tumor_size_percentage = (tumor_size_pixels / total_pixels) * 100
        
        ### INNER LOOP = MULTIPLE RUNS PER IMAGE ###
        if start_on_zero == True:
            runs_per_image = 1
        else:
            runs_per_image = 5
        for run_idx in range(runs_per_image): # 5 runs per image to average out randomness
            state, _ = env.reset(start_on_zero=start_on_zero)
            
            terminated_by_stay = False
            total_reward = 0
            found_tumor = False
            tumor_positions_visited = set()
            steps_to_find = env.max_steps
            tumor_rewards = 0
            
            # For action distribution tracking
            action_counts = np.zeros(env.action_space.n)
            
            # For GIF creation
            frames = []
            if save_gifs:
                initial_frame = env.render(show=False)
                if initial_frame is not None:
                    frames.append(initial_frame)
                    
            ### EPISODE LOOP ###
            for step in range(env.max_steps):
                with torch.no_grad():
                    if agent_type.lower() == "dqn":
                        action = agent.dnnetwork.get_action(state, epsilon=0.00)
                        action_idx = action
                    elif agent_type.lower() == "ppo":
                        # if env_config.get('dataset') == 'polyp':
                        #     # If the state is (5, 60, 60), stack it to (5, 60, 60, 3)
                        #     if len(state.shape) == 3:
                        #         state = np.stack([state] * 3, axis=-1)
                        action, _states = agent.predict(state, deterministic=True)
                        action_idx = int(action)
                    elif agent_type.lower() == "reinforce":
                        action, _, _ = agent.policy.act(state, agent.device)  # handles tensor conversion internally
                        action_idx = action

                
                action_counts[action_idx] += 1
                
                next_state, reward, terminated, truncated, _ = env.step(action_idx)
                state = next_state
                total_reward += reward
                
                # Track tumor-related metrics
                current_overlap = env.current_patch_overlap_with_lesion()
                if current_overlap > 0:
                    tumor_positions_visited.add(tuple(env.agent_pos))
                    if not found_tumor:
                        found_tumor = True
                        steps_to_find = step + 1
                    
                    # Count positive rewards (when on tumor)
                    if reward > 0:
                        tumor_rewards += 1
                
                # Capture frame for GIF
                if save_gifs:
                    frame = env.render(show=False)
                    if frame is not None:
                        frames.append(frame)
                
                if terminated or truncated:
                    terminated_by_stay = (action_idx == 0)
                    break
            
            time_out = (not terminated_by_stay and (step+1) >= env.max_steps)
            final_overlap = env.current_patch_overlap_with_lesion()
            
            hard_success = terminated_by_stay and final_overlap > 0
            hard_failure = terminated_by_stay and final_overlap == 0
            timeout_success = time_out and final_overlap > 0
            timeout_failure = time_out and final_overlap == 0
            
            results['hard_success'].append(hard_success)
            results['hard_failure'].append(hard_failure)
            results['timeout_success'].append(timeout_success)
            results['timeout_failure'].append(timeout_failure)
            results['average_reward'].append(total_reward)
            results['steps_to_find_tumor'].append(steps_to_find)
            results['total_tumor_reward'].append(tumor_rewards)
            results['tumor_sizes_pixels'].append(tumor_size_pixels)
            results['tumor_sizes_percentage'].append(tumor_size_percentage)
            
            # Save GIF
            gif_path = None
            if save_gifs and frames:
                
                # if episode is hard success
                if hard_success:
                    gif_path = os.path.join(gif_folder, f"episode_{i}_{run_idx}_{os.path.basename(img_path).split('.')[0]}_HS.gif")
                elif hard_failure:
                    gif_path = os.path.join(gif_folder, f"episode_{i}_{run_idx}_{os.path.basename(img_path).split('.')[0]}_HF.gif")
                elif timeout_success:
                    gif_path = os.path.join(gif_folder, f"episode_{i}_{run_idx}_{os.path.basename(img_path).split('.')[0]}_TS.gif")
                elif timeout_failure:
                    gif_path = os.path.join(gif_folder, f"episode_{i}_{run_idx}_{os.path.basename(img_path).split('.')[0]}_TF.gif")
                else:
                    print("Unexpected case: neither hard nor timeout success/failure.")
                # Convert frames to PIL Images and save as GIF
                pil_frames = [Image.fromarray(frame) for frame in frames]
                pil_frames[0].save(
                    gif_path,
                    save_all=True,
                    append_images=pil_frames[1:],
                    duration=500,  # milliseconds per frame
                    loop=0
                )
                if i == 0: # just first
                    print(f"Saved GIF for episode {i} at {gif_path}")
            elif save_gifs == False:
                if i == 0:
                    print("GIF saving disabled.")
            
            episode_detail = {
                'image_path': img_path,
                'episode_idx': i,
                'run_idx': run_idx,
                'terminated_by_stay': terminated_by_stay,
                'timed_out': time_out,
                'hard_success': hard_success,
                'hard_failure': hard_failure,
                'timeout_success': timeout_success,
                'timeout_failure': timeout_failure,
                'final_overlap': final_overlap > 0,
                'total_reward': total_reward,
                'steps_to_find_tumor': steps_to_find,
                'tumor_rewards': tumor_rewards,
                'tumor_size_pixels': tumor_size_pixels,
                'tumor_size_percentage': tumor_size_percentage,
                'action_distribution': action_counts / np.sum(action_counts),
                'action_counts_raw': action_counts,
                'gif_path': gif_path
            }
            results['episode_details'].append(episode_detail)
    
    
    # Calculate separate action distributions
    hard_success_eps = [ep for ep in results['episode_details'] if ep['hard_success']]
    hard_failure_eps = [ep for ep in results['episode_details'] if ep['hard_failure']]
    timeout_success_eps = [ep for ep in results['episode_details'] if ep['timeout_success']]
    timeout_failure_eps = [ep for ep in results['episode_details'] if ep['timeout_failure']]

    
    action_dist_hard_success = calculate_separate_action_distribution(hard_success_eps)
    action_dist_hard_failure = calculate_separate_action_distribution(hard_failure_eps)
    action_dist_timeout_success = calculate_separate_action_distribution(timeout_success_eps)
    action_dist_timeout_failure = calculate_separate_action_distribution(timeout_failure_eps)
    
    # Calculate overall metrics with new tumor size statistics
    overall_results = {
        'hard_success_rate': np.mean(results['hard_success']),
        'hard_failure_rate': np.mean(results['hard_failure']),
        'timeout_success_rate': np.mean(results['timeout_success']),
        'timeout_failure_rate': np.mean(results['timeout_failure']),
        
        'average_reward': np.mean(results['average_reward']),
        'avg_steps_to_find_tumor': np.mean(results['steps_to_find_tumor']),
        'avg_tumor_rewards': np.mean(results['total_tumor_reward']),
        
        'biggest_tumor_pixels': np.max(results['tumor_sizes_pixels']),
        'smallest_tumor_pixels': np.min(results['tumor_sizes_pixels']),
        'biggest_tumor_percentage': np.max(results['tumor_sizes_percentage']),
        'smallest_tumor_percentage': np.min(results['tumor_sizes_percentage']),
        'avg_tumor_size_pixels': np.mean(results['tumor_sizes_pixels']),
        'avg_tumor_size_percentage': np.mean(results['tumor_sizes_percentage']),
        
        'action_distribution': calculate_overall_action_distribution(results['episode_details']),
        'action_distribution_hard_success': action_dist_hard_success,
        'action_distribution_hard_failure': action_dist_hard_failure,
        'action_distribution_timeout_success': action_dist_timeout_success,
        'action_distribution_timeout_failure': action_dist_timeout_failure,
        'episode_details': results['episode_details']
    }
    
    # Print summary
    print("\n" + "="*60)
    print(f"TEST RESULTS ({agent_type.upper()} Agent)")
    print("="*60)
    print(f"✅Hard Success (correct STAY): {overall_results['hard_success_rate']*100:.2f}%")
    print(f"   ❌Hard Failure (wrong STAY): {overall_results['hard_failure_rate']*100:.2f}%")
    print(f"✔️Timeout Success (lucky): {overall_results['timeout_success_rate']*100:.2f}%")
    print(f"   ❌Timeout Failure: {overall_results['timeout_failure_rate']*100:.2f}%")

    print(f"Average Episode Reward: {overall_results['average_reward']:.2f}")
    print(f"Average Steps to Find Tumor: {overall_results['avg_steps_to_find_tumor']:.2f}")
    print(f"Average Tumor Rewards per Episode: {overall_results['avg_tumor_rewards']:.2f}")
    print(f"Tumor Size Statistics:")
    print(f"  Biggest Tumor: {overall_results['biggest_tumor_pixels']:.0f} pixels ({overall_results['biggest_tumor_percentage']:.2f}%)")
    print(f"  Smallest Tumor: {overall_results['smallest_tumor_pixels']:.0f} pixels ({overall_results['smallest_tumor_percentage']:.2f}%)")
    print(f"  Average Tumor: {overall_results['avg_tumor_size_pixels']:.0f} pixels ({overall_results['avg_tumor_size_percentage']:.2f}%)")
    print(f"Overall Action Distribution: {overall_results['action_distribution']}")
    print(f"  Hard Successful Episodes: {overall_results['action_distribution_hard_success']}")
    print(f"  Hard Unsuccessful Episodes: {overall_results['action_distribution_hard_failure']}")
    print(f"  Timeout Successful Episodes: {overall_results['action_distribution_timeout_success']}")
    print(f"  Timeout Unsuccessful Episodes: {overall_results['action_distribution_timeout_failure']}")
    
    # Print individual episode results
    if print_all:
        print(f"\nDetailed Results for {len(results['episode_details'])//runs_per_image} episodes:")
        print("-" * 80)
        for i, detail in enumerate(results['episode_details']):
            if detail['run_idx'] == 0:
                print(f"Episode {detail['episode_idx']}: {os.path.basename(detail['image_path'])}, gif in: {detail['gif_path'] if detail['gif_path'] else 'N/A'}")
                print(f"  Hard Success: {detail['hard_success']}, Hard Failure: {detail['hard_failure']}")
                print(f"  Timeout Success: {detail['timeout_success']}, Timeout Failure: {detail['timeout_failure']}")
                print(f"  Total Reward: {detail['total_reward']:.2f}, Steps to Find: {detail['steps_to_find_tumor']}")
                print(f"  Tumor Size: {detail['tumor_size_pixels']} pixels ({detail['tumor_size_percentage']:.2f}%)")
                print(f"  Action Distribution: {detail['action_distribution']}")
                if detail['gif_path']:
                    print(f"  GIF saved: {detail['gif_path']}")
                print()
    
    return overall_results

def count_tumor_pixels(env):
    """Count total number of tumor pixels in the mask"""
    if hasattr(env, 'mask'):
        return np.sum(env.mask > 0)
    elif hasattr(env, 'original_mask'):
        return np.sum(env.original_mask > 0)
    else:
        # Fallback: try to access the mask through available attributes
        try:
            mask = env.lesion_mask if hasattr(env, 'lesion_mask') else None
            if mask is not None:
                return np.sum(mask > 0)
        except:
            pass
    return 0

def calculate_overall_action_distribution(episode_details):
    total_actions = np.zeros_like(episode_details[0]['action_counts_raw'])
    
    for detail in episode_details:
        total_actions += detail['action_counts_raw']
    
    # Normalize to get overall distribution
    overall_dist = total_actions / np.sum(total_actions)
    return overall_dist

def calculate_separate_action_distribution(episode_list):
    """Calculate action distribution for a specific list of episodes"""
    if len(episode_list) == 0:
        return np.array([])  # Return empty array if no episodes
    
    total_actions = np.zeros_like(episode_list[0]['action_distribution'])
    
    for episode in episode_list:
        total_actions += episode['action_distribution']
    
    # Normalize to get distribution
    distribution = total_actions / len(episode_list)
    return distribution
