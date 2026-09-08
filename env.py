import cv2
import gymnasium as gym
from gymnasium import spaces
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import matplotlib.pyplot as plt
import sys
from PIL import Image, ImageDraw, ImageFont

class GlioblastomaPositionalEncoding(gym.Env):
    metadata = {"render_modes": ["human"], "render_fps": 4} 

    def __init__(self, image_path, mask_path, dataset='glio', mode='train', grid_size=4, tumor_threshold=0.01, rewards = [10.0, -5.0, 10.0, -10, 0.0, -0.01], action_space=spaces.Discrete(3), max_steps=20, render_mode="human"): # cosntructor with the brain image, the mask and a size
        super().__init__()
        self.dataset = dataset
        
        self.image = np.load(image_path).astype(np.float32)
        self.mask = np.load(mask_path).astype(np.uint8)
        
        img_min, img_max = self.image.min(), self.image.max()
        if img_max > 1.0:
            self.image = (self.image - img_min) / (img_max - img_min + 1e-8)

        self.grid_size = grid_size
        self.block_size = self.image.shape[0] // grid_size
        
        self.action_space = action_space
        self.tumor_threshold = tumor_threshold
        self.rewards = rewards
        self.render_mode = render_mode

        # UPDATED: for both datasets
        if self.dataset == 'glio':
            self.observation_space = spaces.Box(
                low=0, high=1,
                shape=(3, self.block_size, self.block_size),  # (3, H, W)
                dtype=np.float32
            )
        elif self.dataset == 'polyp':
            self.observation_space = spaces.Box(
                low=0, high=1,
                shape=(5, self.block_size, self.block_size),  # (image channels + 2 positional encodings) (5, H, W)
                dtype=np.float32
            )

        self.agent_pos = [0, 0]
        self.prev_pos = None
        self.prev_prev_pos = None
        self.current_step = 0
        if max_steps == 0:
            self.max_steps = sys.maxsize
        else:
            self.max_steps = max_steps
        
    def _random_shift(self):
        pad = 20
        if self.image.ndim == 3:
            H, W, C = self.image.shape
        else:
            H, W = self.image.shape
            C = None # grayscale

        while True:
            if C is not None:
                canvas = np.zeros((H + 2*pad, W + 2*pad, C), dtype=self.image.dtype)
            else:
                canvas = np.zeros((H + 2*pad, W + 2*pad), dtype=self.image.dtype)
            
            canvas_mask = np.zeros((H + 2*pad, W + 2*pad), dtype=self.mask.dtype)

            # random offset
            y_off = np.random.randint(0, 2*pad+1)
            x_off = np.random.randint(0, 2*pad+1)

            # place original image
            if C is not None:
                canvas[y_off:y_off+H, x_off:x_off+W, :] = self.image
            else:
                canvas[y_off:y_off+H, x_off:x_off+W] = self.image
            
            canvas_mask[y_off:y_off+H, x_off:x_off+W] = self.mask

            # crop
            if C is not None:
                new_image = canvas[pad:pad+H, pad:pad+W, :]
            else:
                new_image = canvas[pad:pad+H, pad:pad+W]

            new_mask = canvas_mask[pad:pad+H, pad:pad+W]

            # check if new_mask still contains tumor
            if np.sum(new_mask > 0) > 0:
                self.image = new_image
                self.mask = new_mask
                return

    def reset(self, seed=None, options=None, force_on_target=False, start_on_zero=False):
        super().reset(seed=seed)
        
        self._random_shift()  # Apply random shift on reset

        if start_on_zero:
            self.agent_pos = [0, 0]
        else:
            if force_on_target: # start on tumor so it can see good reward if stay
                tumor_indices = np.where(self.mask > 0)
                # Pick a random pixel within the tumor
                idx = np.random.randint(len(tumor_indices[0]))
                one = tumor_indices[0][idx]
                two = tumor_indices[1][idx]
                self.agent_pos = [one // self.block_size, two // self.block_size]
            else:
                # Standard random start
                self.agent_pos = [np.random.randint(self.grid_size), np.random.randint(self.grid_size)]
        
        self.current_step = 0
        self.prev_pos = None
        self.prev_prev_pos = None
        obs = self._get_obs()
        info = {}
        return obs, info

    def step(self, action):
        self.current_step += 1
        prev_pos = self.agent_pos.copy()    # store position BEFORE applying action
                
        if action == 0: # END episode
            reward = self._get_reward(action, prev_pos)
            terminated = True
            obs = self._get_obs()
            
            # return obs, reward, terminated, False, {}
            self.prev_prev_pos = self.prev_pos.copy() if self.prev_pos is not None else None
            self.prev_pos = prev_pos.copy()     # store for next step
            return obs, reward, terminated, False, {}
        
        # Apply action (respect grid boundaries)
        if self.action_space.n == 3:
            if action == 1 and self.agent_pos[0] < self.grid_size - 1: # down
                self.agent_pos[0] += 1
            elif action == 2 and self.agent_pos[1] < self.grid_size - 1: # right
                self.agent_pos[1] += 1
                
        elif self.action_space.n == 5:
            if action == 1 and self.agent_pos[0] < self.grid_size - 1: # down
                self.agent_pos[0] += 1
            elif action == 2 and self.agent_pos[1] < self.grid_size - 1: # right
                self.agent_pos[1] += 1
            elif action == 3 and self.agent_pos[0] > 0: # up
                self.agent_pos[0] -= 1
            elif action == 4 and self.agent_pos[1] > 0: # left
                self.agent_pos[1] -= 1
        
        reward = self._get_reward(action, prev_pos)
    
        terminated = self.current_step >= self.max_steps
        obs = self._get_obs()

        # track previous positions for oscillation detection
        self.prev_prev_pos = self.prev_pos.copy() if self.prev_pos is not None else None
        self.prev_pos = prev_pos.copy()     # store for next step

        return obs, reward, terminated, False, {}

    def _get_reward(self, action, prev_pos): 
        # ============= Oscilation and out of bounds REWARD LOGIC =============
        # # oscillation = agent returns to the previous position (A→B→A)
        # if self.prev_pos is not None and self.agent_pos == self.prev_pos:
        #     return -1.0
        # out of bounds move attempted
        attempted_move_but_blocked = (action != 0) and (prev_pos == self.agent_pos)
        if attempted_move_but_blocked:
            #print("Out of bounds move attempted") # DEBUGGING
            return -0.5  # penalty for trying to move out of bounds
        
        # 1. Immediate Loop (Hitting Wall or trying to stay without Action 0)
        # Result: A -> A
        if self.prev_pos is not None and self.agent_pos == self.prev_pos:
            return -5.0 # Increase this! Hitting walls is bad.
            
        # 2. 2-Step Oscillation (The "Pacing" Fix)
        # Result: A -> B -> A
        # We check if the current position is the same as where we were 2 steps ago
        if self.prev_prev_pos is not None and self.agent_pos == self.prev_prev_pos:
            return -3.0 # Strong penalty for going back to where you just came from
            
        # out of bounds move attempted check (redundant with #1 but keeps logic safe)
        attempted_move_but_blocked = (action != 0) and (prev_pos == self.agent_pos)
        if attempted_move_but_blocked:
            return -5.0
        # ============= Oscilation and out of bounds REWARD LOGIC =============
        
        # ============= INSIDE COMPUTATION =============
        # Previous:
        r0_prev = prev_pos[0] * self.block_size
        c0_prev = prev_pos[1] * self.block_size
        patch_mask_prev = self.mask[r0_prev:r0_prev+self.block_size, c0_prev:c0_prev+self.block_size]
        tumor_count_prev = np.sum(np.isin(patch_mask_prev, [1, 4]))
        inside_prev = tumor_count_prev > 0
        
        # Current:
        r0 = self.agent_pos[0] * self.block_size
        c0 = self.agent_pos[1] * self.block_size
        patch_mask = self.mask[r0:r0+self.block_size, c0:c0+self.block_size]
        tumor_count_curr = np.sum(np.isin(patch_mask, [1, 4]))
        inside = tumor_count_curr > 0 
        # ============= INSIDE COMPUTATION =============
            
        # [action STAY on tumor, action STAY off tumor, moving into tumor, moving out when was on tumor, movement cost inside tumor, movement cost outside tumor]
        if action == 0:
            if inside:
                return self.rewards[0]  # reward for staying on tumor
            else:
                return self.rewards[1]  # penalty for staying off tumor
        else:
            if inside and not inside_prev:
                return self.rewards[2] # reward for moving into tumor
            elif not inside and inside_prev:
                return self.rewards[3] # penalty for moving out of tumor
            elif inside and inside_prev:
                return self.rewards[4] # movement cost inside tumor
            elif not inside and not inside_prev:
                # ============= DISTANCE COMPUTATION =============
                tumor_indices = np.argwhere(self.mask > 0)
                tumor_centroid = np.mean(tumor_indices, axis=0)

                # 2. Compute agent centers before/after
                agent_center = np.array([self.agent_pos[0] * self.block_size + self.block_size / 2, self.agent_pos[1] * self.block_size + self.block_size / 2])
                prev_agent_center = np.array([prev_pos[0] * self.block_size + self.block_size / 2, prev_pos[1] * self.block_size + self.block_size / 2])

                new_dist = np.linalg.norm(agent_center - tumor_centroid)
                prev_dist = np.linalg.norm(prev_agent_center - tumor_centroid)

                # 3. Distance shaping
                distance_reward = (prev_dist - new_dist) * 0.02  # IMPORTANT: slightly stronger
                # ============= DISTANCE COMPUTATION =============
                return self.rewards[5] + distance_reward # movement cost outside tumor
            
    def _get_obs(self):
        r0 = self.agent_pos[0] * self.block_size
        c0 = self.agent_pos[1] * self.block_size
        
        # Extract image patch
        patch = self.image[r0:r0+self.block_size, c0:c0+self.block_size].astype(np.float32)
        
        if self.dataset == 'polyp':
            size = self.block_size
            patch_resized = cv2.resize(patch, (size, size), interpolation=cv2.INTER_LINEAR)
            
            if patch_resized.ndim == 2:
                # If grayscale, convert to 3-channel by stacking
                patch_resized = patch_resized[..., None]  
            patch = np.transpose(patch_resized, (2, 0, 1)) # patch, (2, 0, 1))

            # Create position encoding channels (normalized to [0, 1]) (AND SINGLE CHANNEL)
            pos_row = np.full((1, self.block_size, self.block_size), 
                            self.agent_pos[0] / (self.grid_size - 1))
            pos_col = np.full((1, self.block_size, self.block_size), 
                            self.agent_pos[1] / (self.grid_size - 1))
        
            obs=np.concatenate([patch, pos_row, pos_col], axis=0)
            return obs.astype(np.float32)
        
        if self.dataset == 'glio':
            # Stack into (3, H, W) format            
            pos_row = np.full_like(patch, self.agent_pos[0] / (self.grid_size - 1))
            pos_col = np.full_like(patch, self.agent_pos[1] / (self.grid_size - 1))
            
            # Stack into (3, H, W) format
            obs = np.stack([patch, pos_row, pos_col], axis=0)
            
            return obs

    '''
    Polyp Path: patch(3,60,60) + row(1,60,60) + col(1,60,60) --- concat ---> (5, 60, 60). ✅

    Glio Path: patch(60,60) + row(60,60) + col(60,60) --- stack ---> (3, 60, 60). ✅
    '''
        

    def render(self, show=True):
        if self.render_mode != "human": # would be rgb_array or ansi
            return  # Only render in human mode

        if self.image.ndim == 2:
            # Create RGB visualization image
            # not necessary since it's grayscale, but i want to draw the mask and position
            vis_img = np.stack([self.image] * 3, axis=-1).astype(np.float32)
        else:
            vis_img = self.image.copy().astype(np.float32)

        # Overlay tumor mask in red [..., 0] 
        tumor_overlay = np.zeros_like(vis_img) # do all blank but here we have 3 channels, mask is 2D
        tumor_overlay[..., 0] = (self.mask > 0).astype(float) # red channel. set to float to avoid issues when blending in vis_img

        # transparency overlay (crec que es el mateix valor que tinc a l'altra notebook)
        alpha = 0.4
        vis_img = (1 - alpha) * vis_img + alpha * tumor_overlay

        if show:
            # Plotting
            fig, ax = plt.subplots(figsize=(3, 3))
            ax.imshow(vis_img, cmap='gray', origin='upper')

            # Draw grid lines
            # alpha for transparency again
            for i in range(1, self.grid_size):
                ax.axhline(i * self.block_size, color='white', lw=1, alpha=0.5)
                ax.axvline(i * self.block_size, color='white', lw=1, alpha=0.5)

            # Draw agent position
            r0 = self.agent_pos[0] * self.block_size
            c0 = self.agent_pos[1] * self.block_size
            rect = patches.Rectangle(
                (c0, r0), # (x,y) bottom left corner
                self.block_size, # width
                self.block_size, # height
                linewidth=2,
                edgecolor='yellow',
                facecolor='none'
            )
            ax.add_patch(rect)

            ax.set_title(f"Agent at {self.agent_pos} | Step {self.current_step}/{self.max_steps}")
            ax.axis('off')
            plt.show()
            return None
        else: #just return without showing but draw the agent position
            rgb_array = (vis_img * 255).astype(np.uint8)
        
            # Draw grid lines directly on the array
            for i in range(1, self.grid_size):
                # Horizontal line
                y = i * self.block_size
                rgb_array[y-1:y+1, :] = [255, 255, 255]  # White line
                
                # Vertical line  
                x = i * self.block_size
                rgb_array[:, x-1:x+1] = [255, 255, 255]  # White line
            
            # Draw agent position as a yellow rectangle
            r0 = self.agent_pos[0] * self.block_size
            c0 = self.agent_pos[1] * self.block_size
            
            # Draw rectangle borders (yellow)
            rgb_array[r0:r0+2, c0:c0+self.block_size] = [255, 255, 0]  # Top border
            rgb_array[r0+self.block_size-2:r0+self.block_size, c0:c0+self.block_size] = [255, 255, 0]  # Bottom border
            rgb_array[r0:r0+self.block_size, c0:c0+2] = [255, 255, 0]  # Left border
            rgb_array[r0:r0+self.block_size, c0+self.block_size-2:c0+self.block_size] = [255, 255, 0]  # Right border
            
            # Add step counter text to the image
            pil_img = Image.fromarray(rgb_array)
            draw = ImageDraw.Draw(pil_img)
            
            # Use default font (you can also load a specific font)
            try:
                font = ImageFont.truetype("arial.ttf", 16)
            except:
                font = ImageFont.load_default()
            
            # Draw step counter in top-left corner
            step_text = f"Step: {self.current_step}/{self.max_steps}"
            draw.text((5, 5), step_text, fill=(255, 255, 0), font=font)  # Yellow text
            
            # Convert back to numpy array
            rgb_array = np.array(pil_img)
            return rgb_array
        
    def current_patch_overlap_with_lesion(self, pos=None): # FALTAAA chat
        """ Returns the number of overlapping lesion pixels between the agent's current patch and the ground-truth mask. If > 0, the agent is correctly over the lesion (TP). """
        if pos is None:
            row, col = self.agent_pos
        else:
            row, col = pos
        patch_h = self.block_size # not grid_size because grid_size is number of patches per side
        patch_w = self.block_size
        
        y0 = row * patch_h
        y1 = y0 + patch_h
        x0 = col * patch_w
        x1 = x0 + patch_w
        # extract mask region under current patch
        patch_mask = self.mask[y0:y1, x0:x1]
        # count how many pixels of lesion (nonzero)
        overlap = np.sum(patch_mask > 0)
        return overlap

