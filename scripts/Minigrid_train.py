import gymnasium as gym
import numpy as np
from stable_baselines3 import PPO
from stable_baselines3.common.env_util import make_vec_env
from stable_baselines3.common.torch_layers import BaseFeaturesExtractor
from stable_baselines3.common.evaluation import evaluate_policy
import torch.nn as nn
from collections import deque
import cv2, wandb
from wandb.integration.sb3 import WandbCallback
from stable_baselines3.common.vec_env import VecFrameStack, VecVideoRecorder
from minigrid.wrappers import ImgObsWrapper
import minigrid
from minigrid.core.grid import Grid
from minigrid.minigrid_env import MiniGridEnv
from minigrid.core.constants import COLOR_NAMES, DIR_TO_VEC, TILE_PIXELS, COLOR_TO_IDX, OBJECT_TO_IDX
import argparse
import matplotlib.pyplot as plt
from stable_baselines3.common.callbacks import EvalCallback, CallbackList
import random
import torch

from gymnasium.envs.registration import register
from custom_dynamic_obstacles_env import CustomDynamicObstaclesEnv

def set_random_seed(seed):
    assert torch.cuda.is_available()
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False

def format_number(n):
    if n < 1000:
        return str(int(n))
    elif n < 1000000:
        return f"{n / 1000:.0f}K" if n % 1000 == 0 else f"{n / 1000:.1f}K"
    elif n < 1000000000:
        return f"{n / 1000000:.0f}M" if n % 1000000 == 0 else f"{n / 1000000:.1f}M"

def create_one_hot_vector(val, num_classes):
    assert len(val) == len(num_classes)
    total_length = sum(num_classes)
    
    one_hot_vector = np.zeros(total_length, dtype=int)
    start_idx = 0
    
    for value, n_class in zip(val, num_classes):
        if value >= n_class:
            raise ValueError("Value in 'val' must be less than corresponding 'num_classes'")
        one_hot_vector[start_idx + value] = 1
        start_idx += n_class
    
    return one_hot_vector

class DynamicObstaclesEnvWrapper(gym.Wrapper):
    """
    Wraps the DynamicObstacles environment.
    """
    def __init__(self, env, ROWS, COLS, concept_version=1):
        super(DynamicObstaclesEnvWrapper, self).__init__(env)
        self.env = env
        self.ROWS = ROWS
        self.COLS = COLS
        self.observation_space = gym.spaces.Box(low=0, high=255, shape=(3, self.ROWS, self.COLS), dtype=np.uint8)
        self.concept_version = concept_version
        if concept_version == 1:
            self.task_types = ['classification'] * 5
            self.num_classes = [2, 2, 4, 2, 2]
            self.concept_names = ['agent_at_right', 'agent_at_bottom', 'agent_direction', 'obstacle_right', 'obstacle_below']
        elif concept_version == 2:
            grid = self.env.unwrapped.grid
            self.task_types = ['classification'] * ((grid.width-2) * (grid.height-2))
            self.num_classes = [len(OBJECT_TO_IDX)] * ((grid.width-2) * (grid.height-2) - 1) + [4]
            self.concept_names = [f'cell_{i}_{j}_type' for i in range(1, grid.width-1) for j in range(1, grid.height-1)]
            self.concept_names[-1] = 'agent_direction'
        elif concept_version == 3:
            if self.env.unwrapped.grid.width == 5:
                self.task_types = ['classification'] * 11
                self.num_classes = [6, 6, 4, 6, 6, 6, 6, 2, 2, 2, 2]
                self.concept_names = [
                    'agent_position_x', 'agent_position_y', 'agent_direction', 
                    'obstacle1_position_x', 'obstacle1_position_y', 
                    'obstacle2_position_x', 'obstacle2_position_y', 
                    'movable_right', 'movable_down', 'movable_left', 'movable_up'
                ]
            elif self.env.unwrapped.grid.width == 6:
                self.task_types = ['classification'] * 13
                self.num_classes = [6, 6, 4, 6, 6, 6, 6, 6, 6, 2, 2, 2, 2]
                self.concept_names = [
                    'agent_position_x', 'agent_position_y', 'agent_direction', 
                    'obstacle1_position_x', 'obstacle1_position_y', 
                    'obstacle2_position_x', 'obstacle2_position_y', 
                    'obstacle3_position_x', 'obstacle3_position_y', 
                    'movable_right', 'movable_down', 'movable_left', 'movable_up'
                ]
        self.reset()

    def get_concept(self):
        """
        Returns high-level concepts about the environment state.
        """
        if self.concept_version == 1:
            return self._get_concept_v1()
        elif self.concept_version == 2:
            return self._get_concept_v2()
        elif self.concept_version == 3:
            return self._get_concept_v3()

    def _get_concept_v1(self):
        """
        Returns high-level concepts about the environment state:
        - agent_at_right: boolean, true if agent is in the rightmost column
        - agent_at_bottom: boolean, true if agent is at the bottom row
        - agent_direction: int (0: right, 1: down, 2: left, 3: up)
        - obstacle_right: boolean, true if there's an obstacle directly to the right of the agent
        - obstacle_below: boolean, true if there's an obstacle directly below the agent
        """
        agent_pos = self.env.unwrapped.agent_pos
        agent_dir = self.env.unwrapped.agent_dir
        grid = self.env.unwrapped.grid

        # Check if the agent is at the bottom row or in the rightmost column
        agent_at_right = (agent_pos[0] == grid.width-1 - 1)
        agent_at_bottom = (agent_pos[1] == grid.height-1 - 1)

        # Check for Ball obstacles below and to the right of the agent
        cell_right = grid.get(agent_pos[0] + 1, agent_pos[1]) if agent_pos[0] + 1 < grid.width-1 else None
        obstacle_right = (cell_right != None and cell_right.type == 'ball') # Treats out-of-bounds not as an obstacle

        cell_below = grid.get(agent_pos[0], agent_pos[1] + 1) if agent_pos[1] + 1 < grid.height-1 else None
        obstacle_below = (cell_below != None and cell_below.type == 'ball') # Treats out-of-bounds not as an obstacle
        
        # obstacle_right = not self.can_move(agent_pos, 0, grid)
        # obstacle_below = not self.can_move(agent_pos, 1, grid)

        numbers = [
            int(agent_at_right),
            int(agent_at_bottom),
            agent_dir,
            int(obstacle_right),
            int(obstacle_below),
        ]

        return np.array(numbers, dtype=np.float32)

    def _get_concept_v2(self):
        """
        Version 2 concepts: Each cell's type in the entire grid (except the goal), plus agent's direction.
        """
        agent_dir = self.env.unwrapped.agent_dir
        grid = self.env.unwrapped.grid

        concepts = []
        for i in range(1, grid.width-1):
            for j in range(1, grid.height-1):
                cell = grid.get(i, j)
                cell_type = OBJECT_TO_IDX[cell.type] if cell is not None else OBJECT_TO_IDX['empty']
                if cell_type == OBJECT_TO_IDX['empty'] and self.env.unwrapped.agent_pos == (i, j):
                    cell_type = OBJECT_TO_IDX['agent']
                concepts.append(cell_type)
        concepts[-1] = agent_dir  # Adding the agent's direction as the last concept

        return np.array(concepts, dtype=np.float32)

    def _get_concept_v3(self):
        """
        Returns high-level concepts about the environment state for concept version 3:
        - agent_position: tuple (x, y)
        - agent_direction: int (0: right, 1: down, 2: left, 3: up)
        - obstacle1_position: tuple (x, y) of the first blue ball
        - obstacle2_position: tuple (x, y) of the second blue ball
        - direction_movable: boolean values for 'right', 'down', 'left', 'up'
        """
        agent_pos = self.env.unwrapped.agent_pos
        agent_dir = self.env.unwrapped.agent_dir
        grid = self.env.unwrapped.grid

        def get_cell(x, y):
            if x < 0 or x >= grid.width or y < 0 or y >= grid.height:
                return None
            return grid.get(x, y)

        def is_movable(x, y):
            cell = get_cell(x, y)
            return cell is None or cell.type == 'empty'

        # Agent Position
        agent_position = (agent_pos[0], agent_pos[1])

        # Agent Direction
        agent_direction = agent_dir

        # Obstacle Positions
        obstacle_positions = []
        for x in range(grid.width):
            for y in range(grid.height):
                cell = get_cell(x, y)
                if cell and cell.type == 'ball':
                    obstacle_positions.append((x, y))

        obstacle_positions = sorted(obstacle_positions)
        assert len(obstacle_positions) == 2 or len(obstacle_positions) == 3

        # Check direction_movable in all four directions
        direction_movable = {
            'right': self.can_move(agent_pos, 0, grid),
            'down': self.can_move(agent_pos, 1, grid),
            'left': self.can_move(agent_pos, 2, grid),
            'up': self.can_move(agent_pos, 3, grid),
        }

        numbers = [
            agent_position[0], agent_position[1],
            agent_direction,
        ]
        for i in range(len(obstacle_positions)):
            numbers.extend([
                obstacle_positions[i][0], obstacle_positions[i][1]
            ])
        numbers.extend([
            int(direction_movable['right']),
            int(direction_movable['down']),
            int(direction_movable['left']),
            int(direction_movable['up'])
        ])

        return np.array(numbers, dtype=np.float32)

    def can_move(self, position, direction, grid):
        """
        Helper function to determine if movement in a specific direction is possible.
        """
        next_pos = DIR_TO_VEC[direction]
        if 1 <= position[0] + next_pos[0] < grid.width-1 and 1 <= position[1] + next_pos[1] < grid.height-1:
            next_cell = grid.get(position[0] + next_pos[0], position[1] + next_pos[1])
            return next_cell is None or next_cell.can_overlap()
        else:
            return False  # Out of bounds

    def reset(self, **kwargs):
        observation, info = self.env.reset(**kwargs)
        img = self.get_image()
        obs = img
        # obs = self.get_concept()
        return obs, info

    def step(self, action):
        observation, reward, done, truncated, info = self.env.step(action)
        img = self.get_image()
        obs = img
        # obs = self.get_concept()
        if done:
            info['terminal_observation'] = obs
        return obs, reward, done, truncated, info

    def get_image(self):
        img = self.env.render()
        img = cv2.resize(img, (self.COLS, self.ROWS))
        img = img.transpose((2, 0, 1))
        return img

class DoorKeyEnvWrapper(gym.Wrapper):
    """
    Wraps the DoorKey environment.
    """
    def __init__(self, env, ROWS, COLS):
        super(DoorKeyEnvWrapper, self).__init__(env)
        self.env = env
        self.ROWS = ROWS
        self.COLS = COLS
        self.observation_space = gym.spaces.Box(low=0, high=255, shape=(3, self.ROWS, self.COLS), dtype=np.uint8)
        # self.observation_space = gym.spaces.Box(low=0, high=6, shape=(12,), dtype=np.uint8)
        self.task_types = ['classification']*12
        self.num_classes = [8, 8, 4, 8, 8, 8, 8, 2, 2, 2, 2, 2]  # Additional classes for key and door
        self.concept_names = ['agent_position_x', 'agent_position_y', 'agent_direction', 'key_position_x', 'key_position_y', 'door_position_x', 'door_position_y', 'door_open', 'direction_movable_right', 'direction_movable_down', 'direction_movable_left', 'direction_movable_up']
        self.reset()

    def get_concept(self):
        """
        Returns high-level concepts about the environment state:
        - agent_position: tuple (x, y)
        - agent_direction: int (0: right, 1: down, 2: left, 3: up)
        - key_position: tuple (x, y) of the key
        - door_position: tuple (x, y) of the door
        - door_open: boolean value, if the door is open
        - direction_movable: dictionary with boolean values for 'right', 'down', 'left', 'up'
        """
        agent_pos = self.env.unwrapped.agent_pos
        agent_dir = self.env.unwrapped.agent_dir
        grid = self.env.unwrapped.grid
        key_pos = (0, 0) # default one if not found (carrying)
        door_pos = None
        door_open = False

        # Locate door, key, and goal positions
        for x in range(grid.width):
            for y in range(grid.height):
                cell = grid.get(x, y)
                if cell is not None:
                    if cell.type == 'door':
                        door_pos = (x, y)
                        door_open = cell.is_open  # Check if the door is open
                    elif cell.type == 'key':
                        key_pos = (x, y)

        # Check direction_movable in all four directions
        direction_movable = {
            'right': self.can_move(agent_pos, 0, grid),
            'down': self.can_move(agent_pos, 1, grid),
            'left': self.can_move(agent_pos, 2, grid),
            'up': self.can_move(agent_pos, 3, grid),
        }

        infos = {
            'agent_position': agent_pos,
            'agent_direction': agent_dir,
            'key_position': key_pos,
            'door_position': door_pos,
            'door_open': door_open,  # Add door_open to infos
            'direction_movable': direction_movable
        }

        numbers = []
        for key, value in infos.items():
            if key == 'direction_movable':
                for k, v in value.items():
                    numbers.append(int(v))
            elif isinstance(value, tuple):
                numbers.extend([x for x in value])
            else:
                numbers.append(value)

        return np.array(numbers, dtype=np.float32)

    def can_move(self, position, direction, grid):
        """
        Helper function to determine if movement in a specific direction is possible.
        """
        next_pos = DIR_TO_VEC[direction]
        if 1 <= position[0] + next_pos[0] < grid.width-1 and 1 <= position[1] + next_pos[1] < grid.height-1:
            next_cell = grid.get(position[0] + next_pos[0], position[1] + next_pos[1])
            return next_cell is None or next_cell.can_overlap()
        else:
            return False  # Out of bounds

    def reset(self, **kwargs):
        observation, info = self.env.reset(**kwargs)
        img = self.get_image()
        obs = img
        # obs = self.get_concept()
        return obs, info

    def step(self, action):
        observation, reward, done, truncated, info = self.env.step(action)
        img = self.get_image()
        obs = img
        # obs = self.get_concept()
        if done:
            info['terminal_observation'] = obs
        return obs, reward, done, truncated, info

    def get_image(self):
        img = self.env.render()
        img = cv2.resize(img, (self.COLS, self.ROWS))
        img = img.transpose((2, 0, 1))
        return img

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Sample code.")

    parser.add_argument('--run_id', type=int, default=-1)
    parser.add_argument('--concept_loss_type', type=str, default="vanilla_freeze")
    parser.add_argument('--game', type=str, default=None)
    parser.add_argument('--accept_rate', type=float, default=1)
    parser.add_argument('--active_learning', action='store_true')
    parser.add_argument('--unlabeled_set_ratio', type=int, default=10)
    parser.add_argument('--model_ensembles', type=int, default=5)
    parser.add_argument('--num_samples', type=int, default=500)
    parser.add_argument('--num_queries', type=int, default=1)

    args = parser.parse_args()

    config = {
        "total_timesteps": int(1e6),
        "num_envs": 8,
        "env_name": "Minigrid",
        "concept_loss_type": "vanilla_freeze" if args.concept_loss_type is None else args.concept_loss_type,
        "con_coef": 0.5,
        "n_steps": 256*2,
        "n_epochs": 10,
        "batch_size": 128*2,
        "ent_coef": 0.01,
        "learning_rate": 3e-4,
        "vf_coef": 0.5,
        "share_features_extractor": True,
        "intervention": False,
        "game": args.game, # 'DoorKey'
        "seed": 0,
        "accept_rate": args.accept_rate,
        "active_learning": args.active_learning,
        "unlabeled_set_ratio": args.unlabeled_set_ratio,
        "model_ensembles": args.model_ensembles,
        "gpt4o": False,
        "hashing": True,
        "concept_version": 3 if args.game == 'DynamicObstacles' else 1,
    }

    num_samples = args.num_samples
    num_queries = args.num_queries

    if config['game'] == 'DoorKey': # larger for 7x7
        grid_size = 7
        config["total_timesteps"] = int(4e6)
    elif config['game'] == 'DynamicObstacles':
        grid_size = 5
        config["total_timesteps"] = int(1e6)


    if config["concept_loss_type"] == "early_query":
        num_queries = 1
        config["accept_rate"] = 1
        config["active_learning"] = False
        config["run_name"] = f"early_query-{num_samples}labels-{format_number(config['total_timesteps'])}-{config['game']}"
        if config["game"] == 'DoorKey':
            config["run_name"] += f"-{grid_size}x{grid_size}"
        config["concept_loss_type"] = "vanilla_freeze"

    elif config["concept_loss_type"] == "uncertainty_based_query":    
        num_queries = 1
        config["accept_rate"] = 1
        config["active_learning"] = True
        config["run_name"] = f"uncertainty_based_query-{num_samples}labels-{format_number(config['total_timesteps'])}-{config['game']}"
        if config["game"] == 'DoorKey':
            config["run_name"] += f"-{grid_size}x{grid_size}"
        config["concept_loss_type"] = "vanilla_freeze"

    elif config["concept_loss_type"] == "random_query":
        num_queries = 1
        assert config["accept_rate"] < 1 # use random
        config["active_learning"] = False
        config["run_name"] = f"random_query-{num_samples}labels-{config['accept_rate']}-{format_number(config['total_timesteps'])}-{config['game']}"
        if config["game"] == 'DoorKey':
            config["run_name"] += f"-{grid_size}x{grid_size}"
        config["concept_loss_type"] = "vanilla_freeze"

    elif config["concept_loss_type"] == "no_concept":
        config["run_name"] = f"vanilla-raw_arch-0505-{format_number(config['total_timesteps'])}-{config['game']}"
        if config["game"] == 'DoorKey':
            config["run_name"] += f"-{grid_size}x{grid_size}"

    elif config["concept_loss_type"] == "joint":
        config["run_name"] = f"joint-{format_number(config['total_timesteps'])}-{config['game']}"
        if config["game"] == 'DoorKey':
            config["run_name"] += f"-{grid_size}x{grid_size}"

    elif config["concept_loss_type"] == "vanilla_freeze":
        config["run_name"] = str(num_samples) + "total_labels-" + str(num_queries) + "iterations" + "-v12-" + format_number(config['total_timesteps']) + "-" + config['game']
        if config["game"] == 'DoorKey':
            config["run_name"] += f"-{grid_size}x{grid_size}"

        if config["share_features_extractor"]:
            config["run_name"] += "-shared"
        else:
            config["run_name"] += "-unshared"

        if config["accept_rate"] < 1:
            config["run_name"] += '-' + str(config["accept_rate"]) + 'accept'

        if config["active_learning"]:
            config["run_name"] += f'-active({args.unlabeled_set_ratio},{args.model_ensembles})'

        if config["intervention"]:
            config["run_name"] += "-intervene"

    else:
        assert False

    if config["concept_version"] > 1:
        ver = config["concept_version"]
        config["run_name"] += f"-v{ver}_concept"

    ROWS = int(160)
    COLS = int(160)

    if config["gpt4o"]:
        print('ATTENTION!!!!!!!! Using GPT-4o')
        print('ATTENTION!!!!!!!! Using GPT-4o')
        print('ATTENTION!!!!!!!! Using GPT-4o')
        config["run_name"] += "-gpt4o-" + str(ROWS)

    print(config)
    if args.run_id == -1:
        pass
    else:
        seed_list = [123, 456, 789, 1011, 1213, 1415]
        config["seed"] = seed_list[args.run_id]
        run = wandb.init(
            project=f"concept-RL-Minigrid-{config['game']}",
            group=config["run_name"],
            name=config["run_name"]+'-'+str(config["seed"]),
            config=config,
            sync_tensorboard=True,  # auto-upload sb3's tensorboard metrics
            monitor_gym=True,  # auto-upload the videos of agents playing the game
            save_code=True,  # optional
        )

    # create environments & set the seed
    set_random_seed(config["seed"])
    if config['game'] == 'DynamicObstacles':
        register( # fix the collision with wall bug in the original environment
            id=f'MiniGrid-Custom-Dynamic-Obstacles-{grid_size}x{grid_size}-v0',
            entry_point='custom_dynamic_obstacles_env:CustomDynamicObstaclesEnv',
            kwargs={'size': grid_size, 'n_obstacles': grid_size // 2} # 5 -> 2; 6 -> 3
        )
        env_id = f'MiniGrid-Custom-Dynamic-Obstacles-{grid_size}x{grid_size}-v0'
        env = make_vec_env(env_id, n_envs=config["num_envs"], seed=config["seed"], env_kwargs={'highlight': False}, wrapper_class=DynamicObstaclesEnvWrapper, wrapper_kwargs={"ROWS": ROWS, "COLS": COLS, "concept_version": config["concept_version"]})
        non_vectorized_env = gym.make(env_id, render_mode="rgb_array", highlight=False)
        non_vectorized_env = DynamicObstaclesEnvWrapper(non_vectorized_env, ROWS, COLS, config["concept_version"])
        non_vectorized_env.reset(seed=config["seed"])
    if config['game'] == 'DoorKey':
        register(
            id="MiniGrid-DoorKey-7x7-v0",
            entry_point="minigrid.envs:DoorKeyEnv",
            kwargs={"size": 7},
        )
        env_id = f"MiniGrid-DoorKey-{grid_size}x{grid_size}-v0" # 6 or 7
        env = make_vec_env(env_id, n_envs=config["num_envs"], seed=config["seed"], env_kwargs={'highlight': False}, wrapper_class=DoorKeyEnvWrapper, wrapper_kwargs={"ROWS": ROWS, "COLS": COLS})
        non_vectorized_env = gym.make(env_id, render_mode="rgb_array", highlight=False)
        non_vectorized_env = DoorKeyEnvWrapper(non_vectorized_env, ROWS, COLS)
        non_vectorized_env.reset(seed=config["seed"])

    gpt4o_prompt = None
    gpt4o_checker = None
    gpt4o_path = None
    if config["gpt4o"] and config['game'] == 'DoorKey':
        assert grid_size == 6 or grid_size == 7
        from gpt4o_checker_DoorKey import concept_str_to_list
        from gpt4o import prompt_door_key, prompt_door_key_large
        if grid_size == 6:
            gpt4o_prompt = prompt_door_key
        else:
            gpt4o_prompt = prompt_door_key_large
        gpt4o_checker = concept_str_to_list
        gpt4o_path = f"gpt_queries/{run.id}"
    elif config["gpt4o"] and config['game'] == 'DynamicObstacles' and config["concept_version"] == 1:
        assert grid_size == 5
        from gpt4o_checker_DynamicObstacles import concept_str_to_list
        from gpt4o import prompt_dynamic_obstacles
        gpt4o_prompt = prompt_dynamic_obstacles
        gpt4o_checker = concept_str_to_list
        gpt4o_path = f"gpt_queries/{run.id}"
    elif config["gpt4o"] and config['game'] == 'DynamicObstacles' and config["concept_version"] == 2:
        assert grid_size == 5
        from gpt4o_checker_DynamicObstacles_v2 import concept_str_to_list
        from gpt4o import prompt_dynamic_obstacles_v2
        gpt4o_prompt = prompt_dynamic_obstacles_v2
        gpt4o_checker = concept_str_to_list
        gpt4o_path = f"gpt_queries/{run.id}"
    elif config["gpt4o"] and config['game'] == 'DynamicObstacles' and config["concept_version"] == 3:
        assert grid_size == 5 or grid_size == 6
        from gpt4o_checker_DynamicObstacles_v3 import concept_str_to_list
        from gpt4o import prompt_dynamic_obstacles_v3, prompt_dynamic_obstacles_v3_large
        if grid_size == 5:
            gpt4o_prompt = prompt_dynamic_obstacles_v3
        else:
            gpt4o_prompt = prompt_dynamic_obstacles_v3_large
        gpt4o_checker = concept_str_to_list
        gpt4o_path = f"gpt_queries/{run.id}"

    model = PPO(
        "MultiInputPolicy" if isinstance(non_vectorized_env.observation_space, gym.spaces.Dict) else ("CnnPolicy" if len(non_vectorized_env.observation_space.shape) >= 2 else "MlpPolicy"),
        env,
        n_steps=config["n_steps"],
        n_epochs=config["n_epochs"],
        learning_rate=lambda progression: config["learning_rate"] * progression,
        ent_coef=config["ent_coef"],
        clip_range=lambda progression: 0.1 * progression,
        batch_size=config["batch_size"],
        verbose=1 if args.run_id == -1 else 0, # multiple runs -> silent
        seed=config["seed"],
        tensorboard_log=f"runs",
        policy_kwargs={
            "concept_dim": len(non_vectorized_env.get_concept()),
            "task_types": non_vectorized_env.task_types,
            "num_classes": non_vectorized_env.num_classes,
            "concept_names": non_vectorized_env.concept_names,
            "share_features_extractor": config["share_features_extractor"], 
            "net_arch": dict(pi=[64, 64], vf=[64, 64]) if config["share_features_extractor"]==True else dict(pi=[64, 64], vf=[]), # after the concept layer
            "focal_loss": False,
        },
        concept_loss_type=config["concept_loss_type"],
        con_coef=config["con_coef"],
        intervention=config["intervention"],
        non_vectorized_env=non_vectorized_env,
        num_samples=num_samples,
        accept_rate=config["accept_rate"],
        active_learning=config["active_learning"],
        unlabeled_set_ratio=config["unlabeled_set_ratio"],
        model_ensembles=config["model_ensembles"],
        gpt4o=config["gpt4o"],
        hashing=config["hashing"],
        gpt4o_prompt=gpt4o_prompt,
        gpt4o_checker=gpt4o_checker,
        gpt4o_path=gpt4o_path,
    )

    if args.run_id != -1:
        eval_env = env
        eval_callback = EvalCallback(
            eval_env,
            n_eval_episodes=20,
            best_model_save_path=f"models/{run.id}",
            log_path=f"models/{run.id}",
            eval_freq=(config["total_timesteps"] // 10 // config["num_envs"] - 1),
            deterministic=True,
            render=False)
        cb_list = CallbackList([eval_callback, WandbCallback()])
    else:
        cb_list = None

    if config["concept_loss_type"] == "vanilla_freeze":
        model.learn(
            total_timesteps=config["total_timesteps"],
            query_num_times=num_queries,
            query_labels_per_time=num_samples // num_queries,
            callback=cb_list,
        )
    else:
        model.learn(
            total_timesteps=config["total_timesteps"],
            callback=cb_list,
        )
    if args.run_id != -1:
        model.save(f"models/{run.id}/model.zip")

    mean_reward, std_reward = evaluate_policy(model, eval_env, n_eval_episodes=20)
    wandb.log({"final/mean_reward": mean_reward, "final/std_reward": std_reward})

    env.close()
    wandb.finish()