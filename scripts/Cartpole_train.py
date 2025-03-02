import gymnasium as gym
import numpy as np
from stable_baselines3 import PPO
from stable_baselines3.common.env_util import make_vec_env
from stable_baselines3.common.torch_layers import BaseFeaturesExtractor
import torch
import torch.nn as nn
from torch.nn import functional as F
from collections import deque
import cv2, wandb, random
from wandb.integration.sb3 import WandbCallback
from stable_baselines3.common.vec_env import VecFrameStack, VecVideoRecorder
from stable_baselines3.common.callbacks import EvalCallback, CallbackList
import argparse
import matplotlib.pyplot as plt
from stable_baselines3.common.callbacks import EvalCallback, CallbackList
import random
import torch

def set_random_seed(seed):
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

class VisionCartPoleEnv(gym.Wrapper):
    def __init__(self, env, ROWS, COLS, img_stack=4):
        super(VisionCartPoleEnv, self).__init__(env)
        self.env = env
        self.img_stack = img_stack
        self.ROWS = ROWS
        self.COLS = COLS
        self.observation_space = gym.spaces.Dict({
            'images': gym.spaces.Box(low=0, high=255, shape=(img_stack, self.ROWS, self.COLS), dtype=np.uint8),
            'last_action': gym.spaces.Box(low=np.array([0]), high=np.array([1]), dtype=np.uint8)
        })
        self.frames = deque(maxlen=img_stack)
        self.last_action = None
        self.current_concept = None
        self.task_types = ['regression', 'regression', 'regression', 'regression']
        self.num_classes = [0, 0, 0, 0]
        self.concept_names = ['Cart_Position', 'Cart_Velocity', 'Pole_Angle', 'Pole_Angular_Velocity']
        self.reset()

    def get_concept(self):
        return self.current_concept

    def reset(self, **kwargs):
        observation, info = self.env.reset(**kwargs)
        self.frames.clear()
        img = self.get_image()
        for _ in range(self.img_stack):
            self.frames.append(img)
        self.last_action = 0
        self.current_concept = observation
        return {'images': np.array(self.frames), 'last_action': np.array([self.last_action])}, info

    def step(self, action):
        observation, reward, done, truncated, info = self.env.step(action)
        img = self.get_image()
        self.frames.append(img)
        self.last_action = action
        self.current_concept = observation
        obs = {'images': np.array(self.frames), 'last_action': np.array([self.last_action])}
        if done:
            info['terminal_observation'] = obs
        return obs, reward, done, truncated, info

    def get_image(self):
        img = self.env.render()
        assert img is not None
        img = cv2.resize(img, (self.COLS, self.ROWS))
        img = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
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
        "total_timesteps": int(4e6),
        "num_envs": 8,
        "env_name": "CartPole",
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
        "seed": 0,
        "accept_rate": args.accept_rate,
        "active_learning": args.active_learning,
        "unlabeled_set_ratio": args.unlabeled_set_ratio,
        "model_ensembles": args.model_ensembles,
        "hashing": False,
        "use_v9": True,
        "gpt4o": False,
    }

    num_samples = args.num_samples
    num_queries = args.num_queries

    if config["concept_loss_type"] == "early_query":
        num_queries = 1
        config["accept_rate"] = 1
        config["active_learning"] = False
        config["run_name"] = f"early_query-{num_samples}labels-{format_number(config['total_timesteps'])}"
        config["concept_loss_type"] = "vanilla_freeze"

    elif config["concept_loss_type"] == "uncertainty_based_query":    
        num_queries = 1
        config["accept_rate"] = 1
        config["active_learning"] = True
        config["run_name"] = f"uncertainty_based_query-{num_samples}labels-{format_number(config['total_timesteps'])}"
        config["concept_loss_type"] = "vanilla_freeze"

    elif config["concept_loss_type"] == "random_query":
        num_queries = 1
        assert config["accept_rate"] < 1 # use random
        config["active_learning"] = False
        config["run_name"] = f"random_query-{num_samples}labels-{config['accept_rate']}-{format_number(config['total_timesteps'])}"
        config["concept_loss_type"] = "vanilla_freeze"

    elif config["concept_loss_type"] == "no_concept":
        config["run_name"] = f"vanilla-raw_arch-0505-{format_number(config['total_timesteps'])}"

    elif config["concept_loss_type"] == "joint":
        config["run_name"] = f"joint-{format_number(config['total_timesteps'])}"

    elif config["concept_loss_type"] == "vanilla_freeze":
        config["run_name"] = str(num_samples) + "total_labels-" + str(num_queries) + "iterations" + "-v9"

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
        config["run_name"] += "-" + format_number(config['total_timesteps'])
    else:
        assert False

    ROWS = int(160)
    COLS = int(240)

    if config["gpt4o"]:
        print('ATTENTION!!!!!!!! Using GPT-4o')
        print('ATTENTION!!!!!!!! Using GPT-4o')
        print('ATTENTION!!!!!!!! Using GPT-4o')
        config["run_name"] += "-gpt4o-" + str(ROWS)

        # previous ones have different order -- so let's follow that practice
        parts = config["run_name"].split('-')
        index_4M = parts.index('4M')
        index_v9 = parts.index('v9')
        parts.insert(index_v9 + 1, parts.pop(index_4M))
        config["run_name"] = '-'.join(parts)

    print(config)
    if args.run_id == -1:
        pass
    else:
        seed_list = [123, 456, 789, 1011, 1213, 1415]
        config["seed"] = seed_list[args.run_id]
        run = wandb.init(
            project="concept-RL",
            group=config["run_name"],
            name=config["run_name"]+'-'+str(config["seed"]),
            config=config,
            sync_tensorboard=True,  # auto-upload sb3's tensorboard metrics
            monitor_gym=True,  # auto-upload the videos of agents playing the game
            save_code=True,  # optional
        )

    # create environments & set the seed
    set_random_seed(config["seed"])
    env_id = "CartPole-v1"
    env = make_vec_env(env_id, n_envs=config["num_envs"], wrapper_class=VisionCartPoleEnv, wrapper_kwargs={"ROWS": ROWS, "COLS": COLS})
    non_vectorized_env = gym.make("CartPole-v1", render_mode="rgb_array")
    non_vectorized_env = VisionCartPoleEnv(non_vectorized_env, ROWS, COLS)
    non_vectorized_env.reset(seed=config["seed"])

    gpt4o_prompt = None
    gpt4o_checker = None
    gpt4o_path = None
    if config["gpt4o"]:
        from gpt4o_checker_CartPole import concept_str_to_list
        from gpt4o import prompt_cart_pole
        gpt4o_prompt = prompt_cart_pole
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
        hashing=config["hashing"],
        use_v9=config["use_v9"],
        gpt4o=config["gpt4o"],
        gpt4o_prompt=gpt4o_prompt,
        gpt4o_checker=gpt4o_checker,
        gpt4o_path=gpt4o_path,
    )

    if args.run_id != -1:
        cb_list = WandbCallback()
    else:
        cb_list = None

    if config["concept_loss_type"] == "vanilla_freeze":
        model.learn(
            total_timesteps=config["total_timesteps"],
            use_v9=config["use_v9"],
            query_num_times=num_queries,
            query_labels_per_time=num_samples // num_queries,
            callback=cb_list,
        )
    else:
        model.learn(
            total_timesteps=config["total_timesteps"],
            use_v9=config["use_v9"],
            callback=cb_list,
        )
    if args.run_id != -1:
        model.save(f"models/{run.id}/model.zip")
    env.close()
    wandb.finish()
