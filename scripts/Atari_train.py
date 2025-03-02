import gymnasium as gym
import numpy as np
from stable_baselines3 import PPO
from stable_baselines3.common.env_util import make_vec_env
from stable_baselines3.common.torch_layers import BaseFeaturesExtractor
from stable_baselines3.common.vec_env import SubprocVecEnv, VecFrameStack
from stable_baselines3.common.evaluation import evaluate_policy
from stable_baselines3.common.utils import obs_as_tensor
import torch
import torch.nn as nn
from torch.nn import functional as F
from collections import deque
import cv2, wandb, random
from wandb.integration.sb3 import WandbCallback
from stable_baselines3.common.vec_env import VecFrameStack, VecVideoRecorder
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.callbacks import EvalCallback, CallbackList
import argparse
import matplotlib.pyplot as plt
from stable_baselines3.common.callbacks import EvalCallback, CallbackList
import random
import torch
from ocatari.core import OCAtari
from typing import Callable
from regularized_PPO import KLRegularizedPPO

ROWS = int(210)
COLS = int(160)
concept_version = 2

def set_random_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False

def to_one_hot(arr, m):
    arr = arr.astype(int)
    n = len(arr)
    one_hot = np.eye(m)[arr]
    return one_hot.reshape(-1)

def format_number(n):
    if n < 1000:
        return str(int(n))
    elif n < 1000000:
        return f"{n / 1000:.0f}K" if n % 1000 == 0 else f"{n / 1000:.1f}K"
    elif n < 1000000000:
        return f"{n / 1000000:.0f}M" if n % 1000000 == 0 else f"{n / 1000000:.1f}M"

class ImageConceptBoxingEnv(gym.Env):
    def __init__(self, env_name: str, ROWS, COLS, max_frames) -> None:
        super().__init__()
        self.ocatari_env = OCAtari(env_name, mode="ram", hud=False, obs_mode=None)
        self.max_frames = max_frames
        self.ROWS = ROWS
        self.COLS = COLS
        self.reference_list = self._init_ref_vector()
        self.images = deque(maxlen=self.max_frames)
        self.concepts = deque(maxlen=self.max_frames)
        self.concept_version = concept_version
        if concept_version == 1 or concept_version == 2:
            self.task_types = ['classification', 'classification', 'classification', 'classification'] * self.max_frames
            self.num_classes = [210, 210, 210, 210] * self.max_frames
            self.concept_names = [f"{item}_{i}" for i in range(self.max_frames) for item in ['Agent_x', 'Agent_y', 'Enemy_x', 'Enemy_y']]
        elif concept_version == 3:
            self.task_types = ['classification', 'classification', 'classification', 'classification', 'classification', 'classification', 'classification', 'classification'] * self.max_frames
            self.num_classes = [210, 210, 210, 210, 210, 210, 210, 210] * self.max_frames
            self.concept_names = [f"{item}_{i}" for i in range(self.max_frames) for item in ['Agent_x', 'Agent_y', 'Agent_w', 'Agent_h', 'Enemy_x', 'Enemy_y', 'Enemy_w', 'Enemy_h']]
            self.reset()
        else:
            assert False

    @property
    def observation_space(self):
        # width, height, channels = 160, 210, 3
        return gym.spaces.Box(low=0, high=255, shape=(self.max_frames, self.ROWS, self.COLS), dtype=np.uint8)
        # return gym.spaces.Box(low=0, high=255, shape=(len(self.concept_names) * 210,), dtype=np.uint8)

    @property
    def action_space(self):
        return self.ocatari_env.action_space

    def step(self, *args, **kwargs):
        obs, reward, truncated, terminated, info = self.ocatari_env.step(*args, **kwargs)
        assert obs.shape[0] == self.ROWS and obs.shape[1] == self.COLS
        self.images.append(cv2.cvtColor(obs, cv2.COLOR_RGB2GRAY))
        self.concepts.append(self.get_current_concept())
        return np.array(self.images), reward, truncated, terminated, info
        # return to_one_hot(self.get_concept(), 210), reward, truncated, terminated, info

    def reset(self, *args, **kwargs):
        obs, info = self.ocatari_env.reset(*args, **kwargs)
        # self.ocatari_env.detect_objects(self.ocatari_env._objects, self.ocatari_env._env.env.unwrapped.ale.getRAM(), self.ocatari_env.game_name, self.ocatari_env.hud)
        assert obs.shape[0] == self.ROWS and obs.shape[1] == self.COLS
        for i in range(self.max_frames):
            self.images.append(cv2.cvtColor(obs, cv2.COLOR_RGB2GRAY))
            self.concepts.append(self.get_current_concept())
        return np.array(self.images), info
        # return to_one_hot(self.get_concept(), 210), info

    def render(self, *args, **kwargs):
        return self.ocatari_env.render(*args, **kwargs)

    def close(self, *args, **kwargs):
        return self.ocatari_env.close(*args, **kwargs)

    def get_current_concept(self):
        if self.concept_version == 1: # xy
            temp_vector = np.zeros(2 * len(self.reference_list), dtype=np.float32)
            temp_ref_list = self.reference_list.copy()
            for o in self.ocatari_env.objects:
                idx = temp_ref_list.index(o.category)
                start = idx * 2
                flat = [item for sublist in o.h_coords for item in sublist] # h_coords: History of coordinates, i.e. current (x, y) and previous (x, y) position.
                temp_vector[start:start + 2] = flat[:2]
                temp_ref_list[idx] = ""
            for i, d in enumerate(temp_ref_list):
                if d != "":
                    temp_vector[i*2:i*2+2] = [0.0, 0.0]
            return temp_vector
        elif self.concept_version == 2: # xy (center)
            temp_vector = np.zeros(2 * len(self.reference_list), dtype=np.float32)
            temp_ref_list = self.reference_list.copy()
            for o in self.ocatari_env.objects:
                idx = temp_ref_list.index(o.category)
                start = idx * 2
                flat = [item for item in o.center] # h_coords: History of coordinates, i.e. current (x, y) and previous (x, y) position.
                temp_vector[start:start + 2] = flat[:2]
                temp_ref_list[idx] = ""
            for i, d in enumerate(temp_ref_list):
                if d != "":
                    temp_vector[i*2:i*2+2] = [0.0, 0.0]
            return temp_vector
        elif self.concept_version == 3: # xywh -- unfortunately, the width and height are constant so not helpful!!
            temp_vector = np.zeros(4 * len(self.reference_list), dtype=np.float32)
            temp_ref_list = self.reference_list.copy()
            for o in self.ocatari_env.objects:
                idx = temp_ref_list.index(o.category)
                start = idx * 4
                flat = [item for item in o.xy] + [item for item in o.wh] # xywh (center position and width/height)
                temp_vector[start:start + 4] = flat[:4]
                temp_ref_list[idx] = ""
            for i, d in enumerate(temp_ref_list):
                if d != "":
                    temp_vector[i*4:i*4+4] = [0.0, 0.0, 0.0, 0.0]
            return temp_vector

    def get_concept(self):
        return np.array(self.concepts).flatten()

    def _init_ref_vector(self):
        reference_list = []
        obj_counter = {}
        for o in self.ocatari_env.max_objects:
            if o.category not in obj_counter.keys():
                obj_counter[o.category] = 0
            obj_counter[o.category] += 1
        for k in list(obj_counter.keys()):
            reference_list.extend([k for i in range(obj_counter[k])])
        return reference_list

class ImageConceptPongEnv(gym.Env): # Player, Ball, Enemy
    def __init__(self, env_name: str, ROWS, COLS, max_frames) -> None:
        super().__init__()
        self.ocatari_env = OCAtari(env_name, mode="ram", hud=False, obs_mode=None)
        self.max_frames = max_frames
        self.ROWS = ROWS
        self.COLS = COLS
        self.reference_list = self._init_ref_vector()
        self.images = deque(maxlen=self.max_frames)
        self.concepts = deque(maxlen=self.max_frames)
        self.concept_version = concept_version
        self.task_types = ['classification', 'classification', 'classification', 'classification'] * self.max_frames
        self.num_classes = [210, 210, 210, 210] * self.max_frames
        self.concept_names = [f"{item}_{i}" for i in range(self.max_frames) for item in ['Agent_y', 'Ball_x', 'Ball_y', 'Enemy_y']]
        self.reset()

    @property
    def observation_space(self):
        # width, height, channels = 160, 210, 3
        return gym.spaces.Box(low=0, high=255, shape=(self.max_frames, self.ROWS, self.COLS), dtype=np.uint8)
        # return gym.spaces.Box(low=0, high=255, shape=(len(self.concept_names) * 210,), dtype=np.uint8)

    @property
    def action_space(self):
        return self.ocatari_env.action_space

    def step(self, *args, **kwargs):
        obs, reward, truncated, terminated, info = self.ocatari_env.step(*args, **kwargs)
        assert obs.shape[0] == self.ROWS and obs.shape[1] == self.COLS
        self.images.append(cv2.cvtColor(obs, cv2.COLOR_RGB2GRAY))
        self.concepts.append(self.get_current_concept())
        return np.array(self.images), reward, truncated, terminated, info
        # return to_one_hot(self.get_concept(), 210), reward, truncated, terminated, info

    def reset(self, *args, **kwargs):
        obs, info = self.ocatari_env.reset(*args, **kwargs)
        assert obs.shape[0] == self.ROWS and obs.shape[1] == self.COLS
        for i in range(self.max_frames):
            self.images.append(cv2.cvtColor(obs, cv2.COLOR_RGB2GRAY))
            self.concepts.append(self.get_current_concept())
        return np.array(self.images), info
        # return to_one_hot(self.get_concept(), 210), info

    def render(self, *args, **kwargs):
        return self.ocatari_env.render(*args, **kwargs)

    def close(self, *args, **kwargs):
        return self.ocatari_env.close(*args, **kwargs)

    def get_current_concept(self):
        temp_vector = np.zeros(2 * len(self.reference_list), dtype=np.float32)
        temp_ref_list = self.reference_list.copy()
        for o in self.ocatari_env.objects:
            idx = temp_ref_list.index(o.category)
            start = idx * 2
            if self.concept_version == 1: # top-left coordinate
                flat = [item for sublist in o.h_coords for item in sublist][:2] # h_coords: History of coordinates, i.e. current (x, y) and previous (x, y) position.
            elif self.concept_version == 2: # center coordinate
                flat = o.center
                assert len(flat) == 2
            if o.x < 0 or o.y < 0 or o.w < 0 or o.h < 0:
                flat = [0.0, 0.0]
            temp_vector[start:start + 2] = flat
            temp_ref_list[idx] = ""
        for i, d in enumerate(temp_ref_list):
            if d != "":
                temp_vector[i*2:i*2+2] = [0.0, 0.0]
        return [temp_vector[i] for i in [1, 2, 3, 5]]

    def get_concept(self):
        return np.array(self.concepts).flatten()

    def _init_ref_vector(self):
        reference_list = []
        obj_counter = {}
        for o in self.ocatari_env.max_objects:
            if o.category not in obj_counter.keys():
                obj_counter[o.category] = 0
            obj_counter[o.category] += 1
        for k in list(obj_counter.keys()):
            reference_list.extend([k for i in range(obj_counter[k])])
        return reference_list

def make_concept_env(game: str, rank: int, seed: int = 0, env_class = None) -> Callable: # object info input
    def _init() -> gym.Env:
        env = env_class(game, ROWS, COLS, args.max_frames)
        env = Monitor(env)
        env.reset(seed=seed + rank)
        return env

    set_random_seed(seed)
    return _init

def evaluate_run(env, n_episodes=20, intervene=False):
  deterministic = True
  rewards = []
  concept_acc = []
  for _ in range(n_episodes):
      obs, info = env.reset()
      done = False
      truncated = False
      total_reward = 0
      length = 0
      while not done and not truncated:
          if not isinstance(obs, dict):
              vec_obs = obs.reshape(1, *obs.shape)
          else:
              vec_obs = {}
              for key, value in obs.items():
                  vec_obs[key] = value.reshape(1, *value.shape)
          concepts_pred = model.policy.predict_concepts(obs_as_tensor(vec_obs, model.device))
          concepts_gt = obs_as_tensor(env.get_concept().reshape(1, -1), model.device)
          if not intervene:
            action, _states = model.policy.predict(vec_obs, deterministic=deterministic)
          else:
            action, values, log_probs = model.policy(obs_as_tensor(vec_obs, model.device), concepts=concepts_gt)
            action = action.cpu().numpy()
          action = action[0]
          obs, reward, done, truncated, info = env.step(action)
          total_reward += reward
          length += 1
          
          for i in range(0, len(concepts_pred[0]), 2):
            concept_acc.append(float((concepts_pred[0][i]-concepts_gt[0][i])**2+(concepts_pred[0][i+1]-concepts_gt[0][i+1])**2 <= 25))
      rewards.append(total_reward)
  mean_reward = np.mean(rewards)
  mean_concept_acc = np.mean(concept_acc)
  return mean_reward, mean_concept_acc

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Sample code.")

    parser.add_argument('--run_id', type=int, default=-1)
    parser.add_argument('--concept_loss_type', type=str, default="vanilla_freeze")
    parser.add_argument('--game', type=str, default=None)
    parser.add_argument('--accept_rate', type=float, default=1)
    parser.add_argument('--active_learning', action='store_true')
    parser.add_argument('--unlabeled_set_ratio', type=int, default=2)
    parser.add_argument('--model_ensembles', type=int, default=5)
    parser.add_argument('--num_samples', type=int, default=500)
    parser.add_argument('--num_queries', type=int, default=1)
    parser.add_argument('--max_frames', type=int, default=2)
    parser.add_argument('--active_learning_ratio', type=float, default=1)
    parser.add_argument('--active_learning_bsz', type=int, default=20)
    parser.add_argument('--intervene', action='store_true')
    parser.add_argument('--no_share_features_extractor', action='store_true')
    parser.add_argument('--use_gpt4o', action='store_true')


    args = parser.parse_args()

    config = {
        "total_timesteps": int(1.5e7),
        "num_envs": 8,
        "env_name": "Atari",
        "concept_loss_type": "vanilla_freeze" if args.concept_loss_type is None else args.concept_loss_type,
        "con_coef": 0.5,
        "n_steps": 128,
        "n_epochs": 4,
        "batch_size": 128*2,
        "ent_coef": 0.01,
        "learning_rate": 3e-4,
        "vf_coef": 0.5,
        "share_features_extractor": not args.no_share_features_extractor,
        "active_learning_ratio": args.active_learning_ratio,
        "active_learning_bsz": args.active_learning_bsz,
        "intervention": args.intervene,
        "seed": 0,
        "accept_rate": args.accept_rate,
        "active_learning": args.active_learning,
        "unlabeled_set_ratio": args.unlabeled_set_ratio,
        "model_ensembles": args.model_ensembles,
        "hashing": True,
        "use_v9": True,
        "gpt4o": args.use_gpt4o,
        "max_frames": args.max_frames,
    }

    num_samples = args.num_samples
    num_queries = args.num_queries

    if config["concept_loss_type"] == "concept_input":
        config["run_name"] = f"concept_input-{format_number(config['total_timesteps'])}"
    
    elif config["concept_loss_type"] == "early_query":
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
        config["run_name"] = str(num_samples) + "labels-" + str(num_queries) + "iters" + "-v9"

        if config["share_features_extractor"]:
            config["run_name"] += "-shared"
        else:
            config["run_name"] += "-unshared"

        if config["accept_rate"] < 1:
            config["run_name"] += '-' + str(config["accept_rate"]) + 'accept'

        if config["active_learning"]:
            config["run_name"] += f'-active({args.unlabeled_set_ratio},{args.model_ensembles},{args.active_learning_bsz})'
            if config["active_learning_ratio"] != 1:
                config["run_name"] += f",ratio{config['active_learning_ratio']}"

        if config["intervention"]:
            config["run_name"] += "-intervene"
        config["run_name"] += "-" + format_number(config['total_timesteps'])
    
    elif config["concept_loss_type"] == "finetune_policy":
        exp_name = "LICORICE"
        expert_model_path = "models/xxxxxxxx/model.zip"
        # revert arguments to default
        config["accept_rate"] = 1
        config["active_learning"] = False
        num_samples = 500
        num_queries = 1
        config["run_name"] = f"finetune_policy({exp_name})-{format_number(config['total_timesteps'])}"
    else:
        assert False

    if config["gpt4o"]:
        print('ATTENTION!!!!!!!! Using GPT-4o')
        print('ATTENTION!!!!!!!! Using GPT-4o')
        print('ATTENTION!!!!!!!! Using GPT-4o')
        config["run_name"] += "-gpt4o-" + str(ROWS)
    
    config["run_name"] += "-" + args.game
    if concept_version != 1:
        config["run_name"] += f"-v{concept_version}_concept"
    config["run_name"] += "-classification-fixed"

    print(config)
    if args.run_id == -1:
        pass
    else:
        seed_list = [123, 456, 789, 1011, 1213, 1415]
        config["seed"] = seed_list[args.run_id]
        run = wandb.init(
            project="concept-RL-Atari",
            group=config["run_name"],
            name=config["run_name"]+'-'+str(config["seed"]),
            config=config,
            sync_tensorboard=True,  # auto-upload sb3's tensorboard metrics
            monitor_gym=True,  # auto-upload the videos of agents playing the game
            save_code=True,  # optional
        )

    # create environments & set the seed
    set_random_seed(config["seed"])
    env_id = "ALE/" + args.game +"-v5"
    if args.game == "Boxing":
        env = SubprocVecEnv([make_concept_env(game=env_id, rank=i, seed=config["seed"], env_class=ImageConceptBoxingEnv) for i in range(config["num_envs"])])
        non_vectorized_env = ImageConceptBoxingEnv(env_id, ROWS, COLS, args.max_frames)
        non_vectorized_env.reset(seed=config["seed"])
    elif args.game == "Pong":
        env = SubprocVecEnv([make_concept_env(game=env_id, rank=i, seed=config["seed"], env_class=ImageConceptPongEnv) for i in range(config["num_envs"])])
        non_vectorized_env = ImageConceptPongEnv(env_id, ROWS, COLS, args.max_frames)
        non_vectorized_env.reset(seed=config["seed"])

    gpt4o_prompt = None
    gpt4o_checker = None
    gpt4o_path = None
    if config["gpt4o"]:
        from gpt4o_checker_Boxing import concept_str_to_list
        from gpt4o import prompt_boxing
        gpt4o_prompt = prompt_boxing
        gpt4o_checker = concept_str_to_list
        gpt4o_path = f"gpt_queries/{run.id}"

    PPO_class = PPO
    kwargs = {}
    if config["concept_loss_type"] == "finetune_policy":
        expert_model = PPO.load(expert_model_path)
        PPO_class = KLRegularizedPPO
        kwargs["anchor_policy"] = expert_model.policy
    model = PPO_class(
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
            "net_arch": dict(pi=[64, 64], vf=[64, 64]),
            "use_embedding": True,
        },
        concept_loss_type=config["concept_loss_type"],
        con_coef=config["con_coef"],
        intervention=config["intervention"],
        non_vectorized_env=non_vectorized_env,
        num_samples=num_samples,
        accept_rate=config["accept_rate"],
        active_learning=config["active_learning"],
        unlabeled_set_ratio=config["unlabeled_set_ratio"],
        active_learning_ratio=config["active_learning_ratio"],
        active_learning_bsz=config["active_learning_bsz"],
        model_ensembles=config["model_ensembles"],
        hashing=config["hashing"],
        use_v9=config["use_v9"],
        gpt4o=config["gpt4o"],
        gpt4o_prompt=gpt4o_prompt,
        gpt4o_checker=gpt4o_checker,
        gpt4o_path=gpt4o_path,
        **kwargs,
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
            callback=cb_list,
        )

    if config["concept_loss_type"] != "concept_input" and config["concept_loss_type"] != "no_concept":
        non_vectorized_env.reset(seed=42)
        mean_reward, mean_concept_acc = evaluate_run(non_vectorized_env, n_episodes=20, intervene=config["intervention"])
        wandb.log({"final/mean_reward": mean_reward, "final/mean_concept_acc": mean_concept_acc})
    else:
        non_vectorized_env.reset(seed=42)
        mean_reward, std_reward = evaluate_policy(model, non_vectorized_env, n_eval_episodes=20)
        wandb.log({"final/mean_reward": mean_reward, "final/std_reward": std_reward})

    if args.run_id != -1:
        model.save(f"models/{run.id}/model.zip")

    env.close()
    wandb.finish()
