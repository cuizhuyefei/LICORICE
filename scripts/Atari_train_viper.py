import gymnasium as gym
import numpy as np
from stable_baselines3 import PPO
from stable_baselines3.common.env_util import make_vec_env
from stable_baselines3.common.torch_layers import BaseFeaturesExtractor
from stable_baselines3.common.vec_env import SubprocVecEnv, VecFrameStack, VecTransposeImage
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

# test Boxing env
from collections import deque
from gymnasium import spaces
from stable_baselines3.common.atari_wrappers import AtariWrapper
class NoopResetEnv(gym.Wrapper):
    def __init__(self, env=None, noop_max=30):
        """Sample initial states by taking random number of no-ops on reset.
        No-op is assumed to be action 0.
        """
        super(NoopResetEnv, self).__init__(env)
        self.noop_max = noop_max
        self.override_num_noops = None
        assert env.unwrapped.get_action_meanings()[0] == 'NOOP'

    def reset(self, *args, **kwargs):
        """ Do no-op action for a number of steps in [1, noop_max]."""
        self.env.reset(*args, **kwargs)
        if self.override_num_noops is not None:
            noops = self.override_num_noops
        else:
            noops = np.random.randint(1, self.noop_max + 1)
        assert noops > 0
        obs = None
        for _ in range(noops):
            obs, _, done, terminated, info = self.env.step(0)
            if done:
                obs, info = self.env.reset(*args, **kwargs)
        return obs, info


class FireResetEnv(gym.Wrapper):
    def __init__(self, env=None):
        """For environments where the user need to press FIRE for the game to start."""
        super(FireResetEnv, self).__init__(env)
        assert env.unwrapped.get_action_meanings()[1] == 'FIRE'
        assert len(env.unwrapped.get_action_meanings()) >= 3

    def reset(self, *args, **kwargs):
        self.env.reset(*args, **kwargs)
        obs, _, truncated, terminated, _ = self.env.step(1)
        if truncated or terminated:
            self.env.reset(*args, **kwargs)
        obs, _, truncated, terminated, _ = self.env.step(2)
        if truncated or terminated:
            self.env.reset(*args, **kwargs)
        info = {}
        return obs, info


class EpisodicLifeEnv(gym.Wrapper):
    def __init__(self, env=None):
        """Make end-of-life == end-of-episode, but only reset on true game over.
        Done by DeepMind for the DQN and co. since it helps value estimation.
        """
        super(EpisodicLifeEnv, self).__init__(env)
        self.lives = 0
        self.was_real_done = True
        self.was_real_reset = False

    def step(self, action):
        obs, reward, truncated, terminated, info = self.env.step(action)
        self.was_real_done = truncated or terminated
        # check current lives, make loss of life terminal,
        # then update lives to handle bonus lives
        lives = self.env.unwrapped.ale.lives()
        if lives < self.lives and lives > 0:
            # for Qbert somtimes we stay in lives == 0 condtion for a few frames
            # so its important to keep lives > 0, so that we only reset once
            # the environment advertises done.
            terminated = True
        self.lives = lives
        return obs, reward, truncated, terminated, info

    def reset(self, *args, **kwargs):
        """Reset only when lives are exhausted.
        This way all states are still reachable even though lives are episodic,
        and the learner need not know about any of this behind-the-scenes.
        """
        if self.was_real_done:
            obs, info = self.env.reset(*args, **kwargs)
            self.was_real_reset = True
        else:
            # no-op step to advance from terminal/lost life state
            obs, _, _, _, info = self.env.step(0)
            self.was_real_reset = False
        self.lives = self.env.unwrapped.ale.lives()
        return obs, info


class MaxAndSkipEnv(gym.Wrapper):
    def __init__(self, env=None, skip=4):
        """Return only every `skip`-th frame"""
        super(MaxAndSkipEnv, self).__init__(env)
        # most recent raw observations (for max pooling across time steps)
        self._obs_buffer = deque(maxlen=2)
        self._skip = skip

    def step(self, action):
        total_reward = 0.0
        # done = None
        for _ in range(self._skip):
            obs, reward, truncated, terminated, info = self.env.step(action)
            self._obs_buffer.append(obs)
            total_reward += reward
            if truncated or terminated:
                break

        max_frame = np.max(np.stack(self._obs_buffer), axis=0)

        return max_frame, total_reward, truncated, terminated, info

    def reset(self, *args, **kwargs):
        """Clear past frame buffer and init. to first obs. from inner env."""
        self._obs_buffer.clear()
        obs, info = self.env.reset(*args, **kwargs)
        self._obs_buffer.append(obs)
        return obs, info


class ProcessFrame84(gym.ObservationWrapper):
    def __init__(self, env=None):
        super(ProcessFrame84, self).__init__(env)
        self.observation_space = spaces.Box(low=0, high=255, shape=(84, 84, 1))

    def observation(self, obs):
        return ProcessFrame84.process(obs)

    @staticmethod
    def process(frame):
        if frame.size == 210 * 160 * 3:
            img = np.reshape(frame, [210, 160, 3]).astype(np.float32)
        elif frame.size == 250 * 160 * 3:
            img = np.reshape(frame, [250, 160, 3]).astype(np.float32)
        else:
            assert False, "Unknown resolution."
        img = img[:, :, 0] * 0.299 + img[:, :, 1] * 0.587 + img[:, :, 2] * 0.114
        resized_screen = cv2.resize(img, (84, 110), interpolation=cv2.INTER_AREA)
        x_t = resized_screen[18:102, :]
        x_t = np.reshape(x_t, [84, 84, 1])
        return x_t.astype(np.uint8)


class ClippedRewardsWrapper(gym.RewardWrapper):
    def reward(self, reward):
        """Change all the positive rewards to 1, negative to -1 and keep zero."""
        return np.sign(reward)


class LazyFrames(object):
    def __init__(self, frames):
        """This object ensures that common frames between the observations are only stored once.
        It exists purely to optimize memory usage which can be huge for DQN's 1M frames replay
        buffers.

        This object should only be converted to numpy array before being passed to the model.

        You'd not belive how complex the previous solution was."""
        self._frames = frames

    def __array__(self, dtype=None):
        out = np.concatenate(self._frames, axis=2)
        if dtype is not None:
            out = out.astype(dtype)
        return out


class FrameStack(gym.Wrapper):
    def __init__(self, env, k):
        """Stack k last frames.

        Returns lazy array, which is much more memory efficient.

        See Also
        --------
        baselines.common.atari_wrappers.LazyFrames
        """
        gym.Wrapper.__init__(self, env)
        self.k = k
        self.frames = deque([], maxlen=k)
        shp = env.observation_space.shape
        self.observation_space = spaces.Box(low=0, high=255, shape=(shp[0], shp[1], shp[2] * k))

    def reset(self, *args, **kwargs):
        ob, info = self.env.reset(*args, **kwargs)
        for _ in range(self.k):
            self.frames.append(ob)
        return self._get_ob(), info

    def step(self, action):
        ob, reward, truncated, terminated, info = self.env.step(action)
        self.frames.append(ob)
        return self._get_ob(), reward, truncated, terminated, info

    def _get_ob(self):
        assert len(self.frames) == self.k
        return np.array(LazyFrames(list(self.frames)))

class ScaledFloatFrame(gym.ObservationWrapper):
    def observation(self, obs):
        # careful! This undoes the memory optimization, use
        # with smaller replay buffers only.
        return np.array(obs).astype(np.float32) / 255.0

class TransposeWrapper(gym.ObservationWrapper):
    def observation(self, obs):
        # return np.transpose(obs, (2, 0, 1))
        return get_pong_symbolic(obs)
    def __init__(self, env=None):
        super(TransposeWrapper, self).__init__(env)
        # self.observation_space = spaces.Box(low=0, high=255, shape=(4, 84, 84), dtype=np.uint8)
        self.observation_space = spaces.Box(low=-336, high=336, shape=(7,), dtype=np.float32)
    def step(self, action):
        obs, reward, truncated, terminated, info = self.env.step(action)
        obs = self.observation(obs)
        return obs, reward, truncated, terminated, info
    def reset(self, *args, **kwargs):
        obs, info = self.env.reset(*args, **kwargs)
        obs = self.observation(obs)
        return obs, info

class conceptRLWrapper(gym.ObservationWrapper):
    def observation(self, obs):
        return np.transpose(obs, (2, 0, 1))
        # return get_pong_symbolic(obs)
    def __init__(self, env=None):
        super(conceptRLWrapper, self).__init__(env)
        self.observation_space = spaces.Box(low=0, high=255, shape=(4, 84, 84), dtype=np.uint8)
        self.concept = None
        self.task_types = ['classification'] * 7
        self.num_classes = [88*2, 88, 88*2, 88*2, 88*2, 88*4, 88*8]
        # self.num_classes = [14, 14, 14, 14, 14, 14, 14] # with categorized_concept
        self.offset = [88, 0, 88, 88, 88, 88*2, 88*4]
        # self.num_classes = [88*2//10+1, 88//10+1, 88*2//10+1, 88*2//10+1, 88*2//10+1, 88*4//10+1, 88*8//10+1] # with categorized_concept
        self.concept_names = ['pos_x_ball', 'pos_y_ball', 'vel_x_ball', 'vel_y_ball', 'vel_paddle', 'acc_paddle', 'jerk_paddle']
        # self.observation_space = spaces.Box(low=-336, high=336, shape=(7,), dtype=np.float32)
    def get_concept(self):
        concept = np.array([x+self.offset[i] for i, x in enumerate(self.concept)])
        # concept = self.get_categorized_concept(concept)
        for i in range(len(concept)):
            assert concept[i] < self.num_classes[i]
        return concept
    def step(self, action):
        obs, reward, truncated, terminated, info = self.env.step(action)
        self.concept = get_pong_symbolic(obs)
        obs = self.observation(obs)
        return obs, reward, truncated, terminated, info
    def reset(self, *args, **kwargs):
        obs, info = self.env.reset(*args, **kwargs)
        self.concept = get_pong_symbolic(obs)
        obs = self.observation(obs)
        return obs, info

def wrap_dqn(env):
    """Apply a common set of wrappers for Atari games."""
    assert 'NoFrameskip' in env.spec.id
    env = EpisodicLifeEnv(env)
    env = NoopResetEnv(env, noop_max=30)
    env = MaxAndSkipEnv(env, skip=4)
    if 'FIRE' in env.unwrapped.get_action_meanings():
        env = FireResetEnv(env)
    env = ProcessFrame84(env)
    env = FrameStack(env, 4)
    env = ClippedRewardsWrapper(env)
    # env = TransposeWrapper(env) # concept input
    env = conceptRLWrapper(env) # concept RL
    return env

def _get_our_paddle(obs_t):
    the_slice = obs_t[:,74]
    pad_color = np.full(the_slice.shape, 147)
    diff_to_paddle = np.absolute(the_slice - pad_color)
    return np.argmin(diff_to_paddle) + 3

def _get_enemy_paddle(obs_t):
    the_slice = obs_t[:,9]
    pad_color = np.full(the_slice.shape, 148)
    diff_to_paddle = np.absolute(the_slice - pad_color)
    return np.argmin(diff_to_paddle) + 3

def _get_ball(obs_t):
    if (np.max(obs_t) < 200):
        return np.array([0, 0])
    idx = np.argmax(obs_t)
    return np.unravel_index(idx, obs_t.shape)

def get_pong_symbolic(obs):
    # preprocessing
    obs = obs[:83,:,:]
    obs = [obs[:,:,3], obs[:,:,2], obs[:,:,1], obs[:,:,0]]
    b = [_get_ball(ob)[0] for ob in obs]
    c = [_get_ball(ob)[1] for ob in obs]
    p = [_get_our_paddle(ob) for ob in obs]

    # new state
    symb = []

    # position
    symb.append(b[0] - p[0])
    symb.append(c[0])

    # velocity
    symb.append(b[0] - b[1])
    symb.append(c[0] - c[1])
    symb.append(p[0] - p[1])

    # acceleration
    symb.append(p[0] - 2.0*p[1] + p[2])

    # jerk
    symb.append(p[0] - 3.0*p[1] + 3.0*p[2] - p[3])

    return np.array(symb)

def get_pong_env(seed=None):
    # return AtariWrapper(gym.make('PongNoFrameskip-v4'))
    return wrap_dqn(gym.make('PongNoFrameskip-v4'))
    # return gym.make('PongNoFrameskip-v4')

def make_pong_vec_env(rank, seed=0):
    def _init() -> gym.Env:
        env = get_pong_env(seed=seed + rank)
        env = Monitor(env)
        env.reset(seed=seed + rank)
        return env

    set_random_seed(seed)
    return _init

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
          
          concept_acc.append(model.policy.concept_net.compute_metric(concepts_pred, concepts_gt))
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
    parser.add_argument('--unlabeled_set_ratio', type=int, default=4)
    parser.add_argument('--model_ensembles', type=int, default=5)
    parser.add_argument('--num_samples', type=int, default=500)
    parser.add_argument('--num_queries', type=int, default=1)
    parser.add_argument('--max_frames', type=int, default=2)
    parser.add_argument('--active_learning_ratio', type=float, default=1)
    parser.add_argument('--active_learning_bsz', type=int, default=20)
    parser.add_argument('--intervene', action='store_true')
    parser.add_argument('--no_share_features_extractor', action='store_true')


    args = parser.parse_args()

    config = {
        "total_timesteps": int(1e7),
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
        "gpt4o": False,
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
            config["run_name"] += f'-active({args.unlabeled_set_ratio},{args.model_ensembles})'
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
        config["run_name"] += f"-ver_viper"
    config["run_name"] += "-classification"

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
    if args.game == "Pong":
        env = SubprocVecEnv([make_pong_vec_env(rank=config["seed"]+i, seed=config["seed"]) for i in range(config["num_envs"])])
        non_vectorized_env = get_pong_env()
        non_vectorized_env.reset(seed=config["seed"])
    
    gpt4o_prompt = None
    gpt4o_checker = None
    gpt4o_path = None
    if config["gpt4o"]:
        from gpt4o_checker_Pong import concept_str_to_list
        from gpt4o import prompt_pong
        gpt4o_prompt = prompt_pong
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
            "net_arch": dict(pi=[64, 64], vf=[64, 64]), # if config["share_features_extractor"]==True else dict(pi=[64, 64], vf=[]), # after the concept layer
            "use_embedding": True,
        },
        concept_loss_type=config["concept_loss_type"],
        con_coef=config["con_coef"],
        intervention=config["intervention"],
        non_vectorized_env=non_vectorized_env,
        num_samples=num_samples,
        accept_rate=config["accept_rate"],
        active_learning=config["active_learning"],
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
    if args.run_id != -1:
        model.save(f"models/{run.id}/model.zip")

    if config["concept_loss_type"] != "concept_input" and config["concept_loss_type"] != "no_concept":
        non_vectorized_env.reset(seed=42)
        mean_reward, mean_concept_acc = evaluate_run(non_vectorized_env, n_episodes=20, intervene=config["intervention"])
        wandb.log({"final/mean_reward": mean_reward, "final/mean_concept_acc": mean_concept_acc})
    else:
        non_vectorized_env.reset(seed=42)
        mean_reward, std_reward = evaluate_policy(model, non_vectorized_env, n_eval_episodes=20)
        wandb.log({"final/mean_reward": mean_reward, "final/std_reward": std_reward})

    env.close()
    wandb.finish()
