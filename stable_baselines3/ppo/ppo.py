import warnings, random
from typing import Any, ClassVar, Dict, Optional, Type, TypeVar, Union

import numpy as np
import torch as th
from gymnasium import spaces
from torch.nn import functional as F

from stable_baselines3.common.buffers import RolloutBuffer
from stable_baselines3.common.on_policy_algorithm import OnPolicyAlgorithm
from stable_baselines3.common.policies import ActorCriticCnnPolicy, ActorCriticPolicy, BasePolicy, MultiInputActorCriticPolicy, CustomConceptNet
from stable_baselines3.common.type_aliases import GymEnv, MaybeCallback, Schedule
from stable_baselines3.common.utils import explained_variance, get_schedule_fn
from stable_baselines3.common.utils import obs_as_tensor
from torch.utils.data import Dataset, DataLoader
import hashlib
import matplotlib.pyplot as plt

SelfPPO = TypeVar("SelfPPO", bound="PPO")

############################################ start of gpt-4o part ############################################
# import base64
# import wandb
# from io import BytesIO
# from PIL import Image
# import os, cv2
# import openai
# from openai import OpenAI
# import base64
# openai.api_key = os.environ["OPENAI_API_KEY"]
# client = OpenAI()
# gpt4o_performance = [0 for i in range(100)] # anyway!
# gpt4o_total = 0

# def encode_image(image_path):
#     with open(image_path, "rb") as image_file:
#         return base64.b64encode(image_file.read()).decode('utf-8')

# def gpt4o_qry(img, ground_truth, gpt4o_prompt, gpt4o_checker, concept_names, save_path, additional_info=None, wandb_log_name=''):
#     if not os.path.exists(save_path):
#         os.makedirs(save_path)

#     global gpt4o_performance, gpt4o_total
#     if img.shape[0] == 4 or img.shape[0] == 2: # CartPole or Pong or Boxing
#         plt.imsave(f'{save_path}/{gpt4o_total}.jpg', np.vstack(img.astype(np.uint8)), cmap='gray')
#     else:
#         image = Image.fromarray(img.astype(np.uint8))
#         image.save(f'{save_path}/{gpt4o_total}.jpg')
#     image_base64 = encode_image(f'{save_path}/{gpt4o_total}.jpg')

#     if additional_info:
#         if additional_info == 0:
#             gpt4o_prompt = gpt4o_prompt.replace('[_]', 'pushing cart to the left')
#         elif additional_info == 1:
#             gpt4o_prompt = gpt4o_prompt.replace('[_]', 'pushing cart to the right')
#         else:
#             assert False

#     while True:
#         try:
#             response = client.chat.completions.create(
#                 model="gpt-4o",
#                 messages=[
#                 {
#                     "role": "user",
#                     "content": [
#                     {
#                         "type": "text",
#                         "text": gpt4o_prompt,
#                     },
#                     {
#                         "type": "image_url",
#                         "image_url": {
#                         "url": f"data:image/jpeg;base64,{image_base64}",
#                         },
#                     },
#                     ],
#                 }
#                 ],
#                 max_tokens=900,
#                 temperature=0,
#                 seed=42,
#             )
#             res = gpt4o_checker(response.choices[0].message.content)
#             for x in res:
#                 assert x is not None # in case GPT-4o randomly fails
#             break
#         except Exception as e:
#             print(f"Error generating response: {e}")
#             try:
#                 # save image
#                 image = Image.fromarray(img.astype(np.uint8))
#                 image.save(f'{save_path}/error.jpg')
#                 if 'response' in locals() and response:
#                     with open(f'{save_path}/error.txt', 'w') as f:
#                         f.write(response.choices[0].message.content + '\n') # maybe API error
#             except:
#                 pass
#     with open(f'{save_path}/{gpt4o_total}.txt', 'w') as f:
#         f.write(response.choices[0].message.content + '\n')
#     print(gpt4o_total, response.system_fingerprint, response.usage, res, ground_truth)
#     gpt4o_total += 1
#     gpt4o_performance = gpt4o_performance[:len(ground_truth)]
#     for i in range(len(ground_truth)):
#         if additional_info is not None: #NOTE CartPole
#             gpt4o_performance[i] += (res[i] - ground_truth[i]) ** 2
#         elif img.shape[0] == 2: #NOTE Boxing
#             gpt4o_performance[i] += int(abs(res[i]-ground_truth[i]) <= 5)
#         else:
#             gpt4o_performance[i] += int(res[i] == ground_truth[i])
#     info = {f"gpt4o{wandb_log_name}/overall": sum(gpt4o_performance)/len(gpt4o_performance)/gpt4o_total, f"gpt4o{wandb_log_name}/total": gpt4o_total}
#     for i in range(len(ground_truth)):
#         info[f'gpt4o{wandb_log_name}/'+concept_names[i]] = gpt4o_performance[i]/gpt4o_total
#     wandb.log(info)
#     return res

############################################ end of gpt-4o part ############################################

class LabeledImageSet:
    def __init__(self):
        self.hash_set = {}  # Use dictionary to store hash values and corresponding labels

    def _hash_array(self, array):
        array_bytes = array.tobytes()
        array_hash = hashlib.md5(array_bytes).hexdigest()
        return array_hash

    def add(self, array, label):
        array_hash = self._hash_array(array)
        if array_hash in self.hash_set:
            return False
        else:
            self.hash_set[array_hash] = label
            return True

    def contains(self, array):
        array_hash = self._hash_array(array)
        return array_hash in self.hash_set

    def get_label(self, array):
        array_hash = self._hash_array(array)
        return self.hash_set.get(array_hash, None)

def add_to_dataset(index, dataset, target_data): # target_data.add(dataset[index])
    if isinstance(dataset.observations, dict):
        obs = {k: v[index] for k, v in dataset.observations.items()}
    else:
        obs = dataset.observations[index]

    concept = dataset.concepts[index]
    action = dataset.actions[index]
    reward = dataset.rewards[index]
    episode_start = dataset.episode_starts[index]
    value = th.from_numpy(dataset.values[index])
    log_prob = th.from_numpy(dataset.log_probs[index])

    target_data.buffer_size += 1
    target_data.add(obs, concept, action, reward, episode_start, value, log_prob)

class PPO(OnPolicyAlgorithm):
    """
    Proximal Policy Optimization algorithm (PPO) (clip version)

    Paper: https://arxiv.org/abs/1707.06347
    Code: This implementation borrows code from OpenAI Spinning Up (https://github.com/openai/spinningup/)
    https://github.com/ikostrikov/pytorch-a2c-ppo-acktr-gail and
    Stable Baselines (PPO2 from https://github.com/hill-a/stable-baselines)

    Introduction to PPO: https://spinningup.openai.com/en/latest/algorithms/ppo.html

    :param policy: The policy model to use (MlpPolicy, CnnPolicy, ...)
    :param env: The environment to learn from (if registered in Gym, can be str)
    :param learning_rate: The learning rate, it can be a function
        of the current progress remaining (from 1 to 0)
    :param n_steps: The number of steps to run for each environment per update
        (i.e. rollout buffer size is n_steps * n_envs where n_envs is number of environment copies running in parallel)
        NOTE: n_steps * n_envs must be greater than 1 (because of the advantage normalization)
        See https://github.com/pytorch/pytorch/issues/29372
    :param batch_size: Minibatch size
    :param n_epochs: Number of epoch when optimizing the surrogate loss
    :param gamma: Discount factor
    :param gae_lambda: Factor for trade-off of bias vs variance for Generalized Advantage Estimator
    :param clip_range: Clipping parameter, it can be a function of the current progress
        remaining (from 1 to 0).
    :param clip_range_vf: Clipping parameter for the value function,
        it can be a function of the current progress remaining (from 1 to 0).
        This is a parameter specific to the OpenAI implementation. If None is passed (default),
        no clipping will be done on the value function.
        IMPORTANT: this clipping depends on the reward scaling.
    :param normalize_advantage: Whether to normalize or not the advantage
    :param ent_coef: Entropy coefficient for the loss calculation
    :param vf_coef: Value function coefficient for the loss calculation
    :param max_grad_norm: The maximum value for the gradient clipping
    :param use_sde: Whether to use generalized State Dependent Exploration (gSDE)
        instead of action noise exploration (default: False)
    :param sde_sample_freq: Sample a new noise matrix every n steps when using gSDE
        Default: -1 (only sample at the beginning of the rollout)
    :param rollout_buffer_class: Rollout buffer class to use. If ``None``, it will be automatically selected.
    :param rollout_buffer_kwargs: Keyword arguments to pass to the rollout buffer on creation
    :param target_kl: Limit the KL divergence between updates,
        because the clipping is not enough to prevent large update
        see issue #213 (cf https://github.com/hill-a/stable-baselines/issues/213)
        By default, there is no limit on the kl div.
    :param stats_window_size: Window size for the rollout logging, specifying the number of episodes to average
        the reported success rate, mean episode length, and mean reward over
    :param tensorboard_log: the log location for tensorboard (if None, no logging)
    :param policy_kwargs: additional arguments to be passed to the policy on creation
    :param verbose: Verbosity level: 0 for no output, 1 for info messages (such as device or wrappers used), 2 for
        debug messages
    :param seed: Seed for the pseudo random generators
    :param device: Device (cpu, cuda, ...) on which the code should be run.
        Setting it to auto, the code will be run on the GPU if possible.
    :param _init_setup_model: Whether or not to build the network at the creation of the instance
    """

    policy_aliases: ClassVar[Dict[str, Type[BasePolicy]]] = {
        "MlpPolicy": ActorCriticPolicy,
        "CnnPolicy": ActorCriticCnnPolicy,
        "MultiInputPolicy": MultiInputActorCriticPolicy,
    }

    def __init__(
        self,
        policy: Union[str, Type[ActorCriticPolicy]],
        env: Union[GymEnv, str],
        learning_rate: Union[float, Schedule] = 3e-4,
        n_steps: int = 2048,
        batch_size: int = 64,
        n_epochs: int = 10,
        gamma: float = 0.99,
        gae_lambda: float = 0.95,
        clip_range: Union[float, Schedule] = 0.2,
        clip_range_vf: Union[None, float, Schedule] = None,
        normalize_advantage: bool = True,
        ent_coef: float = 0.0,
        vf_coef: float = 0.5,
        max_grad_norm: float = 0.5,
        use_sde: bool = False,
        sde_sample_freq: int = -1,
        rollout_buffer_class: Optional[Type[RolloutBuffer]] = None,
        rollout_buffer_kwargs: Optional[Dict[str, Any]] = None,
        target_kl: Optional[float] = None,
        stats_window_size: int = 100,
        tensorboard_log: Optional[str] = None,
        policy_kwargs: Optional[Dict[str, Any]] = None,
        verbose: int = 0,
        seed: Optional[int] = None,
        device: Union[th.device, str] = "auto",
        _init_setup_model: bool = True,
        concept_loss_type: Optional[str] = None,
        con_coef: float = 0.5,
        intervention: bool = False,
        non_vectorized_env = None,
        expert_model = None,
        create_test_set = False,
        num_samples = 0,
        accept_rate = 1,
        active_learning = False,
        active_learning_ratio = 1,
        active_learning_bsz = 20,
        unlabeled_set_ratio = 10,
        model_ensembles = 5,
        gpt4o = False,
        initialize_parameters = False,
        hashing = False,
        gpt4o_prompt = None,
        gpt4o_checker = None,
        gpt4o_path = None,
        use_v9 = False,
    ):
        super().__init__(
            policy,
            env,
            learning_rate=learning_rate,
            n_steps=n_steps,
            gamma=gamma,
            gae_lambda=gae_lambda,
            ent_coef=ent_coef,
            vf_coef=vf_coef,
            max_grad_norm=max_grad_norm,
            use_sde=use_sde,
            sde_sample_freq=sde_sample_freq,
            rollout_buffer_class=rollout_buffer_class,
            rollout_buffer_kwargs=rollout_buffer_kwargs,
            stats_window_size=stats_window_size,
            tensorboard_log=tensorboard_log,
            policy_kwargs=policy_kwargs,
            verbose=verbose,
            device=device,
            seed=seed,
            _init_setup_model=False,
            supported_action_spaces=(
                spaces.Box,
                spaces.Discrete,
                spaces.MultiDiscrete,
                spaces.MultiBinary,
            ),
        )

        # Sanity check, otherwise it will lead to noisy gradient and NaN
        # because of the advantage normalization
        if normalize_advantage:
            assert (
                batch_size > 1
            ), "`batch_size` must be greater than 1. See https://github.com/DLR-RM/stable-baselines3/issues/440"

        if self.env is not None:
            # Check that `n_steps * n_envs > 1` to avoid NaN
            # when doing advantage normalization
            buffer_size = self.env.num_envs * self.n_steps
            assert buffer_size > 1 or (
                not normalize_advantage
            ), f"`n_steps * n_envs` must be greater than 1. Currently n_steps={self.n_steps} and n_envs={self.env.num_envs}"
            # Check that the rollout buffer size is a multiple of the mini-batch size
            untruncated_batches = buffer_size // batch_size
            if buffer_size % batch_size > 0:
                warnings.warn(
                    f"You have specified a mini-batch size of {batch_size},"
                    f" but because the `RolloutBuffer` is of size `n_steps * n_envs = {buffer_size}`,"
                    f" after every {untruncated_batches} untruncated mini-batches,"
                    f" there will be a truncated mini-batch of size {buffer_size % batch_size}\n"
                    f"We recommend using a `batch_size` that is a factor of `n_steps * n_envs`.\n"
                    f"Info: (n_steps={self.n_steps} and n_envs={self.env.num_envs})"
                )
        self.batch_size = batch_size
        self.n_epochs = n_epochs
        self.clip_range = clip_range
        self.clip_range_vf = clip_range_vf
        self.normalize_advantage = normalize_advantage
        self.target_kl = target_kl
        self.concept_loss_type = concept_loss_type
        self.con_coef = con_coef
        self.train_step = 0
        self.intervention = intervention
        self.non_vectorized_env = non_vectorized_env
        self.train_concept_step = 0
        self.expert_model = expert_model
        self.create_test_set = create_test_set
        self.accept_rate = accept_rate
        self.active_learning = active_learning
        self.active_learning_ratio = active_learning_ratio
        self.active_learning_bsz = active_learning_bsz
        self.unlabeled_set_ratio = unlabeled_set_ratio
        self.model_ensembles = model_ensembles
        self.total_samples = num_samples
        self.gpt4o = gpt4o
        self.initialize_parameters = initialize_parameters
        self.hashing = hashing
        self.labeled_image_set = LabeledImageSet()
        self.gpt4o_prompt = gpt4o_prompt
        self.gpt4o_checker = gpt4o_checker
        self.gpt4o_path = gpt4o_path
        self.use_v9 = use_v9

        if _init_setup_model:
            self._setup_model()
            
            self.train_data = self.rollout_buffer_class(
                self.total_samples,
                self.observation_space,  # type: ignore[arg-type]
                self.policy_kwargs['concept_dim'],
                self.action_space,
                device=self.device,
                gamma=self.gamma,
                gae_lambda=self.gae_lambda,
                n_envs=1,
                **self.rollout_buffer_kwargs,
            )
            self.valid_data = self.rollout_buffer_class(
                self.total_samples, # this is a loose upper bound!
                self.observation_space,  # type: ignore[arg-type]
                self.policy_kwargs['concept_dim'],
                self.action_space,
                device=self.device,
                gamma=self.gamma,
                gae_lambda=self.gae_lambda,
                n_envs=1,
                **self.rollout_buffer_kwargs,
            )
            self.train_data.buffer_size = 0 # a trick to be compatible with RolloutBuffer!
            self.valid_data.buffer_size = 0
        
    def _setup_model(self) -> None:
        super()._setup_model()

        # Initialize schedules for policy/value clipping
        self.clip_range = get_schedule_fn(self.clip_range)
        if self.clip_range_vf is not None:
            if isinstance(self.clip_range_vf, (float, int)):
                assert self.clip_range_vf > 0, "`clip_range_vf` must be positive, " "pass `None` to deactivate vf clipping"

            self.clip_range_vf = get_schedule_fn(self.clip_range_vf)

    def train_concept(self, num_samples, patience, last_iter):
        batch_size = 32
        n_epochs = 50
        if self.use_v9:
            n_epochs = 100
            patience = 100
        print(f'[seed={self.seed} run] train_concept {n_epochs} epochs with patience {patience}')

        # step1: Generate unlabeled dataset (10N)
        unlabeled_dataset = self.rollout_buffer_class(
            num_samples*self.unlabeled_set_ratio if self.active_learning else num_samples,
            self.observation_space,  # type: ignore[arg-type]
            self.policy_kwargs['concept_dim'],
            self.action_space,
            device=self.device,
            gamma=self.gamma,
            gae_lambda=self.gae_lambda,
            n_envs=1,
            **self.rollout_buffer_kwargs,
        )
        obs, info = self.non_vectorized_env.reset()
        while not unlabeled_dataset.full:
            action, _ = self.policy.predict(obs, deterministic=False)
            obs, reward, done, truncated, info = self.non_vectorized_env.step(action)
            if done or truncated:
                obs, info = self.non_vectorized_env.reset()
            concept = self.non_vectorized_env.get_concept()
            if random.random() < self.accept_rate:
                if not isinstance(obs, dict):
                    vec_obs = obs.reshape(1, *obs.shape)
                else:
                    vec_obs = {}
                    for key, value in obs.items():
                        vec_obs[key] = value.reshape(1, *value.shape)
                # Add the current interaction to the buffer
                unlabeled_dataset.add(
                    obs=vec_obs,
                    concept=concept.reshape(1, -1),  # reshape concept if necessary
                    action=np.array(action).reshape((1, *self.action_space.shape)),
                    reward=np.array([0]),
                    episode_start=np.array([0]),
                    value=th.tensor([0.0]),
                    log_prob=th.tensor([0.0])
                )

        # step2: active learning. determine a subset and further divide into (train_idx, valid_idx) for unlabeled_dataset
        if self.active_learning:
            train_idx = []
            valid_idx = []
            new_train_data = self.rollout_buffer_class(
                self.train_data.buffer_size + num_samples,
                self.observation_space,  # type: ignore[arg-type]
                self.policy_kwargs['concept_dim'],
                self.action_space,
                device=self.device,
                gamma=self.gamma,
                gae_lambda=self.gae_lambda,
                n_envs=1,
                **self.rollout_buffer_kwargs,
            )
            new_train_data.buffer_size = 0
            for idx in range(self.train_data.buffer_size):
                add_to_dataset(idx, self.train_data, new_train_data)
            new_valid_data = self.rollout_buffer_class(
                self.train_data.buffer_size + num_samples,
                self.observation_space,  # type: ignore[arg-type]
                self.policy_kwargs['concept_dim'],
                self.action_space,
                device=self.device,
                gamma=self.gamma,
                gae_lambda=self.gae_lambda,
                n_envs=1,
                **self.rollout_buffer_kwargs,
            )
            new_valid_data.buffer_size = 0
            for idx in range(self.valid_data.buffer_size):
                add_to_dataset(idx, self.valid_data, new_valid_data)
            total_budget = num_samples
            uncertainty_mask = np.ones(unlabeled_dataset.buffer_size, dtype=bool)
            generator_ = unlabeled_dataset.get(1)  # Call the get method, generator is created
            _ = next(generator_)  # Trigger the generator to run at least once
            assert unlabeled_dataset.generator_ready
            while total_budget:
                budget = min(total_budget, self.active_learning_bsz)
                total_budget -= budget

                if new_train_data.buffer_size == 0 or total_budget > num_samples * self.active_learning_ratio: #NOTE in the beginning, randomly select samples; and ensure the speed is fast
                    uncertainty = np.random.rand(unlabeled_dataset.buffer_size)
                else:
                    # train our model
                    nets = [CustomConceptNet(self.policy).cuda() for _ in range(self.model_ensembles)]
                    for _, net in enumerate(nets):
                        net.train()
                        optimizer = th.optim.Adam(net.parameters(), lr=3e-4)
                        scheduler = th.optim.lr_scheduler.LinearLR(optimizer, start_factor=1.0, end_factor=0, total_iters=n_epochs)
                        best_val_loss = float('inf')  # Variable to store the best validation loss, initialized to infinity
                        best_model_params = None  # Used to store the best model parameters in memory
                        best_epoch = 0
                        for epoch in range(n_epochs):
                            net.train()
                            for rollout_data in new_train_data.get(batch_size):
                                obs_batch, concepts_batch = rollout_data.observations, rollout_data.concepts
                                concepts_logits = net.get_logits(obs_batch)
                                concept_loss = net.concept_net.compute_loss(concepts_logits, concepts_batch)
                                optimizer.zero_grad()
                                concept_loss.backward()
                                optimizer.step()
                            if last_iter or self.use_v9:
                                scheduler.step()

                            net.eval()
                            total_val_loss = 0
                            total_val_samples = 0
                            with th.no_grad():
                                for rollout_data in new_valid_data.get(batch_size):
                                    obs_batch, concepts_batch = rollout_data.observations, rollout_data.concepts
                                    concepts_vals = net.forward(obs_batch)
                                    concept_metric = self.policy.concept_net.compute_metric(concepts_vals, concepts_batch)
                                    total_val_loss += concept_metric * concepts_batch.shape[0]
                                    total_val_samples += concepts_batch.shape[0]
                            
                            if all(x == 'regression' for x in self.policy.task_types): # MSELoss
                                avg_val_loss = total_val_loss / total_val_samples
                            elif all(x == 'classification' for x in self.policy.task_types): # -acc
                                avg_val_loss = -total_val_loss / total_val_samples

                            if avg_val_loss < best_val_loss:
                                best_val_loss = avg_val_loss
                                best_epoch = epoch
                                best_model_params = {name: param.clone() for name, param in net.state_dict().items()}
                            if epoch - best_epoch >= patience:
                                break

                        if best_model_params is not None:
                            nets[_].load_state_dict(best_model_params)
                            print(f'[seed={self.seed} run] active learning, choose epoch', best_epoch, 'with valid loss', best_val_loss)

                    # get the most uncertain samples
                    def compute_uncertainty(nets, all_indices, task_type):
                        def pad_tensors(tensor_list):
                            bsz = tensor_list[0].size(0)
                            num_tensors = len(tensor_list)
                            max_len = max(tensor.size(1) for tensor in tensor_list)
                            output = th.zeros(bsz, num_tensors, max_len)
                            for i, tensor in enumerate(tensor_list):
                                length = tensor.size(1)
                                output[:, i, :length] = tensor
                            return output
                        
                        all_predictions = []
                        all_logits = []
                        batch_size = 128
                        for net in nets:
                            net.eval()
                            with th.no_grad():
                                predictions = []
                                logits = []
                                for i in range(0, len(all_indices), batch_size):
                                    batch_inds = all_indices[i:i + batch_size]
                                    batch = unlabeled_dataset._get_samples(batch_inds)
                                    predictions.append(net(batch.observations))
                                    unpadded_logits = net.get_logits(batch.observations)
                                    logits.append(pad_tensors(unpadded_logits))
                                predictions = th.cat(predictions, dim=0)
                                logits = th.cat(logits, dim=0) # (num_samples, concept_dim, max_num_classes)
                            all_predictions.append(predictions)
                            all_logits.append(logits)

                        all_predictions = th.stack(all_predictions, dim=0)  # (num_models, num_samples, concept_dim)
                        all_logits = th.stack(all_logits, dim=0)  # (num_models, num_samples, concept_dim, max_num_classes)

                        uncertainties = []
                        if task_type == 'classification':
                            for i in range(all_predictions.shape[2]):
                                concept_predictions = all_predictions[:, :, i]  # (num_models, num_samples)
                                majority_votes, _ = th.mode(concept_predictions, dim=0)  # (num_samples,)
                                majority_count = th.sum(concept_predictions == majority_votes, dim=0)  # (num_samples,)
                                uncertainty = 1 - majority_count.float() / len(nets)  # (num_samples,)
                                uncertainties.append(uncertainty)
                        elif task_type == 'regression':
                            for i in range(all_predictions.shape[2]):
                                concept_values = all_predictions[:, :, i]  # (num_models, num_samples)
                                uncertainty = th.var(concept_values, dim=0)  # (num_samples,)
                                uncertainties.append(uncertainty)
                        else:
                            assert False

                        uncertainties = th.stack(uncertainties, dim=1)  # (num_samples, concept_dim)
                        uncertainty = uncertainties.mean(dim=1).cpu().numpy()  # take average for the concepts
                        return uncertainty
                    
                    # Retrieve the index of all unmarked samples
                    all_indices = np.arange(unlabeled_dataset.buffer_size)
                    assert all(x == 'classification' for x in self.policy.task_types) or all(x == 'regression' for x in self.policy.task_types)
                    uncertainty = compute_uncertainty(nets, all_indices, task_type=self.policy.task_types[0])

                uncertainty[~uncertainty_mask] = -np.inf

                # Select the data points to be marked
                indices_to_label = []
                for idx in np.argsort(-uncertainty):
                    if not uncertainty_mask[idx] or self.hashing and self.labeled_image_set.contains(unlabeled_dataset.observations[idx].transpose((1, 2, 0))):  # check if it is already marked
                        continue
                    if self.hashing:
                        self.labeled_image_set.add(unlabeled_dataset.observations[idx].transpose((1, 2, 0)), unlabeled_dataset.concepts[idx])
                    uncertainty_mask[idx] = False  # mark
                    indices_to_label.append(idx)
                    if self.gpt4o:
                        if isinstance(unlabeled_dataset.observations, dict): # only CartPole
                            unlabeled_dataset.concepts[idx] = gpt4o_qry(unlabeled_dataset.observations['images'][idx], unlabeled_dataset.concepts[idx], self.gpt4o_prompt, self.gpt4o_checker, self.policy.concept_names, self.gpt4o_path, unlabeled_dataset.observations['last_action'][idx])
                        elif unlabeled_dataset.observations[idx].shape[0] == 2 or unlabeled_dataset.observations[idx].shape[0] == 4: # Boxing or Pong
                            unlabeled_dataset.concepts[idx] = gpt4o_qry(unlabeled_dataset.observations[idx], unlabeled_dataset.concepts[idx], self.gpt4o_prompt, self.gpt4o_checker, self.policy.concept_names, self.gpt4o_path)
                        else:
                            unlabeled_dataset.concepts[idx] = gpt4o_qry(unlabeled_dataset.observations[idx].transpose((1, 2, 0)), unlabeled_dataset.concepts[idx], self.gpt4o_prompt, self.gpt4o_checker, self.policy.concept_names, self.gpt4o_path)
                    if random.random() < 0.1 or new_valid_data.buffer_size == 0: # make sure |valid| = 0 never happens
                        valid_idx.append(idx)
                        add_to_dataset(idx, unlabeled_dataset, new_valid_data)
                    else:
                        train_idx.append(idx)
                        add_to_dataset(idx, unlabeled_dataset, new_train_data)
                    if len(indices_to_label) == budget:
                        break
                print(f'[seed={self.seed} run] budget:', budget, 'uncertainty:', [round(x, 3) for x in uncertainty[indices_to_label]][:100])
            del new_train_data
            del new_valid_data
        else:
            labeled_idx_ = np.random.permutation(num_samples) # no active learning: naively select the first num_samples
            train_idx = labeled_idx_[:int(len(labeled_idx_) * 0.9)]
            valid_idx = labeled_idx_[int(len(labeled_idx_) * 0.9):]
        train_idx = np.array(train_idx)
        valid_idx = np.array(valid_idx)
        for idx in train_idx:
            add_to_dataset(idx, unlabeled_dataset, self.train_data)
        for idx in valid_idx:
            add_to_dataset(idx, unlabeled_dataset, self.valid_data)
        del unlabeled_dataset
        assert self.train_data.full and self.valid_data.full
        print(f'[seed={self.seed} run] train:', self.train_data.buffer_size, 'valid:', self.valid_data.buffer_size)

        # step4: train the concept network g on D_{train}, using D_{validation} for early stopping
        import wandb
        self.policy.set_training_mode(True)
        optimizer = th.optim.Adam(self.policy.parameters(), lr=3e-4)
        scheduler = th.optim.lr_scheduler.LinearLR(optimizer, start_factor=1.0, end_factor=0, total_iters=n_epochs)
        min_valid_loss = float('inf')
        min_valid_epoch = 0
        best_model_state = None
        for epoch in range(n_epochs):
            concept_all_metrics = [[] for i in range(self.policy.concept_dim)]
            concept_metrics = []
            train_concept_metrics = []
            for rollout_data in self.train_data.get(batch_size):
                obs_batch, concepts_batch = rollout_data.observations, rollout_data.concepts

                pi_features = self.policy.extract_features(obs_batch)
                if not self.policy.share_features_extractor:
                    pi_features = pi_features[0]
                concepts_logits = self.policy.concept_net.get_logits(pi_features)
                
                concepts_vals = self.policy.concept_net(pi_features)
                concept_metric = self.policy.concept_net.compute_metric(concepts_vals, concepts_batch)
                train_concept_metrics.append(concept_metric)

                concept_loss = self.policy.concept_net.compute_loss(concepts_logits, concepts_batch)
                optimizer.zero_grad()
                concept_loss.backward()
                optimizer.step()

            with th.no_grad():
                for rollout_data in self.valid_data.get(batch_size):
                    obs_batch, concepts_batch = rollout_data.observations, rollout_data.concepts

                    pi_features = self.policy.extract_features(obs_batch)
                    if not self.policy.share_features_extractor:
                        pi_features = pi_features[0]
                    concepts_vals = self.policy.concept_net(pi_features)
                    concept_metric = self.policy.concept_net.compute_metric(concepts_vals, concepts_batch)
                    concept_metrics.append(concept_metric)
                    tmp = self.policy.concept_net.compute_all_metrics(concepts_vals, concepts_batch)
                    for i in range(self.policy.concept_dim):
                        concept_all_metrics[i].append(tmp[i])
            try:
                recs = {"overall": np.mean(concept_metrics), "train_overall": np.mean(train_concept_metrics), "lr": optimizer.param_groups[0]['lr'], "epoch": self.train_concept_step}
                for i in range(self.policy.concept_dim):
                    recs[f"{self.policy.concept_names[i]}"] = np.mean(concept_all_metrics[i])
                wandb.log(recs)
            except:
                pass
            self.train_concept_step += 1
            if all(x == 'regression' for x in self.policy.task_types): # MSELoss
                current_valid_loss = np.mean(concept_metrics)
            elif all(x == 'classification' for x in self.policy.task_types): # -acc
                current_valid_loss = -np.mean(concept_metrics)
            else:
                assert False
            
            if current_valid_loss < min_valid_loss:
                min_valid_loss = current_valid_loss
                min_valid_epoch = epoch
                best_model_state = {k: v.clone() for k, v in self.policy.state_dict().items()}
            
            if epoch - min_valid_epoch >= patience:
                self.train_concept_step += n_epochs - epoch - 1 # make sure the total number of steps is correct for wandb
                break
            
            if last_iter or self.use_v9:
                scheduler.step()
        
        if best_model_state:
            self.policy.load_state_dict(best_model_state)
            print(f'[seed={self.seed} run] best model at epoch', min_valid_epoch, 'with valid loss', min_valid_loss)

    def train(self) -> None:
        """
        Update policy using the currently gathered rollout buffer.
        """
        # Switch to train mode (this affects batch norm / dropout)
        self.policy.set_training_mode(True)
        # Update optimizer learning rate
        self._update_learning_rate([self.policy.optimizer, self.policy.optimizer_exclude_concept]) # decay for both optimizers!!!
        # Compute current clip range
        clip_range = self.clip_range(self._current_progress_remaining)  # type: ignore[operator]
        # Optional: clip range for the value function
        if self.clip_range_vf is not None:
            clip_range_vf = self.clip_range_vf(self._current_progress_remaining)  # type: ignore[operator]

        entropy_losses = []
        pg_losses, value_losses, concept_metrics = [], [], []
        concept_all_metrics = [[] for i in range(self.policy.concept_dim)]
        clip_fractions = []

        continue_training = True
        # train for n_epochs epochs
        for epoch in range(self.n_epochs):
            approx_kl_divs = []
            # Do a complete pass on the rollout buffer
            for rollout_data in self.rollout_buffer.get(self.batch_size):
                actions = rollout_data.actions
                if isinstance(self.action_space, spaces.Discrete):
                    # Convert discrete action from float to long
                    actions = rollout_data.actions.long().flatten()

                # Re-sample the noise matrix because the log_std has changed
                if self.use_sde:
                    self.policy.reset_noise(self.batch_size)

                concepts_pred, values, log_prob, entropy = self.policy.evaluate_actions(rollout_data.observations, actions, rollout_data.concepts if self.intervention else None)
                values = values.flatten()
                # Normalize advantage
                advantages = rollout_data.advantages
                # Normalization does not make sense if mini batchsize == 1, see GH issue #325
                if self.normalize_advantage and len(advantages) > 1:
                    advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)

                # ratio between old and new policy, should be one at the first iteration
                ratio = th.exp(log_prob - rollout_data.old_log_prob)

                # clipped surrogate loss
                policy_loss_1 = advantages * ratio
                policy_loss_2 = advantages * th.clamp(ratio, 1 - clip_range, 1 + clip_range)
                policy_loss = -th.min(policy_loss_1, policy_loss_2).mean()

                # Logging
                pg_losses.append(policy_loss.item())
                clip_fraction = th.mean((th.abs(ratio - 1) > clip_range).float()).item()
                clip_fractions.append(clip_fraction)

                if self.clip_range_vf is None:
                    # No clipping
                    values_pred = values
                else:
                    # Clip the difference between old and new value
                    # NOTE: this depends on the reward scaling
                    values_pred = rollout_data.old_values + th.clamp(
                        values - rollout_data.old_values, -clip_range_vf, clip_range_vf
                    )
                # Value loss using the TD(gae_lambda) target
                value_loss = F.mse_loss(rollout_data.returns, values_pred)
                value_losses.append(value_loss.item())

                # Entropy loss favor exploration
                if entropy is None:
                    # Approximate entropy when no analytical form
                    entropy_loss = -th.mean(-log_prob)
                else:
                    entropy_loss = -th.mean(entropy)

                entropy_losses.append(entropy_loss.item())

                loss_type = self.concept_loss_type
                if loss_type == "joint_then_vanilla":
                    loss_type = "joint" if self._current_progress_remaining > 0.8 else "vanilla_freeze"
                if loss_type == "ppo_then_joint":
                    loss_type = "no_concept" if self._current_progress_remaining > 0.1 else "joint"
                if loss_type == "joint_concept":
                    loss_type = "joint" if self._current_progress_remaining > 0.2 else "concept"
                if loss_type == "iterative_joint":
                    loss_type = "joint" if self._current_progress_remaining < 0.5 else "iterative"
                if loss_type == "iterative":
                    loss_type = "no_concept" if self.train_step % 2 == 1 else "concept"

                # concept_loss = F.mse_loss(rollout_data.concepts, concepts_pred)

                concept_metrics.append(self.policy.concept_net.compute_metric(concepts_pred, rollout_data.concepts))
                tmp = self.policy.concept_net.compute_all_metrics(concepts_pred, rollout_data.concepts)
                for i in range(self.policy.concept_dim):
                    concept_all_metrics[i].append(tmp[i])

                if loss_type == "joint": # joint (PPO and concept) training
                    pi_features = self.policy.extract_features(rollout_data.observations)
                    if not self.policy.share_features_extractor:
                        pi_features = pi_features[0]
                    concepts_logits = self.policy.concept_net.get_logits(pi_features)
                    concept_loss = self.policy.concept_net.compute_loss(concepts_logits, rollout_data.concepts)
                    loss = policy_loss + self.ent_coef * entropy_loss + self.vf_coef * value_loss + self.con_coef * concept_loss
                elif loss_type == "no_concept" or loss_type == "vanilla_freeze": # only PPO
                    loss = policy_loss + self.ent_coef * entropy_loss + self.vf_coef * value_loss
                elif loss_type == "concept": # only concept
                    pi_features = self.policy.extract_features(rollout_data.observations)
                    if not self.policy.share_features_extractor:
                        pi_features = pi_features[0]
                    concepts_logits = self.policy.concept_net.get_logits(pi_features)
                    concept_loss = self.policy.concept_net.compute_loss(concepts_logits, rollout_data.concepts)
                    loss = concept_loss
                else:
                    assert False

                # Calculate approximate form of reverse KL Divergence for early stopping
                # see issue #417: https://github.com/DLR-RM/stable-baselines3/issues/417
                # and discussion in PR #419: https://github.com/DLR-RM/stable-baselines3/pull/419
                # and Schulman blog: http://joschu.net/blog/kl-approx.html
                with th.no_grad():
                    log_ratio = log_prob - rollout_data.old_log_prob
                    approx_kl_div = th.mean((th.exp(log_ratio) - 1) - log_ratio).cpu().numpy()
                    approx_kl_divs.append(approx_kl_div)

                if self.target_kl is not None and approx_kl_div > 1.5 * self.target_kl:
                    continue_training = False
                    if self.verbose >= 1:
                        print(f"Early stopping at step {epoch} due to reaching max kl: {approx_kl_div:.2f}")
                    break

                # Optimization step
                if loss_type == "vanilla_freeze":
                    self.policy.optimizer_exclude_concept.zero_grad()
                else:
                    self.policy.optimizer.zero_grad()
                # Clip grad norm
                loss.backward()
                th.nn.utils.clip_grad_norm_(self.policy.parameters(), self.max_grad_norm)
                if loss_type == "vanilla_freeze":
                    self.policy.optimizer_exclude_concept.step()
                else:
                    self.policy.optimizer.step()

            self._n_updates += 1
            if not continue_training:
                break

        explained_var = explained_variance(self.rollout_buffer.values.flatten(), self.rollout_buffer.returns.flatten())
        self.train_step += 1
        # Logs
        self.logger.record("train/entropy_loss", np.mean(entropy_losses))
        self.logger.record("train/policy_gradient_loss", np.mean(pg_losses))
        self.logger.record("train/value_loss", np.mean(value_losses))
        if len(concept_metrics)>0:
            self.logger.record("train/concept_overall", np.mean(concept_metrics))
            for i in range(self.policy.concept_dim):
                self.logger.record(f"train/concept_{self.policy.concept_names[i]}", np.mean(concept_all_metrics[i]))
        self.logger.record("train/approx_kl", np.mean(approx_kl_divs))
        self.logger.record("train/clip_fraction", np.mean(clip_fractions))
        self.logger.record("train/loss", loss.item())
        self.logger.record("train/explained_variance", explained_var)
        if hasattr(self.policy, "log_std"):
            self.logger.record("train/std", th.exp(self.policy.log_std).mean().item())

        self.logger.record("train/n_updates", self._n_updates, exclude="tensorboard")
        self.logger.record("train/clip_range", clip_range)
        if self.clip_range_vf is not None:
            self.logger.record("train/clip_range_vf", clip_range_vf)

    def learn(
        self: SelfPPO,
        total_timesteps: int,
        callback: MaybeCallback = None,
        log_interval: int = 1,
        tb_log_name: str = "PPO",
        reset_num_timesteps: bool = True,
        progress_bar: bool = False,
        query_num_times: Optional[int] = None,
        query_labels_per_time: Optional[int] = None,
        use_v9: bool = False,
    ) -> SelfPPO:
        return super().learn(
            total_timesteps=total_timesteps,
            callback=callback,
            log_interval=log_interval,
            tb_log_name=tb_log_name,
            reset_num_timesteps=reset_num_timesteps,
            progress_bar=progress_bar,
            query_num_times=query_num_times,
            query_labels_per_time=query_labels_per_time,
            use_v9=use_v9,
        )
