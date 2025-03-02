import warnings, random
from typing import Any, ClassVar, Dict, Optional, Type, TypeVar, Union

import numpy as np
import torch as th
from gymnasium import spaces
from torch.nn import functional as F

from stable_baselines3.common.buffers import RolloutBuffer
from stable_baselines3.common.on_policy_algorithm import OnPolicyAlgorithm
from stable_baselines3 import PPO
from stable_baselines3.common.policies import ActorCriticCnnPolicy, ActorCriticPolicy, BasePolicy, MultiInputActorCriticPolicy, CustomConceptNet
from stable_baselines3.common.type_aliases import GymEnv, MaybeCallback, Schedule
from stable_baselines3.common.utils import explained_variance, get_schedule_fn
from stable_baselines3.common.utils import obs_as_tensor
from torch.utils.data import Dataset, DataLoader
import hashlib
import matplotlib.pyplot as plt

SelfPPO = TypeVar("SelfPPO", bound="PPO")

class KLRegularizedPPO(PPO):
    def __init__(self, policy, env, anchor_policy, kl_coef=0.1, **kwargs):
        super().__init__(policy, env, **kwargs)
        self.anchor_policy = anchor_policy
        self.kl_coef = kl_coef
        # 确保 anchor_policy 是 ActorCriticPolicy 类型
        if isinstance(anchor_policy, ActorCriticPolicy):
            anchor_params = dict(anchor_policy.named_parameters())
            
            for name, param in self.policy.named_parameters():
                if 'features_extractor' in name or 'concept_net' in name:
                    if name in anchor_params:
                        print('copy param', name)
                        param.data.copy_(anchor_params[name].data)
        else:
            print("Warning: anchor_policy is not an instance of ActorCriticPolicy. Skipping parameter copy.")


    def train(self):
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

                # Compute anchor policy log probabilities
                with th.no_grad():
                    _, _, anchor_log_prob, _ = self.anchor_policy.evaluate_actions(rollout_data.observations, actions)

                # KL divergence loss
                kl_loss = F.kl_div(log_prob, anchor_log_prob, reduction='batchmean', log_target=True)

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
                    loss = policy_loss + self.ent_coef * entropy_loss + self.vf_coef * value_loss + self.con_coef * concept_loss + self.kl_coef * kl_loss
                elif loss_type == "no_concept" or loss_type == "vanilla_freeze" or loss_type == "finetune_policy": # only PPO
                    loss = policy_loss + self.ent_coef * entropy_loss + self.vf_coef * value_loss + self.kl_coef * kl_loss
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
                if loss_type == "vanilla_freeze" or loss_type == "finetune_policy":
                    self.policy.optimizer_exclude_concept.zero_grad()
                else:
                    self.policy.optimizer.zero_grad()
                # Clip grad norm
                loss.backward()
                # for name, param in self.policy.named_parameters():
                #     if param.grad != None:
                #         print('hello', name, param.grad.reshape(-1).shape[0])
                    # 检查参数名是否包含特定的子字符串
                    # if ('features_extractor' in name or 'concept_net' in name) and loss_type == "vanilla_freeze":
                    #     # 如果是不需要更新的部分，将梯度置为0
                    #     if param.grad is not None:
                    #         param.grad.data.zero_()
                    #         print('clear', name)
                th.nn.utils.clip_grad_norm_(self.policy.parameters(), self.max_grad_norm)
                if loss_type == "vanilla_freeze" or loss_type == "finetune_policy":
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

if __name__ == '__main__':
    pass