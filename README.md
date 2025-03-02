# LICORICE: Label-Efficient Concept-Based Interpretable Reinforcement Learning

Official code and data of our paper: **LICORICE: Label-Efficient Concept-Based Interpretable Reinforcement Learning**

Zhuorui Ye*, Stephanie Milani*, Geoffrey J. Gordon, Fei Fang

ICLR 2025 (also Oral presentation at RLC 2024 InterpPol Workshop)

[**[Paper link]**](https://arxiv.org/abs/2407.15786)

## Installation

After cloning this repo, you can install the environment using the following commands at the root directory.

```
conda create --name RL python=3.10
conda activate RL
conda install pytorch torchvision pytorch-cuda=11.8 -c pytorch -c nvidia
pip install -e .[extra]
pip install moviepy wandb minigrid ocatari
pip install "gymnasium[atari, accept-rom-license]"
```

If you want to use GPT-4o as the annotator, please uncomment L22-115 of `stable_baselines3/ppo/ppo.py` and then configurate the OpenAI API key.

```bash
export OPENAI_API_KEY=<your-openai-api-key>
```

## Usage

### Run LICORICE

To run LICORICE in the different environments, use the following commands.

```bash
python Cartpole_train.py --concept_loss_type vanilla_freeze --game CartPole --accept_rate 0.02 --num_samples 500 --num_queries 4 --active_learning_bsz 20 --active_learning
python Minigrid_train.py --concept_loss_type vanilla_freeze --game DoorKey --accept_rate 0.1 --num_samples 300 --num_queries 2 --active_learning_bsz 20 --active_learning
python Minigrid_train.py --concept_loss_type vanilla_freeze --game DynamicObstacles --accept_rate 0.1 --num_samples 300 --num_queries 1 --active_learning_bsz 20 --active_learning
python Atari_train.py --concept_loss_type vanilla_freeze --game Boxing --accept_rate 0.1 --num_samples 3000 --num_queries 5 --active_learning_bsz 120 --active_learning
python Atari_train.py --concept_loss_type vanilla_freeze --game Pong --accept_rate 0.1 --num_samples 5000 --num_queries 5 --active_learning_bsz 200 --active_learning
```

For LICORICE in Boxing and Pong with multiple iterations, we can add another RL training stage from scratch with the last trained policy as the anchoring policy, as introduced in section 3 of the paper. After setting the checkpoint path indicated by the variable`expert_model_path` in the code, use these commands.

```bash
python Atari_train.py --concept_loss_type finetune_policy --game Boxing --accept_rate 0.1 --num_samples 3000 --num_queries 5 --active_learning_bsz 120 --active_learning
python Atari_train.py --concept_loss_type finetune_policy --game Pong --accept_rate 0.1 --num_samples 5000 --num_queries 5 --active_learning_bsz 200 --active_learning
```

### Run baselines

To run baselines, you only need to change the arguments to the training scripts. Specifically, setting `--concept_loss_type early_query` means Sequential-Q, setting `--concept_loss_type random_query` means Random-Q and setting `--concept_loss_type uncertainty_based_query` means Disagreement-Q. Set other arguments according to your need.

### Other notes

- Following the given installing steps, the stable\_baselines3 repo is installed with the model architecture changed. If you want to do PPO training without the concept-bottleneck layer in the policy network, please create another conda environment simply with the `pip install -e .[extra]` step replaced by `pip install 'stable-baselines3[extra]'`.

- For DynamicObstalces, we provide 3 versions of the set of concepts in the code but we only use `concept_version=3` for the paper.

## Citation

If you find this code useful, please consider citing our paper:

```bibtex
@misc{ye2024conceptbasedinterpretablereinforcementlearning,
      title={Concept-Based Interpretable Reinforcement Learning with Limited to No Human Labels}, 
      author={Zhuorui Ye and Stephanie Milani and Geoffrey J. Gordon and Fei Fang},
      year={2024},
      eprint={2407.15786},
      archivePrefix={arXiv},
      primaryClass={cs.LG},
      url={https://arxiv.org/abs/2407.15786}, 
}
```