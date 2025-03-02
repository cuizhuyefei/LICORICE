import numpy as np
import matplotlib.pyplot as plt
import torch
from stable_baselines3 import PPO  # TODO import our actual alg

def get_concept_activations(model, samples):
    """
    get intermediate concept predictions from the model
    """
    concept_activations = []
    for obs in samples:
        with torch.no_grad():
            obs_tensor = torch.FloatTensor(obs).unsqueeze(0)
            
            # TODO implement the get concept activations method
            concepts = model.policy.get_concepts(obs_tensor)
            concept_activations.append(concepts.numpy())
    
    return np.array(concept_activations).squeeze()

def show_top_concept_activations(model, observations, concept_idx, num_show=5):
    """
    show observations that most strongly activate a specific concept
    """
    concepts = get_concept_activations(model, observations)
    
    # get top activating observations for the concept
    top_indices = np.argsort(concepts[:, concept_idx])[-num_show:][::-1]
    
    # plot results
    fig, axes = plt.subplots(1, num_show, figsize=(15, 3))
    for i, idx in enumerate(top_indices):
        if len(observations[idx].shape) == 3:  # image observation
            axes[i].imshow(observations[idx])
        else:  
            axes[i].bar(range(len(observations[idx])), observations[idx])
        axes[i].set_title(f'Concept {concept_idx}: {concepts[idx, concept_idx]:.3f}')
        axes[i].axis('off' if len(observations[idx].shape) == 3 else 'on')
    
    plt.suptitle(f'Top {num_show} Activating Observations for Concept {concept_idx}')
    plt.tight_layout()
    plt.show()

# Example:
# model = PPO.load("concept_bottleneck_model")
# env = gym.make('YourEnv-v0')
# do some rollouts and get the observations
# show_top_concept_activations(model, env, concept_idx=0)