# QTrade — Deep Reinforcement Learning for MSFT Stock Trading

INM707 Deep Reinforcement Learning Coursework — City, University of London (MSc Artificial Intelligence, 2025–2026)

This project implements a stock trading agent on Microsoft Corporation (MSFT) daily price data using three reinforcement learning algorithms: Q-learning, Deep Q-Network (DQN) with Double and Dueling improvements, and Proximal Policy Optimisation (PPO). A CartPole-v1 implementation is included as algorithmic validation against Stable Baselines 3.

## Repository Contents

| File | Description |
|---|---|
| `DRL_QTrade_Clean.ipynb` | Main notebook covering Tasks 1–10 |

## Coursework Tasks

**Basic Tasks (Q-learning):**
1. Environment definition and MDP formulation
2. State transitions and reward function
3. Q-learning parameters and exploration policies
4. Base Q-learning training and convergence
5. Parameter sweeps (α, γ, ε-decay, ε-greedy vs. softmax)
6. Quantitative and qualitative analysis

**Advanced Tasks:**
7. DQN implementation with Double DQN and Dueling DQN improvements
8. DQN debugging and analysis (six independent runs, three failure modes, three fixes), plus pre-fix and post-fix ablation studies isolating each improvement
9. PPO implementation on MSFT trading environment + CartPole-v1 validation
10. Three-algorithm comparison, action distribution analysis, and entropy ablation

## Results Summary

| Algorithm | First-100 | Last-100 | Peak | Converged |
|---|---|---|---|---|
| Q-Learning | 183 | 380 | 398 | ~ep 440 |
| DQN (Double + Dueling) | 165 | 397 | 418 | ~ep 1,000 |
| PPO | 182 | 217 | 221 | ~ep 100 |

**CartPole-v1 validation** — both custom PPO (timestep-based) and SB3 PPO achieved smoothed-100 reward of 500.0 (the maximum), confirming the implementation is algorithmically correct.

## Running the Notebook

The notebook is designed for Google Colab with Google Drive integration.

1. Open `DRL_QTrade_Clean.ipynb` in Google Colab
2. Mount Google Drive when prompted (cell 2)
3. The notebook expects `/content/drive/MyDrive/DRL_CW/` as the save directory — figures, weights, and reward arrays are saved there automatically
4. Run cells sequentially; training cells produce reward arrays that downstream visualisation cells consume

GPU is recommended for DQN and PPO training (cells 30+).

## Dependencies

```
numpy
pandas
matplotlib
yfinance
torch
gymnasium
stable-baselines3
```

All available via `pip install` and pre-installed in Google Colab except `stable-baselines3`, which the notebook installs in cell 45.

## Key Implementation Details

- **Environment**: 4-state discretisation of MSFT daily price changes (s0–s3) with 3 actions (hold, buy, sell). Action masking prevents impossible trades (buy when holding, sell when flat).
- **DQN architecture**: Shared 64-unit Tanh trunk → separate value and advantage heads (Dueling). Online network selects target actions, target network evaluates them (Double DQN). Huber (SmoothL1) loss, replay buffer of 20,000 transitions, target network update every 200 steps.
- **PPO architecture**: Actor-Critic with shared 64-unit Tanh trunk. Generalised Advantage Estimation (GAE, λ=0.95). Clipped surrogate objective (ε=0.2). On the CartPole validation, timestep-based rollout collection (N_STEPS=2,048 per update) was required to match SB3's stability — episode-based collection produced policy oscillation due to short early episodes.
- **Reproducibility**: All random seeds (Python, NumPy, PyTorch, CUDA) are fixed at SEED=42 in the setup cell. As documented by Henderson et al. (2018), DRL exhibits inherent run-to-run variance even with fixed seeds; cited results reflect specific training runs.

## Trained Models and Saved Results

All trained model weights, reward arrays, and generated figures are available via the project's Google Drive folder:

**[Google Drive — DRL_CW folder](https://drive.google.com/drive/folders/1R21VYVkNOw2VM_WZ6gpVlIFHk7DrdhwP?usp=sharing)**

This includes:

- **Q-learning**: `q_table_base.npy`, `q_rewards.npy`, `q_rewards_eg.npy`, `q_rewards_softmax.npy`
- **DQN**: `dqn_weights.pth`, `dqn_rewards_fixed.npy`, plus 8 ablation reward arrays (`dqn_ablation_*.npy` for pre-fix, `dqn_ablation_fixed_seeded_*.npy` for post-fix)
- **PPO**: `ppo_weights.pth`, `ppo_rewards.npy`, `ppo_weights_seeded_high_entropy.pth`, `ppo_rewards_seeded_high_entropy.npy`
- **CartPole**: `ppo_cartpole_weights.pth`, `ppo_cartpole_custom_rewards.npy`, `ppo_cartpole_sb3_rewards.npy`
- **Figures**: `fig1_msft_price.png` through `fig16_action_distribution.png` (all report figures)

To run the notebook with these pre-trained models, mount the Drive folder in Colab and the notebook will load existing weights instead of retraining (saves approximately 4 hours of compute).

## Author

**Ibrahim Alqubaisi**
City, University of London — MSc Artificial Intelligence
