# QTrade — Deep Reinforcement Learning for MSFT Stock Trading

INM707 Deep Reinforcement Learning Coursework — City, University of London (MSc Artificial Intelligence, 2025–2026)

This project implements a stock trading agent on Microsoft Corporation (MSFT) daily price data using three reinforcement learning algorithms: Q-learning, Deep Q-Network (DQN) with Double and Dueling improvements, and Proximal Policy Optimisation (PPO). A CartPole-v1 implementation is included as algorithmic validation against Stable Baselines 3.

The aim of the project is to understand how reinforcement learning algorithms behave on a sequential decision-making problem and to identify the structural choices (reward function, state representation, algorithm class) that determine their success or failure — not to develop a deployment-ready profitable trading system.

## Repository Contents

| File | Description |
|---|---|
| `DRL_QTrade_Clean.ipynb` | Main notebook covering Tasks 1–10 plus three additional experiments |
| `INM707_Report.tex` | LaTeX source of the report (IEEE two-column format) |
| `README.md` | This file |

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

**Additional Experiments (Sections VI.C, VIII.E, X.G of the report):**

- **Reward design experiment** — Q-learning retrained with a P&L-based reward (actual portfolio change per timestep) instead of the prescriptive R[s,a] matrix; held-out evaluation against the prescriptive agent and a buy-and-hold benchmark on 2014–2017 test data.
- **State representation experiment** — DQN retrained with a 5-feature continuous state (price change, holding, 5-day momentum, 20-day volatility, price-vs-MA50) instead of the 2-feature discrete state, isolating the contribution of state representation alone.
- **Multi-seed reproducibility analysis** — All three algorithms re-trained across multiple random seeds (Q-learning ×5, PPO ×3, DQN ×3) following Henderson et al. (2018) recommendations for DRL reporting; standard deviation reported alongside mean for each headline metric.

## Results Summary

### Headline performance (single-run, training set)

| Algorithm | First-100 | Last-100 | Peak | Converged |
|---|---|---|---|---|
| Q-Learning | 183 | 380 | 398 | ~ep 440 |
| DQN (Double + Dueling) | 165 | 397 | 418 | ~ep 1,000 |
| PPO | 182 | 217 | 221 | ~ep 100 |

### Multi-seed reproducibility (Last-100 reward, mean ± std)

| Algorithm | Seeds | Mean | Std | Range |
|---|---|---|---|---|
| Q-Learning | 5 | 379.26 | **1.10** | 377.90 – 380.83 |
| PPO | 3 | 208.37 | **3.20** | 205.61 – 212.86 |
| DQN (post-fix) | 3 | 404.27 | **6.15** | 397.28 – 412.25 |

Tabular Q-learning is essentially deterministic (CV ≈ 0.3%); PPO is reproducible (CV ≈ 1.5%) despite being deep RL; DQN is the most variable (CV ≈ 1.5% on Last-100 with the widest range), but post-fix variance is far smaller than the pre-fix 6-run investigation that motivated the Section VIII.B fixes.

### Reward design experiment (out-of-sample test, 2014–2017)

| Strategy | Return | Trades | Win rate |
|---|---|---|---|
| Buy-and-hold benchmark | **+155.17%** | 1 | – |
| Prescriptive Q-learning | +82.84% | 354 | 70.1% |
| P&L Q-learning | 0.00% | 0 | – |

Both RL agents underperform the passive benchmark, demonstrating that reward design and state representation jointly impose the project's structural ceiling on profitable trading.

### State representation experiment

| Configuration | Last-100 | Peak |
|---|---|---|
| DQN, 2-state discrete | 398 | 426 |
| DQN, 5-feature continuous | **440 (+10.6%)** | 467 |

### CartPole-v1 validation

Both custom PPO (timestep-based) and SB3 PPO achieved smoothed-100 reward of 500.0 (the maximum), confirming the implementation is algorithmically correct.

## Running the Notebook

The notebook is designed for Google Colab with Google Drive integration.

1. Open `DRL_QTrade_Clean.ipynb` in Google Colab
2. Mount Google Drive when prompted (cell 2)
3. The notebook expects `/content/drive/MyDrive/DRL_CW/` as the save directory — figures, weights, and reward arrays are saved there automatically
4. Run cells sequentially; training cells produce reward arrays that downstream visualisation cells consume

GPU is recommended for DQN and PPO training cells. Total compute time for a full run from scratch is approximately 6–8 hours; loading pre-trained weights from the Drive folder (linked below) reduces this to under 30 minutes.

## Dependencies

```
numpy
pandas
matplotlib
seaborn
yfinance
torch
gymnasium
stable-baselines3
```

All available via `pip install` and pre-installed in Google Colab except `stable-baselines3`, which the notebook installs at the start of the CartPole validation cell.

## Key Implementation Details

- **Environment**: 4-state discretisation of MSFT daily price changes (s0–s3) with 3 actions (hold, buy, sell). Action masking prevents impossible trades (buy when holding, sell when flat). Training data is the most recent 1,500 trading days from 2018–2023; test data is 2014–2017 (1,007 days, never seen during training).
- **DQN architecture**: Shared 64-unit Tanh trunk → separate value and advantage heads (Dueling). Online network selects target actions, target network evaluates them (Double DQN). Huber (SmoothL1) loss, replay buffer of 20,000 transitions, target network update every 200 steps. Action masking is applied consistently to both online action selection and target Q-value computation.
- **PPO architecture**: Actor-Critic with shared 64-unit Tanh trunk. Generalised Advantage Estimation (GAE, λ=0.95). Clipped surrogate objective (ε=0.2). On the CartPole validation, timestep-based rollout collection (N_STEPS=2,048 per update) was required to match SB3's convergence — episode-based collection produced policy oscillation due to short early episodes.
- **Reproducibility**: All random seeds (Python, NumPy, PyTorch, CUDA) are fixed at SEED=42 in the setup cell. The multi-seed reproducibility analysis (results table above) validates that headline numbers are robust to random seed selection within each algorithm class.

## Trained Models and Saved Results

All trained model weights, reward arrays, and generated figures are available via the project's Google Drive folder:

**[Google Drive — DRL_CW folder](PASTE_YOUR_DRIVE_LINK_HERE)**

This includes:

- **Q-learning**: `q_table_base.npy`, `q_rewards.npy`, plus parameter sweep arrays
- **Q-learning (P&L reward)**: `q_table_pnl.npy`, `q_rewards_pnl.npy`
- **DQN (original)**: `dqn_weights.pth`, `dqn_rewards_fixed.npy`, plus 8 ablation reward arrays
- **DQN (rich-state, 5-feature)**: `dqn_weights_rich_state.pth`, `dqn_rewards_rich_state.npy`
- **PPO**: `ppo_weights.pth`, `ppo_rewards.npy`, plus high-entropy variant
- **CartPole**: `ppo_cartpole_weights.pth`, `ppo_cartpole_custom_rewards.npy`, `ppo_cartpole_sb3_rewards.npy`
- **Multi-seed arrays**: `qlearning_multiseed.npy`, `ppo_multiseed.npy`, `dqn_multiseed.npy`
- **Figures**: `fig1_msft_price.png` through `fig22_dqn_multiseed.png` (all report figures)

To run the notebook with these pre-trained models, mount the Drive folder in Colab and the notebook will load existing weights instead of retraining (saves approximately 4 hours of compute).

## References

Key references cited in the report:

- Mnih et al., 2015 — Deep Q-Networks (DQN)
- van Hasselt et al., 2016 — Double DQN
- Wang et al., 2016 — Dueling DQN
- Schulman et al., 2017 — Proximal Policy Optimisation (PPO)
- Schulman et al., 2016 — Generalised Advantage Estimation (GAE)
- Henderson et al., 2018 — Deep RL reproducibility
- Kabbani & Duman, 2022 — DRL trading with technical indicators
- Zhang et al., 2019 — DRL for trading with volatility scaling
- Liu et al., 2020 — FinRL library

Full bibliography in the report.

## Author

**Ibrahim Alqubaisi**
City, University of London — MSc Artificial Intelligence
