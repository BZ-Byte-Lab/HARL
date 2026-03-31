# Smart Home Automation via Deep Reinforcement Learning: A Formal Description

## 1. Problem Formulation

This project formulates smart home device control as a **Markov Decision Process (MDP)**, trains a **Deep Q-Network (DQN)** agent to learn optimal control policies from simulated occupant behavior patterns, and subsequently **distills** the learned neural policy into interpretable per-device decision trees.

The system manages $N = 5$ devices:

| Index $i$ | Device Name | Power $p_i$ |
|---|---|---|
| 0 | `living_room_light` | 1.0 |
| 1 | `bedroom_light` | 0.7 |
| 2 | `kitchen_light` | 0.9 |
| 3 | `smart_plug_tv` | 1.5 |
| 4 | `smart_plug_desk_or_coffee` | 1.2 |

> **Code reference**: Device names and power profile are defined in [config.py:5-13](config.py#L5-L13).

---

## 2. MDP Definition

The MDP is defined as a tuple $\mathcal{M} = \langle \mathcal{S}, \mathcal{A}, \mathcal{T}, \mathcal{R}, \gamma \rangle$.

### 2.1 State Space $\mathcal{S}$

The state vector $\mathbf{s}_t \in [0, 1]^{9}$ at time step $t$ is defined as:

$$
\mathbf{s}_t = \left( \tau_t,\ w_t,\ h_t,\ \ell_t,\ d_t^{(0)},\ d_t^{(1)},\ d_t^{(2)},\ d_t^{(3)},\ d_t^{(4)} \right)
$$

where:

| Symbol | Description | Normalization |
|---|---|---|
| $\tau_t$ | Time of day | $\tau_t = \frac{m_t}{1440}$, where $m_t$ is the current minute of day |
| $w_t$ | Day of week | $w_t = \frac{\text{day\_index}}{D - 1}$, where $D = 7$ is episode days |
| $h_t$ | Occupancy (is home) | $h_t \in \{0, 1\}$ |
| $\ell_t$ | Ambient light level | $\ell_t \in [0, 1]$, computed via the light model (Section 3) |
| $d_t^{(i)}$ | Current on/off state of device $i$ | $d_t^{(i)} \in \{0, 1\}$ |

> **Code reference**: The observation vector is constructed in [pattern_env.py:93-105](pattern_env.py#L93-L105). The state dimension `state_dim = 9` is set in [config.py:35](config.py#L35).

### 2.2 Action Space $\mathcal{A}$

Each action $a \in \{0, 1, \ldots, 2^N - 1\}$ is an integer that encodes the joint on/off configuration of all $N$ devices via **bit encoding**:

$$
a = \sum_{i=0}^{N-1} d^{(i)} \cdot 2^{i}
$$

The decoding function extracts the device vector:

$$
d^{(i)} = \left\lfloor \frac{a}{2^i} \right\rfloor \bmod 2, \quad \forall\, i \in \{0, \ldots, N-1\}
$$

This yields $|\mathcal{A}| = 2^5 = 32$ possible actions.

> **Code reference**: Encoding/decoding functions `action_to_device_vector` and `device_vector_to_action` are implemented in [world_state.py:10-18](world_state.py#L10-L18). The `action_dim = 32` is set in [config.py:36](config.py#L36).

### 2.3 Transition Dynamics $\mathcal{T}$

The environment is **deterministic given the world plan**: transitions follow a pre-generated episode trajectory. At each step, the environment advances by $\Delta t = 15$ minutes (configurable as `step_minutes`). Each episode spans 7 days, giving:

$$
T_{\text{episode}} = \frac{1440}{\Delta t} \times D = 96 \times 7 = 672 \text{ steps}
$$

The state transition depends on:
1. The world trajectory advancing to the next step (time, light, occupancy update).
2. The agent's action overwriting the device state vector.

> **Code reference**: Episode length is defined in [config.py:30-33](config.py#L30-L33). The step function is in [pattern_env.py:46-71](pattern_env.py#L46-L71).

### 2.4 Discount Factor

$$
\gamma = 0.99
$$

> **Code reference**: [config.py:38](config.py#L38).

---

## 3. World Simulation Model

### 3.1 Daily Plan Generation

Each day $j \in \{0, \ldots, 6\}$ produces a `DailyPlan` containing stochastic schedule parameters. Key time events are sampled from **truncated Gaussian distributions**:

$$
t_{\text{event}} = \text{clip}\left(\lfloor \mathcal{N}(\mu, \sigma^2) \rceil,\ t_{\min},\ t_{\max}\right)
$$

where $\lfloor \cdot \rceil$ denotes rounding to the nearest integer.

For example, weekday wake time:

$$
t_{\text{wake}}^{\text{weekday}} \sim \text{clip}\left(\lfloor \mathcal{N}(390, 20^2) \rceil,\ 360,\ 450\right) \quad \text{(i.e., } \mu = 6\text{:}30,\ \text{range } [6\text{:}00, 7\text{:}30]\text{)}
$$

Weekend wake time:

$$
t_{\text{wake}}^{\text{weekend}} \sim \text{clip}\left(\lfloor \mathcal{N}(510, 20^2) \rceil,\ 480,\ 540\right) \quad \text{(i.e., } \mu = 8\text{:}30,\ \text{range } [8\text{:}00, 9\text{:}00]\text{)}
$$

Days with index $j \geq 5$ are classified as weekends.

> **Code reference**: The `_jittered_minute` method is at [world_state.py:234-236](world_state.py#L234-L236). Day plan generation is in [world_state.py:81-124](world_state.py#L81-L124).

### 3.2 Occupancy Model

The binary occupancy function $h(m)$ for minute $m$ on a given day is:

$$
h(m) = \begin{cases}
0, & \text{if weekday and } t_{\text{leave}} \leq m < t_{\text{return}} \\
0, & \text{if outing and } t_{\text{outing\_start}} \leq m < t_{\text{outing\_end}} \\
1, & \text{otherwise}
\end{cases}
$$

On weekdays, outings occur with probability $P_{\text{outing}}^{\text{weekday}} = 0.12$. On weekends, outings occur with probability $P_{\text{outing}}^{\text{weekend}} = 0.15$.

> **Code reference**: `_is_home` is at [world_state.py:145-159](world_state.py#L145-L159). Outing probabilities are at [world_state.py:127](world_state.py#L127) and [world_state.py:138](world_state.py#L138).

### 3.3 Ambient Light Model

The natural light level $\ell(m)$ at minute $m$ is modeled as a **raised-sine function** with cloud attenuation and Gaussian noise:

$$
\ell_{\text{base}}(m) = \begin{cases}
\left[\sin\left(\pi \cdot \frac{m - m_{\text{sunrise}}}{m_{\text{sunset}} - m_{\text{sunrise}}}\right)\right]^{1.2}, & \text{if } m_{\text{sunrise}} < m < m_{\text{sunset}} \\
0, & \text{otherwise}
\end{cases}
$$

$$
\ell(m) = \text{clip}\left(\ell_{\text{base}}(m) \cdot c + \epsilon,\ 0,\ 1\right), \quad \epsilon \sim \mathcal{N}(0, 0.03^2)
$$

where $c \sim \text{Uniform}(0.78, 1.00)$ is the daily cloudiness factor (sampled once per day).

> **Code reference**: `_light_level` is at [world_state.py:224-232](world_state.py#L224-L232). Cloudiness is sampled at [world_state.py:108](world_state.py#L108).

### 3.4 Ideal Device Configuration (Ground Truth Labels)

The ideal (expert) device configuration $\mathbf{d}^*(m)$ is a rule-based policy that reflects realistic human habits. The logic is summarized as:

**All devices off when:**
- Not at home ($h = 0$), or
- Before wake ($m < t_{\text{wake}}$), or
- After sleep ($m \geq t_{\text{sleep}}$)

**Device-specific rules (when home and awake):**

| Device $i$ | Condition for $d^{*(i)} = 1$ |
|---|---|
| 0 (living room) | Weekday evening $18\text{:}00 \leq m < t_{\text{wind\_down}}$; Weekend with $\ell < 0.55$; Dark fallback $\ell < 0.25$ |
| 1 (bedroom) | Morning 30 min $t_{\text{wake}} \leq m < t_{\text{wake}} + 30$; Wind-down period |
| 2 (kitchen) | Morning until leave; Weekday return period |
| 3 (TV) | Weekday evening; Weekend TV time window |
| 4 (desk/coffee) | Morning 45 min $t_{\text{wake}} \leq m < t_{\text{wake}} + 45$; Weekend mid-morning if $\ell < 0.60$ |

**Bright override:** If $\ell > 0.85$, all lights (devices 0, 1, 2) are forced off:

$$
d^{*(i)} = 0, \quad \forall\, i \in \{0, 1, 2\} \quad \text{if } \ell > 0.85
$$

**Wind-down mode:** During wind-down period ($t_{\text{wind\_down}} \leq m < t_{\text{sleep}}$), only the bedroom light stays on:

$$
\mathbf{d}^*_{\text{wind\_down}} = (0, 1, 0, 0, 0)
$$

> **Code reference**: The full ideal device logic is in [world_state.py:161-222](world_state.py#L161-L222).

---

## 4. Reward Function

The reward at step $t$ is a **weighted linear combination** of four components:

$$
r_t = w_{\text{habit}} \cdot R_{\text{habit}}(t) + w_{\text{comfort}} \cdot R_{\text{comfort}}(t) + w_{\text{energy}} \cdot R_{\text{energy}}(t) + w_{\text{switch}} \cdot R_{\text{switch}}(t)
$$

with default weights:

| Weight | Value |
|---|---|
| $w_{\text{habit}}$ | 0.50 |
| $w_{\text{comfort}}$ | 0.25 |
| $w_{\text{energy}}$ | 0.15 |
| $w_{\text{switch}}$ | 0.10 |

> **Code reference**: Reward weights are defined in [config.py:17-21](config.py#L17-L21). The reward computation is in [pattern_env.py:51-57](pattern_env.py#L51-L57).

### 4.1 Habit Match Reward $R_{\text{habit}}$

Measures how well the agent's device configuration matches the ideal:

$$
R_{\text{habit}}(t) = 2 \cdot \frac{\sum_{i=0}^{N-1} \mathbb{1}[d_t^{(i)} = d^{*(i)}_t]}{N} - 1
$$

This maps to $[-1, +1]$: a value of $+1$ means perfect match, $-1$ means complete mismatch.

> **Code reference**: [pattern_env.py:108-113](pattern_env.py#L108-L113).

### 4.2 Comfort Reward $R_{\text{comfort}}$

A context-dependent reward based on occupancy and lighting conditions:

**When not home** ($h_t = 0$):

$$
R_{\text{comfort}}(t) = \begin{cases}
1.0, & \text{if } \sum_i d_t^{(i)} = 0 \\
-\frac{\sum_i d_t^{(i)}}{5}, & \text{otherwise}
\end{cases}
$$

**When home** ($h_t = 1$), with $L_{\text{active}} = \sum_{i=0}^{2} d_t^{(i)}$ (number of active lights):

$$
R_{\text{comfort}}(t) = 0.2 + R_{\text{dark}} + R_{\text{bright}} + R_{\text{tv\_dark}}
$$

where:

$$
R_{\text{dark}} = \begin{cases}
+0.8, & \text{if } \ell_t < 0.35 \text{ and } L_{\text{active}} > 0 \\
-1.2, & \text{if } \ell_t < 0.35 \text{ and } L_{\text{active}} = 0 \\
0, & \text{otherwise}
\end{cases}
$$

$$
R_{\text{bright}} = \begin{cases}
-0.25 \cdot L_{\text{active}}, & \text{if } \ell_t > 0.70 \text{ and } L_{\text{active}} > 0 \\
0, & \text{otherwise}
\end{cases}
$$

$$
R_{\text{tv\_dark}} = \begin{cases}
-0.3, & \text{if } \ell_t < 0.45 \text{ and } d_t^{(3)} = 1 \text{ and } L_{\text{active}} = 0 \\
0, & \text{otherwise}
\end{cases}
$$

The final comfort reward is clipped: $R_{\text{comfort}} \in [-1, 1]$.

> **Code reference**: [pattern_env.py:134-150](pattern_env.py#L134-L150). Thresholds `comfort_dark_threshold = 0.35` and `comfort_bright_threshold = 0.70` are in [config.py:53-54](config.py#L53-L54).

### 4.3 Energy Reward $R_{\text{energy}}$

Penalizes total power consumption, normalized by maximum possible consumption:

$$
R_{\text{energy}}(t) = -\frac{\sum_{i=0}^{N-1} p_i \cdot d_t^{(i)}}{\sum_{i=0}^{N-1} p_i}
$$

where $p_i$ is the power coefficient for device $i$. This gives $R_{\text{energy}} \in [-1, 0]$.

> **Code reference**: [pattern_env.py:116-119](pattern_env.py#L116-L119). Power profile is in [config.py:13](config.py#L13).

### 4.4 Switching Penalty $R_{\text{switch}}$

Penalizes frequent toggling of devices between consecutive steps:

$$
R_{\text{switch}}(t) = -\frac{\sum_{i=0}^{N-1} |d_t^{(i)} - d_{t-1}^{(i)}|}{N}
$$

This gives $R_{\text{switch}} \in [-1, 0]$, where $0$ means no devices changed state.

> **Code reference**: [pattern_env.py:120-125](pattern_env.py#L120-L125).

---

## 5. Deep Q-Network (DQN) Agent

### 5.1 Network Architecture

The Q-function is approximated by a multi-layer perceptron (MLP):

$$
Q_\theta : \mathcal{S} \rightarrow \mathbb{R}^{|\mathcal{A}|}
$$

with architecture:

$$
\text{Input}(9) \xrightarrow{\text{Linear}} 128 \xrightarrow{\text{ReLU}} 128 \xrightarrow{\text{ReLU}} 64 \xrightarrow{\text{ReLU}} 32
$$

The network has hidden dimensions $(128, 128, 64)$ and outputs 32 Q-values (one per action).

> **Code reference**: `QNetwork` is defined in [dqn_agent.py:29-42](dqn_agent.py#L29-L42). Hidden dims are in [config.py:48](config.py#L48).

### 5.2 Experience Replay

Transitions $(s_t, a_t, r_t, s_{t+1}, \text{done}_t)$ are stored in a circular replay buffer of capacity $C = 100{,}000$:

$$
\mathcal{D} = \{(s_t, a_t, r_t, s_{t+1}, \text{done}_t)\}_{t=1}^{|\mathcal{D}|}
$$

At each training step, a mini-batch of size $B = 64$ is uniformly sampled from $\mathcal{D}$.

> **Code reference**: `ReplayBuffer` is at [dqn_agent.py:13-26](dqn_agent.py#L13-L26). Capacity and batch size are in [config.py:40-41](config.py#L40-L41).

### 5.3 Target Network

A separate **target network** $Q_{\theta^-}$ is maintained and periodically synchronized:

$$
\theta^{-} \leftarrow \theta \quad \text{every } T_{\text{target}} = 1{,}000 \text{ environment steps}
$$

> **Code reference**: Target update logic is at [dqn_agent.py:103-104](dqn_agent.py#L103-L104). Update interval is in [config.py:42](config.py#L42).

### 5.4 Bellman Update (Loss Function)

For a sampled mini-batch, the temporal difference (TD) target is:

$$
y_t = r_t + \gamma \cdot (1 - \text{done}_t) \cdot \max_{a'} Q_{\theta^-}(s_{t+1}, a')
$$

The loss is the **Smooth L1 (Huber) loss**:

$$
\mathcal{L}(\theta) = \frac{1}{B} \sum_{t \in \text{batch}} \text{SmoothL1}\left(Q_\theta(s_t, a_t) - y_t\right)
$$

where:

$$
\text{SmoothL1}(x) = \begin{cases}
\frac{1}{2} x^2, & \text{if } |x| < 1 \\
|x| - \frac{1}{2}, & \text{otherwise}
\end{cases}
$$

Gradients are clipped to max norm 10.0.

> **Code reference**: The full training step is in [dqn_agent.py:76-106](dqn_agent.py#L76-L106).

### 5.5 Exploration Strategy ($\varepsilon$-Greedy)

The exploration rate decays linearly:

$$
\varepsilon(t) = \varepsilon_{\text{start}} + \min\left(1,\ \frac{t}{T_{\text{decay}}}\right) \cdot (\varepsilon_{\text{end}} - \varepsilon_{\text{start}})
$$

with:
- $\varepsilon_{\text{start}} = 1.0$
- $\varepsilon_{\text{end}} = 0.05$
- $T_{\text{decay}} = 50{,}000$

The action selection policy is:

$$
a_t = \begin{cases}
\text{Uniform}(\mathcal{A}), & \text{with probability } \varepsilon(t) \\
\arg\max_{a} Q_\theta(s_t, a), & \text{with probability } 1 - \varepsilon(t)
\end{cases}
$$

> **Code reference**: Epsilon schedule is at [dqn_agent.py:108-113](dqn_agent.py#L108-L113). Action selection is at [dqn_agent.py:62-70](dqn_agent.py#L62-L70). Decay parameters are in [config.py:44-46](config.py#L44-L46).

### 5.6 Optimization

- **Optimizer**: Adam with learning rate $\alpha = 10^{-4}$
- **Training episodes**: 1,000 (default, 200 via CLI)
- **Steps per episode**: 672

> **Code reference**: Optimizer setup is at [dqn_agent.py:58](dqn_agent.py#L58). Training loop is in [train_pattern.py:49-93](train_pattern.py#L49-L93).

---

## 6. Training Procedure

The training loop for each episode $e \in \{1, \ldots, E\}$:

$$
\boxed{
\begin{aligned}
&\textbf{for } e = 1, \ldots, E: \\
&\quad s_0, \_ \leftarrow \text{env.reset}(\text{seed} = \text{base\_seed} + e) \\
&\quad \textbf{for } t = 0, \ldots, T_{\text{episode}} - 1: \\
&\quad\quad a_t \leftarrow \varepsilon\text{-greedy}(Q_\theta, s_t) \\
&\quad\quad s_{t+1}, r_t, \text{done}_t \leftarrow \text{env.step}(a_t) \\
&\quad\quad \mathcal{D} \leftarrow \mathcal{D} \cup \{(s_t, a_t, r_t, s_{t+1}, \text{done}_t)\} \\
&\quad\quad \theta \leftarrow \theta - \alpha \nabla_\theta \mathcal{L}(\theta) \\
&\quad\quad \text{if } t \bmod T_{\text{target}} = 0: \theta^- \leftarrow \theta \\
\end{aligned}
}
$$

The model is saved as a checkpoint at `artifacts/pattern_dqn.pt` after training.

> **Code reference**: [train_pattern.py:49-93](train_pattern.py#L49-L93).

---

## 7. Evaluation Metrics

During evaluation, the agent runs in **greedy mode** ($\varepsilon = 0$). Three key metrics are computed:

### 7.1 Match Rate

$$
\text{MatchRate} = \frac{\sum_{t=0}^{T-1} \sum_{i=0}^{N-1} \mathbb{1}[d_t^{(i)} = d^{*(i)}_t]}{T \cdot N}
$$

### 7.2 Average Switches Per Step

$$
\text{AvgSwitches} = \frac{\sum_{t=1}^{T-1} \sum_{i=0}^{N-1} |d_t^{(i)} - d_{t-1}^{(i)}|}{T}
$$

### 7.3 Episode Reward

$$
G = \sum_{t=0}^{T-1} r_t
$$

(undiscounted cumulative reward for reporting purposes)

> **Code reference**: Evaluation is implemented in [evaluate_pattern.py:66-121](evaluate_pattern.py#L66-L121).

---

## 8. Policy Distillation into Decision Trees

The final stage distills the DQN's learned policy $\pi_\theta$ into $N = 5$ independent **decision trees** (one per device), making the policy interpretable and deployable without neural network inference.

### 8.1 Dataset Collection

The trained DQN agent is run greedily over $K = 80$ episodes to collect a supervised dataset:

$$
\mathcal{D}_{\text{distill}} = \left\{(\mathbf{s}_t, \pi_\theta(\mathbf{s}_t))\right\}_{t=1}^{K \times T_{\text{episode}}}
$$

yielding $80 \times 672 = 53{,}760$ state-action samples.

The joint action $a_t$ is decoded into per-device binary labels:

$$
y_t^{(i)} = \left\lfloor \frac{a_t}{2^i} \right\rfloor \bmod 2
$$

> **Code reference**: Dataset collection is in [distill_pattern_trees.py:61-84](distill_pattern_trees.py#L61-L84).

### 8.2 Per-Device Decision Tree Training

For each device $i$, a **CART decision tree** $\hat{f}_i$ is trained:

$$
\hat{f}_i : \mathcal{S} \rightarrow \{0, 1\}
$$

with hyperparameters:
- **Maximum depth**: $D_{\max} = 8$
- **Minimum samples per leaf**: 20
- **Split criterion**: Gini impurity (sklearn default)

The Gini impurity at node $m$ is:

$$
G(m) = 1 - \sum_{k=0}^{1} p_k^2
$$

where $p_k$ is the fraction of samples of class $k$ at node $m$. The best split maximizes the impurity reduction:

$$
\Delta G = G(m) - \frac{n_{\text{left}}}{n_m} G(m_{\text{left}}) - \frac{n_{\text{right}}}{n_m} G(m_{\text{right}})
$$

> **Code reference**: Tree training is in [distill_pattern_trees.py:120-132](distill_pattern_trees.py#L120-L132).

### 8.3 Feature Space

The decision trees use the same 9-dimensional feature space as the DQN:

$$
\mathbf{x} = (\underbrace{\tau, w, h, \ell}_{\text{environment context}},\ \underbrace{d^{(0)}, d^{(1)}, d^{(2)}, d^{(3)}, d^{(4)}}_{\text{current device states}})
$$

Feature names with human-readable mappings:

| Feature | Human Name |
|---|---|
| `time_of_day` | time of day |
| `day_of_week` | day of week |
| `is_home` | occupancy state |
| `light_level` | ambient light level |
| `*_current` | current state of respective device |

> **Code reference**: Feature names and humanization are at [distill_pattern_trees.py:16-41](distill_pattern_trees.py#L16-L41).

### 8.4 Distillation Results

The distilled trees achieve high fidelity to the DQN policy:

| Device | Accuracy | Positive Rate | Depth | Leaves |
|---|---|---|---|---|
| `living_room_light` | 0.9808 | 0.2087 | 3 | 8 |
| `bedroom_light` | 0.9783 | 0.2019 | 3 | 7 |
| `kitchen_light` | 0.9706 | 0.2533 | 3 | 7 |
| `smart_plug_tv` | 0.9895 | 0.1185 | 3 | 8 |
| `smart_plug_desk_or_coffee` | 0.9949 | 0.0136 | 3 | 8 |

Despite a max depth of 8, all trees converged to depth 3, indicating the DQN learned relatively simple decision boundaries.

> **Code reference**: Summary output is at [artifacts/decision_trees/summary.txt](artifacts/decision_trees/summary.txt). Report generation is in [distill_pattern_trees.py:135-161](distill_pattern_trees.py#L135-L161).

### 8.5 Natural Language Rule Extraction

Each decision tree is traversed recursively to produce human-readable if-then rules. For a leaf node reached via a path of conditions $\{c_1, c_2, \ldots, c_k\}$:

$$
\text{If } c_1 \wedge c_2 \wedge \cdots \wedge c_k, \text{ then } \begin{cases} \text{turn on device } i, & \text{if } \hat{y} = 1 \\ \text{turn off device } i, & \text{if } \hat{y} = 0 \end{cases}
$$

Thresholds are converted to human-readable formats:
- Time of day: $\tau \rightarrow \lfloor \tau \times 1440 \rfloor$ converted to HH:MM
- Day of week: $w \rightarrow$ day name via index $\lfloor w \times 6 \rceil$
- Binary features: threshold $\geq 0.5 \rightarrow$ "on/home", $< 0.5 \rightarrow$ "off/away"

> **Code reference**: Rule extraction is in [distill_pattern_trees.py:87-117](distill_pattern_trees.py#L87-L117). Threshold formatting is at [distill_pattern_trees.py:44-58](distill_pattern_trees.py#L44-L58).

---

## 9. System Pipeline Overview

The end-to-end pipeline can be expressed as:

$$
\underbrace{\text{WorldSim}}_{\text{world\_state.py}} \xrightarrow{\text{episodes}} \underbrace{\text{Gym Env}}_{\text{pattern\_env.py}} \xrightarrow{\text{train}} \underbrace{\text{DQN}}_{\text{dqn\_agent.py}} \xrightarrow{\text{distill}} \underbrace{\text{Decision Trees}}_{\text{distill\_pattern\_trees.py}}
$$

| Stage | Script | Input | Output |
|---|---|---|---|
| 1. Training | [train_pattern.py](train_pattern.py) | Config hyperparameters | `artifacts/pattern_dqn.pt` |
| 2. Evaluation | [evaluate_pattern.py](evaluate_pattern.py) | Checkpoint | Metrics (match rate, reward) |
| 3. Distillation | [distill_pattern_trees.py](distill_pattern_trees.py) | Checkpoint | Per-device decision tree reports |

---

## 10. Hyperparameter Summary

$$
\begin{array}{|l|l|l|}
\hline
\textbf{Parameter} & \textbf{Symbol} & \textbf{Value} \\
\hline
\text{Number of devices} & N & 5 \\
\text{Time step interval} & \Delta t & 15 \text{ min} \\
\text{Steps per day} & T_{\text{day}} & 96 \\
\text{Episode days} & D & 7 \\
\text{Episode steps} & T & 672 \\
\text{State dimension} & |\mathcal{S}| & 9 \\
\text{Action dimension} & |\mathcal{A}| & 32 \\
\text{Discount factor} & \gamma & 0.99 \\
\text{Learning rate} & \alpha & 10^{-4} \\
\text{Batch size} & B & 64 \\
\text{Replay capacity} & C & 100{,}000 \\
\text{Target update interval} & T_{\text{target}} & 1{,}000 \\
\text{Epsilon start} & \varepsilon_{\text{start}} & 1.0 \\
\text{Epsilon end} & \varepsilon_{\text{end}} & 0.05 \\
\text{Epsilon decay steps} & T_{\text{decay}} & 50{,}000 \\
\text{Hidden dimensions} & - & (128, 128, 64) \\
\text{Training episodes} & E & 1{,}000 \\
\text{Gradient clip norm} & - & 10.0 \\
\text{Decision tree max depth} & D_{\max} & 8 \\
\text{Decision tree min leaf samples} & - & 20 \\
\hline
\end{array}
$$

> **Code reference**: All hyperparameters are centralized in [config.py:16-57](config.py#L16-L57).
