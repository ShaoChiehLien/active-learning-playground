# Active Learning and Grid World

Comparing two Active Learning (AL) strategies applied to Q-learning in a simulated energy grid management problem. Each strategy is evaluated on a 10×10 and a 50×50 grid.

## Problem Setting

The GridWorld simulates a battery storage and transformer load controller. Each state is a `(battery level, transformer load)` pair. Reaching the goal region (battery 70–80%, load 30–60%) yields a large positive reward; hitting a fault zone ends the episode with a large penalty.

The agent has five actions:

| Action | Meaning | Step reward |
|---|---|---|
| ↑ Up | EV charges via its own solar panel; battery level rises | −1 |
| ↓ Down | EV runs on battery with no charging; battery level falls | −1 |
| → Right | Transformer diverts to support other loads, ignoring the EV; transformer load rises | −1 |
| ← Left | Transformer sheds load; transformer load falls | −1 |
| ↗ Up-right | Transformer actively charges the EV; battery rises and load rises | 0 |

Up-right carries no step penalty because it is the productive charging action the system is designed for.

The image below shows learned policies on a 50×50 grid after training — left is the standard Q-learning baseline (random start), right is the AL agent (QBC start only). Arrows indicate the greedy action at each state; the green rectangle is the goal region and red areas are fault zones.

![Learned policy comparison: standard vs. AL Q-learning, Variant A, 50×50 grid](docs/policy-comparison-variant-A-50x50.png)

## Variants

The four notebooks vary along two axes: AL approach and grid size.

### Variant A — QBC Start State Only

A committee of 5 independent Q-tables votes on the best action at every non-wall state. Each episode begins at the state where the committee disagrees most, measured by **vote entropy**. Action selection during exploration is uniform random for both the baseline and AL agent — the only AL difference is the episode start state.

| | Baseline | AL |
|---|---|---|
| **Episode start** | Uniformly random non-wall cell | State with max committee vote entropy |
| **Exploration action** | Uniform random | Uniform random (same as baseline) |

### Variant B — Uncertainty-Guided Action Only

No committee. The episode start state is uniformly random for both agents. During exploration, the AL agent samples actions proportionally to the uncertainty of the resulting next state rather than uniformly at random.

**Uncertainty score per cell:**

$$u(s) = \sigma_a\bigl[Q(s,a)\bigr] + \frac{0.5}{1 + \text{visits}(s)}$$

where $\sigma_a[Q(s,a)]$ is the standard deviation of Q-values across actions at state $s$, and the second term is a visit bonus that decays as the cell is explored.

**Action probability during exploration:**

$$P(a \mid s) = \frac{u\bigl(\text{next}(s,a)\bigr) + \varepsilon}{\sum_{a'} u\bigl(\text{next}(s,a')\bigr) + \varepsilon}$$

The agent samples the action whose resulting neighbour has the highest uncertainty.

| | Baseline | AL |
|---|---|---|
| **Episode start** | Uniformly random non-wall cell | Uniformly random non-wall cell (same as baseline) |
| **Exploration action** | Uniform random | Sampled ∝ uncertainty of next state |

## Design Choices

### 1. Allow four cardinal movement directions (↑, ↓, →, ←)

The first version of the environment only allowed diagonal movement (↗ up-right and ↙ down-left), similar to how a bishop moves in chess. This meant that only states lying on the same diagonal as the goal could ever reach it — agents starting outside that diagonal were permanently blocked from earning any positive reward, causing most episodes to end with large penalties. Adding the four cardinal directions gives every state a path to the goal while keeping the semantics of each action grounded in the physical system.

### 2. Use 50×50 grids in addition to 10×10

On a 10×10 grid the exploration problem is easy enough that the random baseline can stumble onto a good route faster than AL converges. This makes AL look worse despite being a stronger strategy in harder settings. The 50×50 grid has 25× more states on the same episode budget, so the exploration advantage of AL is large enough to show up clearly in the results.

## Findings & Lessons Learned

### 1. Training reward diverges for AL Variant A yet evaluation still favours AL

Because the QBC start strategy deliberately places the agent in the states it is most uncertain about, the agent spends training in genuinely hard situations and accumulates lower per-episode rewards than the baseline. In evaluation (where both agents start from the same fixed states at the same environment snapshot), however, AL outperforms the baseline — a reminder that a dropping or noisy training curve does not always indicate a worse policy, especially when the training distribution is intentionally skewed toward difficulty.

### 2. Variant B AL takes longer to overtake the baseline

The uncertainty-guided action selection imposes an exploration tax: the AL agent is repeatedly directed toward uncertain, often suboptimal states, delaying the accumulation of high-quality Q-value estimates needed to exploit a good policy. Because this tax is paid on every exploration step rather than only at episode start (as in Variant A), the performance crossover with the baseline happens later in training.

## Files

| File | AL approach | Grid |
|---|---|---|
| `grid-world-AL-poc-variant-A-10x10.ipynb` | QBC start state | 10×10 |
| `grid-world-AL-poc-variant-A-50x50.ipynb` | QBC start state | 50×50 |
| `grid-world-AL-poc-variant-B-10x10.ipynb` | Uncertainty action | 10×10 |
| `grid-world-AL-poc-variant-B-50x50.ipynb` | Uncertainty action | 50×50 |

The 50×50 grid has 25× more states than the 10×10 grid but uses the same episode budget, making it a harder exploration problem.

## Setup

```bash
pip install -r requirements.txt
```

Open any of the four notebooks and run all cells.
