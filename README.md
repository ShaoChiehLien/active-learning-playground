# Active Learning and Grid

Applying Query by Committee (QBC) active learning to Q-learning episode start-state selection in a simulated energy grid management problem.

## The Core Idea

A committee of 5 Q-tables votes on the greedy action for every non-wall state. Each training episode begins at the state where the committee disagrees most — measured by vote entropy — rather than a random start. During exploration, action selection is also uncertainty-guided, preferring transitions into states where Q-values vary most.

## Domain

The GridWorld simulates a battery storage and transformer load controller. Each state is a `(battery level, transformer load)` pair. The agent's four actions adjust battery level or transformer load by one grid step. Reaching the goal region (safe operating range) yields +500; hitting a fault zone ends the episode with -10.

## Variants

| | Variant A | Variant D |
|---|---|---|
| **Grid** | 50×50 | 50×50 |
| **Fault zones** | Single wall — load > 90% | 7 scattered fault patches (inverter harmonics, tap-changer thermal, feedline resonance, aging capacitors, etc.) |
| **Goal region** | ~30% of non-wall cells | ~3.5% of non-wall cells |
| **Difficulty** | Simple | Hard |

## Setup

```bash
pip install -r requirements.txt
```

Open `grid-world-AL-poc-variant-A.ipynb` or `grid-world-AL-poc-variant-D.ipynb` and run all cells.
