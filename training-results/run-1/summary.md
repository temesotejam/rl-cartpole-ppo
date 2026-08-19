# Cart-Pole PPO training result

- Preset: `normal`
- Seed: `42`
- Best checkpoint: `100_percent`
- Best mean return: `495.94`

## Evaluation checkpoints

| Stage | Timesteps | Mean return | Survival | Success | RMS angle | RMS cart x | RMS force |
|---|---:|---:|---:|---:|---:|---:|---:|
| random | 0 | 5.47 ± 4.53 | 0.61s | 0.0% | 15.01° | 0.163m | 2.36N |
| 25_percent | 28,672 | 236.54 ± 104.22 | 6.38s | 10.0% | 1.59° | 1.006m | 0.62N |
| 50_percent | 53,248 | 405.24 ± 102.03 | 9.40s | 80.0% | 0.67° | 0.677m | 0.64N |
| 75_percent | 77,824 | 492.99 ± 7.81 | 10.00s | 100.0% | 0.61° | 0.186m | 0.75N |
| 100_percent | 102,400 | 495.94 ± 3.62 | 10.00s | 100.0% | 0.65° | 0.137m | 0.76N |

## How to read this

A successful policy should survive the full 10 s episode, keep the pole angle small, keep the cart near the center, and avoid unnecessarily large motor force.
