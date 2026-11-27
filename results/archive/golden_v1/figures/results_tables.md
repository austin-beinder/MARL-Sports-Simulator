### Keep-away (5 runs; mean episode length, control steps)

| policy | seeds | overall length | possession starts | free-ball starts | fraction truncated | free-ball races won (0-1) | closing speed (m/step) |
|---|---|---|---|---|---|---|---|
| ppo | 5 | 122.10 ± 4.27 | 142.34 ± 6.79 | 94.15 ± 1.88 | 0.00 ± 0.00 | 0.04 ± 0.04 | 0.0076 ± 0.0031 |
| ppo_deterministic | 5 | 122.90 ± 3.99 | 143.69 ± 7.12 | 94.18 ± 1.18 | 0.00 ± 0.00 | 0.04 ± 0.02 | 0.0070 ± 0.0036 |
| untrained | 5 | 85.44 ± 0.17 | 79.90 ± 0.18 | 93.09 ± 0.25 | 0.00 ± 0.00 | 0.05 ± 0.00 | 0.0004 ± 0.0007 |
| random | 1 | 85.46 | 80.14 | 92.81 | 0.00 | 0.05 | 0.0006 |
| still | 1 | 85.12 | 79.31 | 93.14 | 0.00 | 0.00 | 0.0000 |
| flee | 1 | 144.12 | 149.38 | 136.86 | 0.00 | 1.00 | 0.1000 |

### Keep-away + PBRS acquisition shaping, coef 4 (5 runs; mean episode length, control steps)

| policy | seeds | overall length | possession starts | free-ball starts | fraction truncated | free-ball races won (0-1) | closing speed (m/step) |
|---|---|---|---|---|---|---|---|
| ppo | 5 | 127.34 ± 1.98 | 148.22 ± 2.40 | 98.50 ± 3.28 | 0.00 ± 0.00 | 0.15 ± 0.06 | 0.0364 ± 0.0050 |
| ppo_deterministic | 5 | 127.84 ± 1.33 | 149.61 ± 1.61 | 97.79 ± 2.62 | 0.00 ± 0.00 | 0.14 ± 0.08 | 0.0362 ± 0.0041 |
| untrained | 5 | 85.44 ± 0.17 | 79.90 ± 0.18 | 93.09 ± 0.25 | 0.00 ± 0.00 | 0.05 ± 0.00 | 0.0004 ± 0.0007 |
| random | 1 | 85.46 | 80.14 | 92.81 | 0.00 | 0.05 | 0.0006 |
| still | 1 | 85.12 | 79.31 | 93.14 | 0.00 | 0.00 | 0.0000 |
| flee | 1 | 144.12 | 149.38 | 136.86 | 0.00 | 1.00 | 0.1000 |

### PBRS coefficient pilot (PPO, stochastic eval; lengths in control steps)

| PBRS coef | seeds | closing speed (m/step) | alignment (cos) | races won (0-1) | possession starts | free-ball starts | env steps |
|---|---|---|---|---|---|---|---|
| 0 (survival only) | 5 | 0.0079 ± 0.0044 | 0.10 ± 0.07 | 0.13 ± 0.04 | 132.93 ± 6.45 | 95.16 ± 1.41 | 204800 ± 0 |
| 1 | 2 | 0.0115 ± 0.0034 | 0.16 ± 0.05 | 0.07 ± 0.03 | 131.14 ± 1.17 | 95.55 ± 0.17 | 198656 ± 0 |
| 2 | 2 | 0.0175 ± 0.0035 | 0.23 ± 0.04 | 0.07 ± 0.03 | 130.91 ± 2.37 | 94.43 ± 0.20 | 198656 ± 0 |
| 4 | 2 | 0.0227 ± 0.0031 | 0.28 ± 0.05 | 0.19 ± 0.07 | 132.81 ± 1.05 | 96.02 ± 2.26 | 198656 ± 0 |
| 8 | 2 | 0.0199 ± 0.0137 | 0.27 ± 0.17 | 0.12 ± 0.03 | 124.93 ± 1.85 | 95.90 ± 1.21 | 198656 ± 0 |
| 16 | 2 | 0.0260 ± 0.0000 | 0.35 ± 0.03 | 0.00 ± 0.00 | 119.66 ± 1.12 | 93.14 ± 0.00 | 198656 ± 0 |

### goto_target (3 runs; mean reward = minus mean distance, m)

| policy | seeds | mean reward (-m) |
|---|---|---|
| ppo | 3 | -1.98 ± 0.37 |
| ppo_deterministic | 3 | -1.90 ± 0.39 |
| untrained | 3 | -6.57 ± 0.01 |
| random | 1 | -6.54 |
| still | 1 | -6.54 |
| oracle | 1 | -0.90 |
