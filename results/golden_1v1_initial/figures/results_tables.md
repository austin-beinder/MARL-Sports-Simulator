### Keep-away 1v1, PyBullet, survival only (5 runs; mean episode length, control steps)

| policy | seeds | overall length | possession starts | free-ball starts | fraction truncated | free-ball races won (0-1) | closing speed (m/step) |
|---|---|---|---|---|---|---|---|
| ppo | 5 | 123.47 ± 1.96 | 142.34 ± 3.08 | 97.41 ± 0.90 | 0.00 ± 0.00 | 0.07 ± 0.04 | 0.0080 ± 0.0030 |
| ppo_deterministic | 5 | 123.18 ± 1.76 | 142.92 ± 3.09 | 95.92 ± 0.22 | 0.00 ± 0.00 | 0.02 ± 0.03 | 0.0063 ± 0.0042 |
| untrained | 5 | 87.40 ± 0.19 | 81.11 ± 0.39 | 96.09 ± 0.35 | 0.00 ± 0.00 | 0.01 ± 0.02 | 0.0008 ± 0.0003 |
| random | 1 | 86.76 | 80.34 | 95.62 | 0.00 | 0.05 | 0.0011 |
| still | 1 | 86.90 | 80.38 | 95.90 | 0.00 | 0.00 | 0.0002 |
| flee | 1 | 145.46 | 151.17 | 137.57 | 0.00 | 1.00 | 0.0982 |

### Keep-away 1v1, PyBullet, + PBRS acquisition shaping, coef 4 (5 runs; mean episode length, control steps)

| policy | seeds | overall length | possession starts | free-ball starts | fraction truncated | free-ball races won (0-1) | closing speed (m/step) |
|---|---|---|---|---|---|---|---|
| ppo | 5 | 126.62 ± 3.61 | 146.83 ± 4.90 | 98.70 ± 2.64 | 0.00 ± 0.00 | 0.10 ± 0.09 | 0.0266 ± 0.0095 |
| ppo_deterministic | 5 | 127.48 ± 3.85 | 148.12 ± 5.13 | 98.97 ± 3.18 | 0.00 ± 0.00 | 0.09 ± 0.09 | 0.0272 ± 0.0112 |
| untrained | 5 | 87.40 ± 0.19 | 81.11 ± 0.39 | 96.09 ± 0.35 | 0.00 ± 0.00 | 0.01 ± 0.02 | 0.0008 ± 0.0003 |
| random | 1 | 86.76 | 80.34 | 95.62 | 0.00 | 0.05 | 0.0011 |
| still | 1 | 86.90 | 80.38 | 95.90 | 0.00 | 0.00 | 0.0002 |
| flee | 1 | 145.46 | 151.17 | 137.57 | 0.00 | 1.00 | 0.0982 |
