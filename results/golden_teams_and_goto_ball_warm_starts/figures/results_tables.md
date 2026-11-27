### Keep-away 1v1 (PyBullet, kicking), from scratch (5 runs; mean episode length, control steps)

| policy | seeds | overall length | possession starts | free-ball starts | fraction truncated | free-ball races won (0-1) | closing speed (m/step) |
|---|---|---|---|---|---|---|---|
| ppo | 5 | 112.91 ± 1.90 | 123.25 ± 3.45 | 98.64 ± 2.48 | 0.00 ± 0.00 | 0.09 ± 0.08 | 0.0122 ± 0.0113 |
| ppo_deterministic | 5 | 111.72 ± 1.37 | 121.52 ± 1.56 | 98.20 ± 2.31 | 0.00 ± 0.00 | 0.08 ± 0.11 | 0.0121 ± 0.0115 |
| untrained | 5 | 90.21 ± 1.05 | 85.92 ± 1.77 | 96.13 ± 0.19 | 0.00 ± 0.00 | 0.00 ± 0.00 | 0.0007 ± 0.0006 |
| random | 1 | 86.76 | 80.34 | 95.62 | 0.00 | 0.05 | 0.0011 |
| still | 1 | 86.90 | 80.38 | 95.90 | 0.00 | 0.00 | 0.0002 |
| flee | 1 | 145.46 | 151.17 | 137.57 | 0.00 | 1.00 | 0.0982 |

### Keep-away 1v1 (PyBullet, kicking), warm-started from goto_target (5 runs; mean episode length, control steps)

| policy | seeds | overall length | possession starts | free-ball starts | fraction truncated | free-ball races won (0-1) | closing speed (m/step) |
|---|---|---|---|---|---|---|---|
| ppo | 5 | 180.83 ± 18.53 | 193.68 ± 19.69 | 163.09 ± 18.35 | 0.22 ± 0.16 | 0.88 ± 0.11 | 0.0764 ± 0.0062 |
| ppo_deterministic | 5 | 203.41 ± 33.76 | 206.86 ± 24.79 | 198.64 ± 50.59 | 0.43 ± 0.22 | 0.88 ± 0.10 | 0.0793 ± 0.0084 |
| untrained | 5 | 90.21 ± 1.05 | 85.92 ± 1.77 | 96.13 ± 0.19 | 0.00 ± 0.00 | 0.00 ± 0.00 | 0.0007 ± 0.0006 |
| random | 1 | 86.76 | 80.34 | 95.62 | 0.00 | 0.05 | 0.0011 |
| still | 1 | 86.90 | 80.38 | 95.90 | 0.00 | 0.00 | 0.0002 |
| flee | 1 | 145.46 | 151.17 | 137.57 | 0.00 | 1.00 | 0.0982 |

### Keep-away 2v1 (PyBullet, kicking), from scratch (5 runs; mean episode length, control steps)

| policy | seeds | overall length | possession starts | free-ball starts | fraction truncated | free-ball races won (0-1) | closing speed (m/step) |
|---|---|---|---|---|---|---|---|
| ppo | 5 | 103.88 ± 1.73 | 113.05 ± 1.80 | 91.23 ± 1.78 | 0.00 ± 0.00 | 0.04 ± 0.04 | 0.0073 ± 0.0097 |
| ppo_deterministic | 5 | 101.89 ± 2.15 | 109.74 ± 2.60 | 91.05 ± 1.69 | 0.00 ± 0.00 | 0.04 ± 0.04 | 0.0074 ± 0.0103 |
| untrained | 5 | 81.43 ± 1.37 | 75.57 ± 2.51 | 89.51 ± 0.22 | 0.00 ± 0.00 | 0.03 ± 0.03 | -0.0005 ± 0.0001 |
| random | 1 | 82.14 | 76.76 | 89.57 | 0.00 | 0.00 | -0.0015 |
| still | 1 | 81.92 | 76.52 | 89.38 | 0.00 | 0.00 | 0.0000 |
| flee | 1 | 150.76 | 148.17 | 154.33 | 0.00 | 1.00 | 0.0982 |

### Keep-away 2v1 (PyBullet, kicking), warm-started from goto_target (5 runs; mean episode length, control steps)

| policy | seeds | overall length | possession starts | free-ball starts | fraction truncated | free-ball races won (0-1) | closing speed (m/step) |
|---|---|---|---|---|---|---|---|
| ppo | 5 | 132.58 ± 9.08 | 135.10 ± 6.37 | 129.09 ± 13.66 | 0.01 ± 0.02 | 0.69 ± 0.23 | 0.0577 ± 0.0102 |
| ppo_deterministic | 5 | 132.15 ± 12.51 | 139.94 ± 11.34 | 121.38 ± 16.91 | 0.02 ± 0.03 | 0.66 ± 0.19 | 0.0559 ± 0.0110 |
| untrained | 5 | 81.43 ± 1.37 | 75.57 ± 2.51 | 89.51 ± 0.22 | 0.00 ± 0.00 | 0.03 ± 0.03 | -0.0005 ± 0.0001 |
| random | 1 | 82.14 | 76.76 | 89.57 | 0.00 | 0.00 | -0.0015 |
| still | 1 | 81.92 | 76.52 | 89.38 | 0.00 | 0.00 | 0.0000 |
| flee | 1 | 150.76 | 148.17 | 154.33 | 0.00 | 1.00 | 0.0982 |

### Keep-away 2v1 (PyBullet, no kicking), from scratch (5 runs; mean episode length, control steps)

| policy | seeds | overall length | possession starts | free-ball starts | fraction truncated | free-ball races won (0-1) | closing speed (m/step) |
|---|---|---|---|---|---|---|---|
| ppo | 5 | 109.70 ± 0.71 | 122.90 ± 2.74 | 91.48 ± 2.76 | 0.00 ± 0.00 | 0.03 ± 0.04 | 0.0041 ± 0.0033 |
| ppo_deterministic | 5 | 109.30 ± 1.45 | 123.82 ± 2.53 | 89.25 ± 0.24 | 0.00 ± 0.00 | 0.00 ± 0.00 | 0.0040 ± 0.0040 |
| untrained | 5 | 81.82 ± 0.36 | 76.44 ± 0.46 | 89.25 ± 0.45 | 0.00 ± 0.00 | 0.06 ± 0.02 | 0.0010 ± 0.0014 |
| random | 1 | 82.14 | 76.76 | 89.57 | 0.00 | 0.00 | -0.0015 |
| still | 1 | 81.92 | 76.52 | 89.38 | 0.00 | 0.00 | 0.0000 |
| flee | 1 | 150.76 | 148.17 | 154.33 | 0.00 | 1.00 | 0.0982 |

### Keep-away 2v2 (PyBullet, kicking), from scratch (5 runs; mean episode length, control steps)

| policy | seeds | overall length | possession starts | free-ball starts | fraction truncated | free-ball races won (0-1) | closing speed (m/step) |
|---|---|---|---|---|---|---|---|
| ppo | 5 | 77.91 ± 1.97 | 86.25 ± 3.01 | 66.39 ± 0.65 | 0.00 ± 0.00 | 0.03 ± 0.04 | 0.0116 ± 0.0062 |
| ppo_deterministic | 5 | 79.03 ± 1.69 | 87.49 ± 3.17 | 67.34 ± 1.12 | 0.00 ± 0.00 | 0.04 ± 0.04 | 0.0125 ± 0.0062 |
| untrained | 5 | 62.80 ± 0.49 | 60.43 ± 0.86 | 66.08 ± 0.05 | 0.00 ± 0.00 | 0.00 ± 0.00 | -0.0005 ± 0.0006 |
| random | 1 | 63.56 | 61.69 | 66.14 | 0.00 | 0.00 | 0.0022 |
| still | 1 | 62.84 | 60.48 | 66.10 | 0.00 | 0.00 | 0.0000 |
| flee | 1 | 108.76 | 111.10 | 105.52 | 0.00 | 1.00 | 0.0981 |

### Keep-away 2v2 (PyBullet, kicking), warm-started from goto_target (5 runs; mean episode length, control steps)

| policy | seeds | overall length | possession starts | free-ball starts | fraction truncated | free-ball races won (0-1) | closing speed (m/step) |
|---|---|---|---|---|---|---|---|
| ppo | 5 | 89.37 ± 3.91 | 99.02 ± 3.46 | 76.05 ± 4.66 | 0.00 ± 0.00 | 0.41 ± 0.09 | 0.0561 ± 0.0039 |
| ppo_deterministic | 5 | 90.58 ± 4.73 | 98.16 ± 4.62 | 80.12 ± 5.32 | 0.00 ± 0.00 | 0.51 ± 0.08 | 0.0588 ± 0.0045 |
| untrained | 5 | 62.80 ± 0.49 | 60.43 ± 0.86 | 66.08 ± 0.05 | 0.00 ± 0.00 | 0.00 ± 0.00 | -0.0005 ± 0.0006 |
| random | 1 | 63.56 | 61.69 | 66.14 | 0.00 | 0.00 | 0.0022 |
| still | 1 | 62.84 | 60.48 | 66.10 | 0.00 | 0.00 | 0.0000 |
| flee | 1 | 108.76 | 111.10 | 105.52 | 0.00 | 1.00 | 0.0981 |

### Keep-away 3v2 (PyBullet, kicking), from scratch (5 runs; mean episode length, control steps)

| policy | seeds | overall length | possession starts | free-ball starts | fraction truncated | free-ball races won (0-1) | closing speed (m/step) |
|---|---|---|---|---|---|---|---|
| ppo | 5 | 78.30 ± 1.11 | 82.99 ± 1.94 | 71.83 ± 1.12 | 0.00 ± 0.00 | 0.04 ± 0.04 | -0.0043 ± 0.0098 |
| ppo_deterministic | 5 | 78.08 ± 1.07 | 82.79 ± 1.38 | 71.56 ± 1.01 | 0.00 ± 0.00 | 0.03 ± 0.03 | -0.0055 ± 0.0100 |
| untrained | 5 | 64.33 ± 1.08 | 59.68 ± 1.82 | 70.76 ± 0.12 | 0.00 ± 0.00 | 0.01 ± 0.02 | -0.0003 ± 0.0010 |
| random | 1 | 62.40 | 56.24 | 70.90 | 0.00 | 0.05 | -0.0006 |
| still | 1 | 61.96 | 55.45 | 70.95 | 0.00 | 0.00 | 0.0006 |
| flee | 1 | 116.58 | 114.24 | 119.81 | 0.00 | 1.00 | 0.0982 |

### Keep-away 3v2 (PyBullet, kicking), warm-started from goto_target (5 runs; mean episode length, control steps)

| policy | seeds | overall length | possession starts | free-ball starts | fraction truncated | free-ball races won (0-1) | closing speed (m/step) |
|---|---|---|---|---|---|---|---|
| ppo | 5 | 79.37 ± 3.21 | 84.26 ± 4.38 | 72.62 ± 2.91 | 0.00 ± 0.01 | 0.06 ± 0.08 | 0.0190 ± 0.0119 |
| ppo_deterministic | 5 | 79.72 ± 2.74 | 85.10 ± 4.61 | 72.28 ± 2.31 | 0.01 ± 0.01 | 0.05 ± 0.07 | 0.0192 ± 0.0111 |
| untrained | 5 | 64.33 ± 1.08 | 59.68 ± 1.82 | 70.76 ± 0.12 | 0.00 ± 0.00 | 0.01 ± 0.02 | -0.0003 ± 0.0010 |
| random | 1 | 62.40 | 56.24 | 70.90 | 0.00 | 0.05 | -0.0006 |
| still | 1 | 61.96 | 55.45 | 70.95 | 0.00 | 0.00 | 0.0006 |
| flee | 1 | 116.58 | 114.24 | 119.81 | 0.00 | 1.00 | 0.0982 |

### Keep-away 3v3 (PyBullet, kicking), from scratch (5 runs; mean episode length, control steps)

| policy | seeds | overall length | possession starts | free-ball starts | fraction truncated | free-ball races won (0-1) | closing speed (m/step) |
|---|---|---|---|---|---|---|---|
| ppo | 5 | 67.29 ± 0.96 | 77.93 ± 1.72 | 52.60 ± 0.82 | 0.00 ± 0.00 | 0.02 ± 0.03 | 0.0039 ± 0.0087 |
| ppo_deterministic | 5 | 67.88 ± 1.58 | 78.43 ± 1.96 | 53.32 ± 1.24 | 0.00 ± 0.00 | 0.03 ± 0.03 | 0.0020 ± 0.0101 |
| untrained | 5 | 54.94 ± 2.02 | 56.91 ± 3.18 | 52.22 ± 0.56 | 0.00 ± 0.00 | 0.05 ± 0.03 | 0.0007 ± 0.0017 |
| random | 1 | 53.98 | 55.79 | 51.48 | 0.00 | 0.14 | 0.0016 |
| still | 1 | 53.54 | 54.59 | 52.10 | 0.00 | 0.00 | 0.0005 |
| flee | 1 | 97.82 | 98.72 | 96.57 | 0.00 | 1.00 | 0.0981 |

### Keep-away 3v3 (PyBullet, kicking), warm-started from goto_target (5 runs; mean episode length, control steps)

| policy | seeds | overall length | possession starts | free-ball starts | fraction truncated | free-ball races won (0-1) | closing speed (m/step) |
|---|---|---|---|---|---|---|---|
| ppo | 5 | 69.35 ± 2.32 | 79.96 ± 3.61 | 54.70 ± 0.54 | 0.01 ± 0.01 | 0.10 ± 0.06 | 0.0232 ± 0.0089 |
| ppo_deterministic | 5 | 68.08 ± 2.51 | 77.89 ± 3.96 | 54.54 ± 3.25 | 0.00 ± 0.01 | 0.08 ± 0.07 | 0.0228 ± 0.0083 |
| untrained | 5 | 54.94 ± 2.02 | 56.91 ± 3.18 | 52.22 ± 0.56 | 0.00 ± 0.00 | 0.05 ± 0.03 | 0.0007 ± 0.0017 |
| random | 1 | 53.98 | 55.79 | 51.48 | 0.00 | 0.14 | 0.0016 |
| still | 1 | 53.54 | 54.59 | 52.10 | 0.00 | 0.00 | 0.0005 |
| flee | 1 | 97.82 | 98.72 | 96.57 | 0.00 | 1.00 | 0.0981 |

### goto_target warm-up, PyBullet, 1v0, kicking (5 runs)

| policy | seeds | mean reward (-m) |
|---|---|---|
| ppo | 5 | -1.09 ± 0.03 |
| ppo_deterministic | 5 | -1.01 ± 0.02 |
| untrained | 5 | -6.59 ± 0.01 |
| random | 1 | -6.54 |
| still | 1 | -6.54 |
| oracle | 1 | -0.92 |
