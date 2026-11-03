```
Regression check: 6 cases x seeds [90, 91] vs results/silver
Run: 2026-11-03 at commit 1e945de + uncommitted changes
Re-frozen by this run: add goto_target on PyBullet (runs there since Austin's #26)
Comparison: exact

IDENTICAL  keepaway_1v1_mock
IDENTICAL  keepaway_1v1_pybullet
IDENTICAL  keepaway_1v1_pbrs_mock
IDENTICAL  keepaway_1v1_pbrs_pybullet
IDENTICAL  goto_target_mock
NEW        goto_target_pybullet  (not in the record yet; this freeze adds it)

Summary: 5 identical, 1 new
```
