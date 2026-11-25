```
Regression check: 9 cases x seeds [90, 91] vs results/silver
Run: 2026-11-24 at commit ce9bc62
Re-frozen by this run: #36 red_gain 2.0 -> 240 and #41's PyBullet contact-possession changes (PyBullet keepaway); #41's new config keys and CSV columns (all cases); add claim_ball and pass_practice_moving (#41's passing curriculum)
Comparison: exact

SCHEMA     keepaway_1v1_mock  (final len_mean 85.45 -> 85.45)
           keepaway_1v1_mock_seed90/config.json: +shared.free_speed_factor=1.0; +shared.ball_damping=0.4; +shared.start.max_ball_dist=0.0; +shared.start.drill_radius=2.0; +shared.start.pass_distance=2.0; ... (8 more)
           keepaway_1v1_mock_seed91/config.json: +shared.free_speed_factor=1.0; +shared.ball_damping=0.4; +shared.start.max_ball_dist=0.0; +shared.start.drill_radius=2.0; +shared.start.pass_distance=2.0; ... (8 more)
           keepaway_1v1_mock_seed90/train.csv: +columns ['ep_passes_mean', 'ep_frac_passed', 'ep_pass_rate', 'ep_return_share', 'ep_called_share', 'ep_miscalled_per_pickup', 'kicks_per_step']
           keepaway_1v1_mock_seed90/eval.csv: +columns ['passes_mean', 'frac_passed', 'pass_rate', 'return_share', 'called_share', 'miscalled_per_pickup']
           keepaway_1v1_mock_seed91/train.csv: +columns ['ep_passes_mean', 'ep_frac_passed', 'ep_pass_rate', 'ep_return_share', 'ep_called_share', 'ep_miscalled_per_pickup', 'kicks_per_step']
           keepaway_1v1_mock_seed91/eval.csv: +columns ['passes_mean', 'frac_passed', 'pass_rate', 'return_share', 'called_share', 'miscalled_per_pickup']
           eval_keepaway_1v1_mock_seed90_per_run.csv: +columns ['passes_mean', 'frac_passed', 'pass_rate', 'return_share', 'called_share', 'miscalled_per_pickup']
CHANGED    keepaway_1v1_pybullet  (final len_mean 89.5 -> 86.95)
           keepaway_1v1_pybullet_seed90/config.json: +shared.free_speed_factor=1.0; +shared.ball_damping=0.4; +shared.start.max_ball_dist=0.0; +shared.start.drill_radius=2.0; +shared.start.pass_distance=2.0; ... (8 more)
           keepaway_1v1_pybullet_seed91/config.json: +shared.free_speed_factor=1.0; +shared.ball_damping=0.4; +shared.start.max_ball_dist=0.0; +shared.start.drill_radius=2.0; +shared.start.pass_distance=2.0; ... (8 more)
           keepaway_1v1_pybullet_seed90/train.csv: 48 of 87 cells differ; +columns ['ep_passes_mean', 'ep_frac_passed', 'ep_pass_rate', 'ep_return_share', 'ep_called_share', 'ep_miscalled_per_pickup', 'kicks_per_step']; first: row 1 (iteration=1) ep_len_mean 88.26086956521739 -> 86.17391304347827; max |delta| 2.9 (rel 0.5)
           keepaway_1v1_pybullet_seed90/eval.csv: 20 of 70 cells differ; +columns ['passes_mean', 'frac_passed', 'pass_rate', 'return_share', 'called_share', 'miscalled_per_pickup']; first: row 1 (iteration=0) len_mean 88.9 -> 87.1; max |delta| 3.8 (rel 0.0445)
           keepaway_1v1_pybullet_seed90/fingerprint.json: final weights differ
           keepaway_1v1_pybullet_seed91/train.csv: 49 of 87 cells differ; +columns ['ep_passes_mean', 'ep_frac_passed', 'ep_pass_rate', 'ep_return_share', 'ep_called_share', 'ep_miscalled_per_pickup', 'kicks_per_step']; first: row 1 (iteration=1) ep_len_mean 86.04347826086956 -> 84.43478260869566; max |delta| 10.2 (rel 1)
           keepaway_1v1_pybullet_seed91/eval.csv: 20 of 70 cells differ; +columns ['passes_mean', 'frac_passed', 'pass_rate', 'return_share', 'called_share', 'miscalled_per_pickup']; first: row 1 (iteration=0) len_mean 89.0 -> 87.1; max |delta| 5 (rel 0.0728)
           keepaway_1v1_pybullet_seed91/fingerprint.json: final weights differ
           eval_keepaway_1v1_pybullet_seed90.csv: 46 of 114 cells differ; first: row 1 (policy=ppo) len_mean 89.5 -> 86.94999999999999; max |delta| 183 (rel 1.7)
           eval_keepaway_1v1_pybullet_seed90_per_run.csv: 51 of 189 cells differ; +columns ['passes_mean', 'frac_passed', 'pass_rate', 'return_share', 'called_share', 'miscalled_per_pickup']; first: row 1 (policy=ppo) len_mean 89.1 -> 87.6; max |delta| 183 (rel 1.66)
SCHEMA     keepaway_1v1_pbrs_mock  (final len_mean 85.75 -> 85.75)
           keepaway_1v1_pbrs_mock_seed90/config.json: +shared.free_speed_factor=1.0; +shared.ball_damping=0.4; +shared.start.max_ball_dist=0.0; +shared.start.drill_radius=2.0; +shared.start.pass_distance=2.0; ... (8 more)
           keepaway_1v1_pbrs_mock_seed91/config.json: +shared.free_speed_factor=1.0; +shared.ball_damping=0.4; +shared.start.max_ball_dist=0.0; +shared.start.drill_radius=2.0; +shared.start.pass_distance=2.0; ... (8 more)
           keepaway_1v1_pbrs_mock_seed90/train.csv: +columns ['ep_passes_mean', 'ep_frac_passed', 'ep_pass_rate', 'ep_return_share', 'ep_called_share', 'ep_miscalled_per_pickup', 'kicks_per_step']
           keepaway_1v1_pbrs_mock_seed90/eval.csv: +columns ['passes_mean', 'frac_passed', 'pass_rate', 'return_share', 'called_share', 'miscalled_per_pickup']
           keepaway_1v1_pbrs_mock_seed91/train.csv: +columns ['ep_passes_mean', 'ep_frac_passed', 'ep_pass_rate', 'ep_return_share', 'ep_called_share', 'ep_miscalled_per_pickup', 'kicks_per_step']
           keepaway_1v1_pbrs_mock_seed91/eval.csv: +columns ['passes_mean', 'frac_passed', 'pass_rate', 'return_share', 'called_share', 'miscalled_per_pickup']
           eval_keepaway_1v1_pbrs_mock_seed90_per_run.csv: +columns ['passes_mean', 'frac_passed', 'pass_rate', 'return_share', 'called_share', 'miscalled_per_pickup']
CHANGED    keepaway_1v1_pbrs_pybullet  (final len_mean 89.85 -> 88.35)
           keepaway_1v1_pbrs_pybullet_seed90/config.json: +shared.free_speed_factor=1.0; +shared.ball_damping=0.4; +shared.start.max_ball_dist=0.0; +shared.start.drill_radius=2.0; +shared.start.pass_distance=2.0; ... (8 more)
           keepaway_1v1_pbrs_pybullet_seed91/config.json: +shared.free_speed_factor=1.0; +shared.ball_damping=0.4; +shared.start.max_ball_dist=0.0; +shared.start.drill_radius=2.0; +shared.start.pass_distance=2.0; ... (8 more)
           keepaway_1v1_pbrs_pybullet_seed90/train.csv: 57 of 87 cells differ; +columns ['ep_passes_mean', 'ep_frac_passed', 'ep_pass_rate', 'ep_return_share', 'ep_called_share', 'ep_miscalled_per_pickup', 'kicks_per_step']; first: row 1 (iteration=1) ep_len_mean 88.26086956521739 -> 86.17391304347827; max |delta| 2.9 (rel 0.333)
           keepaway_1v1_pbrs_pybullet_seed90/eval.csv: 20 of 70 cells differ; +columns ['passes_mean', 'frac_passed', 'pass_rate', 'return_share', 'called_share', 'miscalled_per_pickup']; first: row 1 (iteration=0) len_mean 88.9 -> 87.1; max |delta| 3 (rel 0.044)
           keepaway_1v1_pbrs_pybullet_seed90/fingerprint.json: final weights differ
           keepaway_1v1_pbrs_pybullet_seed91/train.csv: 60 of 87 cells differ; +columns ['ep_passes_mean', 'ep_frac_passed', 'ep_pass_rate', 'ep_return_share', 'ep_called_share', 'ep_miscalled_per_pickup', 'kicks_per_step']; first: row 1 (iteration=1) ep_len_mean 86.04347826086956 -> 84.43478260869566; max |delta| 9.69 (rel 1)
           keepaway_1v1_pbrs_pybullet_seed91/eval.csv: 20 of 70 cells differ; +columns ['passes_mean', 'frac_passed', 'pass_rate', 'return_share', 'called_share', 'miscalled_per_pickup']; first: row 1 (iteration=0) len_mean 89.0 -> 87.1; max |delta| 3.2 (rel 0.0377)
           keepaway_1v1_pbrs_pybullet_seed91/fingerprint.json: final weights differ
           eval_keepaway_1v1_pbrs_pybullet_seed90.csv: 47 of 114 cells differ; first: row 1 (policy=ppo) len_mean 89.85 -> 88.35; max |delta| 183 (rel 1)
           eval_keepaway_1v1_pbrs_pybullet_seed90_per_run.csv: 51 of 189 cells differ; +columns ['passes_mean', 'frac_passed', 'pass_rate', 'return_share', 'called_share', 'miscalled_per_pickup']; first: row 1 (policy=ppo) len_mean 89.5 -> 89.1; max |delta| 183 (rel 1.62)
SCHEMA     goto_target_mock  (final len_mean 300 -> 300)
           goto_target_mock_seed90/config.json: +shared.free_speed_factor=1.0; +shared.ball_damping=0.4; +shared.start.max_ball_dist=0.0; +shared.start.drill_radius=2.0; +shared.start.pass_distance=2.0; ... (8 more)
           goto_target_mock_seed91/config.json: +shared.free_speed_factor=1.0; +shared.ball_damping=0.4; +shared.start.max_ball_dist=0.0; +shared.start.drill_radius=2.0; +shared.start.pass_distance=2.0; ... (8 more)
           goto_target_mock_seed90/train.csv: +columns ['ep_passes_mean', 'ep_frac_passed', 'ep_pass_rate', 'ep_return_share', 'ep_called_share', 'ep_miscalled_per_pickup', 'kicks_per_step']
           goto_target_mock_seed90/eval.csv: +columns ['passes_mean', 'frac_passed', 'pass_rate', 'return_share', 'called_share', 'miscalled_per_pickup']
           goto_target_mock_seed91/train.csv: +columns ['ep_passes_mean', 'ep_frac_passed', 'ep_pass_rate', 'ep_return_share', 'ep_called_share', 'ep_miscalled_per_pickup', 'kicks_per_step']
           goto_target_mock_seed91/eval.csv: +columns ['passes_mean', 'frac_passed', 'pass_rate', 'return_share', 'called_share', 'miscalled_per_pickup']
           eval_goto_target_mock_seed90_per_run.csv: +columns ['passes_mean', 'frac_passed', 'pass_rate', 'return_share', 'called_share', 'miscalled_per_pickup']
SCHEMA     goto_target_pybullet  (final len_mean 300 -> 300)
           goto_target_pybullet_seed90/config.json: +shared.free_speed_factor=1.0; +shared.ball_damping=0.4; +shared.start.max_ball_dist=0.0; +shared.start.drill_radius=2.0; +shared.start.pass_distance=2.0; ... (8 more)
           goto_target_pybullet_seed91/config.json: +shared.free_speed_factor=1.0; +shared.ball_damping=0.4; +shared.start.max_ball_dist=0.0; +shared.start.drill_radius=2.0; +shared.start.pass_distance=2.0; ... (8 more)
           goto_target_pybullet_seed90/train.csv: +columns ['ep_passes_mean', 'ep_frac_passed', 'ep_pass_rate', 'ep_return_share', 'ep_called_share', 'ep_miscalled_per_pickup', 'kicks_per_step']
           goto_target_pybullet_seed90/eval.csv: +columns ['passes_mean', 'frac_passed', 'pass_rate', 'return_share', 'called_share', 'miscalled_per_pickup']
           goto_target_pybullet_seed91/train.csv: +columns ['ep_passes_mean', 'ep_frac_passed', 'ep_pass_rate', 'ep_return_share', 'ep_called_share', 'ep_miscalled_per_pickup', 'kicks_per_step']
           goto_target_pybullet_seed91/eval.csv: +columns ['passes_mean', 'frac_passed', 'pass_rate', 'return_share', 'called_share', 'miscalled_per_pickup']
           eval_goto_target_pybullet_seed90_per_run.csv: +columns ['passes_mean', 'frac_passed', 'pass_rate', 'return_share', 'called_share', 'miscalled_per_pickup']
NEW        claim_ball_mock  (not in the record yet; this freeze adds it)
NEW        claim_ball_pybullet  (not in the record yet; this freeze adds it)
NEW        pass_practice_moving_pybullet  (not in the record yet; this freeze adds it)

Summary: 2 changed, 4 schema, 3 new
```
