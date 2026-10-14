from pathlib import Path
import csv
import sys

PYBULLET_SRC = Path(__file__).resolve().parents[1]
if str(PYBULLET_SRC) not in sys.path:
    sys.path.insert(0, str(PYBULLET_SRC))

from reporting import MetricsLogger


def test_metrics_logger_writes_csv(tmp_path):
    path = tmp_path / "metrics.csv"
    logger = MetricsLogger(path)

    logger.log({
        "episode": 1,
        "winner": "BLUE",
        "reward": 1.0,
    })
    logger.log({
        "episode": 2,
        "winner": "RED",
        "reward": -1.0,
    })

    with path.open(newline="", encoding="utf-8") as f:
        rows = list(csv.DictReader(f))

    assert rows[0]["winner"] == "BLUE"
    assert rows[1]["reward"] == "-1.0"
