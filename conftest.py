import logging
from datetime import datetime
from pathlib import Path


def pytest_configure(config):
    """Add a timestamped log file handler for every test run."""
    log_dir = Path("logs")
    log_dir.mkdir(exist_ok=True)

    timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    log_path = log_dir / f"{timestamp}.log"

    handler = logging.FileHandler(log_path, encoding="utf-8")
    handler.setLevel(logging.WARNING)
    handler.setFormatter(logging.Formatter(
        fmt="%(asctime)s  %(levelname)-8s  %(name)s:%(lineno)d  %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    ))

    logging.getLogger().addHandler(handler)
    logging.getLogger().setLevel(logging.WARNING)
