import json
import logging
from pathlib import Path
from typing import Dict


def ensure_dirs(paths) -> None:
	for p in paths:
		Path(p).mkdir(parents=True, exist_ok=True)


def setup_logger(log_file: Path) -> logging.Logger:
	log_file.parent.mkdir(parents=True, exist_ok=True)
	logger = logging.getLogger("xgb_training")
	logger.setLevel(logging.INFO)
	logger.handlers.clear()

	formatter = logging.Formatter("%(asctime)s | %(levelname)s | %(message)s")

	file_handler = logging.FileHandler(log_file, mode="w", encoding="utf-8")
	file_handler.setFormatter(formatter)

	stream_handler = logging.StreamHandler()
	stream_handler.setFormatter(formatter)

	logger.addHandler(file_handler)
	logger.addHandler(stream_handler)
	return logger


def save_json(data: Dict, save_path: Path) -> None:
	save_path.parent.mkdir(parents=True, exist_ok=True)
	with save_path.open("w", encoding="utf-8") as f:
		json.dump(data, f, ensure_ascii=False, indent=2)

