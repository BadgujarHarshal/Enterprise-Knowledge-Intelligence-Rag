# app\ingestion\dataset_loader.py

from pathlib import Path
import json


class JSONLLoader:
    """Load records from a JSON Lines (.jsonl) file."""

    def __init__(self, file_path):
        self.file_path = Path(file_path)

    def load(self):
        """Load and return all JSONL records."""

        if not self.file_path.exists():
            raise FileNotFoundError(
                f"JSONL file not found: {self.file_path}"
            )

        records = []

        with self.file_path.open(
            "r",
            encoding="utf-8"
        ) as file:
            for line_number, line in enumerate(file, start=1):
                line = line.strip()

                if not line:
                    continue

                try:
                    records.append(json.loads(line))
                except json.JSONDecodeError as exc:
                    raise ValueError(
                        f"Invalid JSON on line {line_number} "
                        f"in {self.file_path}"
                    ) from exc

        return records