#  app\ingestion\dataset_loader.py

from pathlib import Path
import json


class JSONLLoader:
    """
    Loads JSONL records from a local file.
    """

    def __init__(self, file_path: str | Path):
        self.file_path = Path(file_path)

    def load(self) -> list[dict]:
        if not self.file_path.exists():
            raise FileNotFoundError(
                f"Dataset file not found: {self.file_path}"
            )

        records = []

        with self.file_path.open(
            "r",
            encoding="utf-8"
        ) as file:

            for line in file:
                line = line.strip()

                if not line:
                    continue

                records.append(
                    json.loads(line)
                )

        return records