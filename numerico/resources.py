import json
from pathlib import Path
from typing import Any

PATH = Path(__file__).parent / "resources"


def read_json(name: str) -> Any:
    text = (PATH / name).read_text(encoding="utf-8")

    return json.loads(text)
