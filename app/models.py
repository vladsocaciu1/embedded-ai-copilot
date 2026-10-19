from pathlib import Path
from dataclasses import dataclass

@dataclass
class ProjectFile:
    path: Path  # relative to the project root
    content: str