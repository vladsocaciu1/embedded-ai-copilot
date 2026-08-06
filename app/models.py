from pathlib import Path
from dataclasses import dataclass

@dataclass
class ProjectFile:
    path: Path
    content: str