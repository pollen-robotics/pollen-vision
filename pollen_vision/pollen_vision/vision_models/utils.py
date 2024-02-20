"""A collection of utility functions for the vieion models wrappers"""

from importlib.resources import files
from typing import Any, List, Tuple

import numpy as np


def get_checkpoints_names() -> List[str]:
    """Returns the names of the checkpoints available in the checkpoints directory."""
    path = files("checkpoints")
    names = []
    for file in path.glob("**/*.pt"):  # type: ignore[attr-defined]
        names.append(file.stem)

    for file in path.glob("**/*.pth"):  # type: ignore[attr-defined]
        names.append(file.stem)

    return names


def get_checkpoint_path(name: str) -> Any:
    """Returns the path of the checkpoint based on its name."""
    path = files("checkpoints")
    for file in path.glob("**/*"):  # type: ignore[attr-defined]
        if file.stem == name:
            return str(file.resolve())
    return None


def random_color() -> Tuple[int, int, int]:
    """Returns a random color."""
    return tuple(np.random.randint(0, 255, 3))
