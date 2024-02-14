from importlib.resources import files
from typing import Any, List, Tuple

import numpy as np


def get_checkpoints_names() -> List[str]:
    path = files("checkpoints")
    names = []
    for file in path.glob("**/*.pt"):  # type: ignore[attr-defined]
        names.append(file.stem)

    for file in path.glob("**/*.pth"):  # type: ignore[attr-defined]
        names.append(file.stem)

    return names


def get_checkpoint_path(name: str) -> Any:
    path = files("checkpoints")
    for file in path.glob("**/*"):  # type: ignore[attr-defined]
        if file.stem == name:
            return file.resolve()
    return None


def random_color() -> Tuple[int, int, int]:
    return tuple(np.random.randint(0, 255, 3))
