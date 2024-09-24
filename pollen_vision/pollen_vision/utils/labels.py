from typing import Dict, List, Tuple

import numpy as np


class Labels:
    """A class to store the labels and their colors."""

    def __init__(self) -> None:
        self.labels: Dict[str, Tuple[int, int, int]] = {}
        self.labels[""] = (255, 255, 255)

    def push(self, labels: List[str], colors: List[Tuple[int, int, int]] = []) -> None:
        """Pushes a list of labels and associated color to the main labels dictionary.

        If the color list is not set, a random color will be assigned to each label not already in the dictionnary.
        """
        if colors != []:
            if len(colors) != len(labels):
                raise ValueError("The length of the labels and colors lists must be the same.")

        for label in labels:
            if label not in self.labels:
                if colors != []:
                    self.labels[label] = colors[labels.index(label)]
                else:
                    self.labels[label] = self.random_color()

    def get_color(self, label: str) -> Tuple[int, int, int]:
        """Returns the color of the label."""
        return self.labels[label]

    def random_color(self) -> Tuple[int, int, int]:
        """Returns a random color."""
        return tuple(np.random.randint(0, 255, 3))
