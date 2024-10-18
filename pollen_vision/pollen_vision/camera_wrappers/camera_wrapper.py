import logging
from abc import ABC, abstractmethod
from datetime import timedelta
from typing import Dict, Tuple

import numpy as np
import numpy.typing as npt


class CameraWrapper(ABC):
    def __init__(self) -> None:
        self._logger = logging.getLogger(__name__)

    @abstractmethod
    def get_data(
        self,
    ) -> Tuple[Dict[str, npt.NDArray[np.uint8]], Dict[str, float], Dict[str, timedelta]]:
        pass

    @abstractmethod
    def get_K(self, cam_name: str) -> npt.NDArray[np.float32]:
        pass

    @abstractmethod
    def get_D(self, cam_name: str = "left") -> npt.NDArray[np.float32]:
        pass
