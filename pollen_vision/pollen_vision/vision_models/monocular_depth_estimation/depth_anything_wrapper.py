import numpy as np
import numpy.typing as npt
from PIL import Image
from transformers import pipeline


class DepthAnythingWrapper:
    def __init__(self) -> None:
        self.checkpoint = "LiheYoung/depth-anything-small-hf"  # much faster, don't see much difference in quality
        # self.checkpoint = "LiheYoung/depth-anything-large-hf"
        self.pipe = pipeline(task="depth-estimation", model=self.checkpoint)

    def get_depth(self, rgb: npt.NDArray[np.uint8]) -> npt.NDArray[np.float32]:
        depth = np.asarray(self.pipe(Image.fromarray(rgb))["depth"])  # type: ignore [no-untyped-call]
        depth = np.max(depth.flatten()) - depth
        return depth
