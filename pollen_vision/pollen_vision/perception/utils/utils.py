"""A collection of utility functions for the vision models wrappers"""

from importlib.resources import files
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import numpy.typing as npt
from sklearn.linear_model import LinearRegression
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import PolynomialFeatures


def fuse_depths(
    stereo_depth: npt.NDArray[np.float32],
    mono_depth: npt.NDArray[np.float32],
    fit_rect: Optional[tuple[int, int, int, int]] = None,
    poly_degree: int = 4,
) -> npt.NDArray[np.float32]:
    """
    Fuse a stereo depth map and a monocular etimated depth map using a polynomial regression model.

    Args:
        stereo_depth: The stereo depth map, computed from a stereo camera (considered ground truth here)
        mono_depth: The monocular estimated depth map, computed from a monocular depth estimation model
        fit_rect: The rectangle in which to fit the model (xmin, ymin, xmax, ymax). None to fit the model on the whole image
        poly_degree: The degree of the polynomial used in the regression
    """

    x = mono_depth
    y = stereo_depth

    if fit_rect is not None:
        x = x[fit_rect[0] : fit_rect[2], fit_rect[1] : fit_rect[3]]
        y = y[fit_rect[0] : fit_rect[2], fit_rect[1] : fit_rect[3]]

    # Filter out valid (non-NaN) corresponding points
    valid_indices = ~(np.isnan(y) | np.isnan(x))
    x = x[valid_indices]
    y = y[valid_indices]

    # normalize x (seems to help mitigate deformation)
    x = x / max(x)

    # Model with polynomial features
    model = make_pipeline(PolynomialFeatures(poly_degree), LinearRegression())
    # Fit the model
    model.fit(x.reshape(-1, 1), y.reshape(-1, 1))

    # Sample the model on the full image
    valid_indices = ~(np.isnan(stereo_depth) | np.isnan(mono_depth))
    x = mono_depth[valid_indices]
    x = x / max(x)
    fused_depth = model.predict(x.reshape(-1, 1)).reshape(stereo_depth.shape)

    return np.array(fused_depth, dtype=np.float32)


def get_centroid(mask: npt.NDArray[np.uint8]) -> Tuple[int, int]:
    x_center, y_center = np.argwhere(mask == 1).sum(0) / np.count_nonzero(mask)
    return int(y_center), int(x_center)


def uv_to_xyz(z: float, u: float, v: float, K: npt.NDArray[np.float32]) -> npt.NDArray[np.float32]:
    cx = K[0, 2]
    cy = K[1, 2]
    fx = K[0, 0]
    fy = K[1, 1]

    # Computing position in camera frame
    x = (u - cx) * z / fx
    y = (v - cy) * z / fy

    return np.array([x, y, z])


# in meters
def get_object_width_height(bbox: List[List], mask: npt.NDArray[np.uint8], K: npt.NDArray[np.float32]) -> Tuple[int, int]:

    u, v = get_centroid(mask)
    d = mask.copy()
    d[mask == 0] = 0
    average_depth = d[d != 0].mean()
    xyz = uv_to_xyz(average_depth * 0.1, u, v, K)
    xyz *= 0.01

    xmin, ymin, xmax, ymax = bbox
    u1, v1 = xmin, ymin
    u2, v2 = xmax, ymax

    ltop_xyz = uv_to_xyz(average_depth * 0.1, u1, v1, K)
    rbottom_xyz = uv_to_xyz(average_depth * 0.1, u2, v2, K)

    width = np.abs(rbottom_xyz[0] - ltop_xyz[0])
    height = np.abs(rbottom_xyz[1] - ltop_xyz[1])

    return width, height


# Actually only computes the position ignoring rotation for now. Still returns pose matrix, with rotation set to identity
def get_object_pose_in_world(
    depth: npt.NDArray[np.float32],
    mask: npt.NDArray[np.uint8],
    T_world_cam: npt.NDArray[np.float32],
    K: npt.NDArray[np.float32],
) -> npt.NDArray[np.float32]:
    u, v = get_centroid(mask)

    d = depth.copy()
    d[mask == 0] = 0
    average_depth = d[d != 0].mean()

    xyz = uv_to_xyz(average_depth * 0.1, u, v, K)
    xyz *= 0.01

    T_cam_object = np.eye(4)  # Rotation is identity for now
    T_cam_object[:3, 3] = xyz
    T_world_object = T_world_cam @ T_cam_object
    T_world_object[:3, :3] = np.eye(3)

    return T_world_object


def get_bboxes(predictions: List[Dict]) -> List[List]:  # type: ignore
    """Returns a list of bounding boxes from the predictions."""
    bboxes = []
    for prediction in predictions:
        box = prediction["box"]
        xmin, ymin, xmax, ymax = box.values()
        bboxes.append([xmin, ymin, xmax, ymax])

    return bboxes


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


def get_labels(predictions: List[Dict]) -> List[str]:  # type: ignore
    """Returns a list of labels from the predictions."""
    labels = []
    for prediction in predictions:
        labels.append(prediction["label"])

    return labels


def get_scores(predictions: List[Dict]) -> List[float]:  # type: ignore
    """Returns a list of scores from the predictions."""
    scores = []
    for prediction in predictions:
        scores.append(prediction["score"])

    return scores
