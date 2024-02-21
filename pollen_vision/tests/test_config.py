from pollen_vision.camera_wrappers.depthai.cam_config import CamConfig
from pollen_vision.camera_wrappers.depthai.utils import (
    get_config_file_path,
    get_config_files_names,
)
from pollen_vision.vision_models.utils import get_checkpoint_path, get_checkpoints_names


def test_config() -> None:
    list_config = get_config_files_names()
    for conf in list_config:
        c = CamConfig(get_config_file_path(conf), 50, resize=(1280, 720), exposure_params=None)
        assert c is not None


def test_get_checkpoints_names() -> None:
    valid_names = get_checkpoints_names()

    assert get_checkpoint_path("dummy") is None
    assert "mobile_sam" in valid_names
