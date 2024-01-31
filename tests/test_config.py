import pytest

from src.depthai_wrappers.cam_config import CamConfig
from src.depthai_wrappers.utils import get_config_file_path, get_config_files_names


def test_config():
    list_config = get_config_files_names()
    for conf in list_config:
        c = CamConfig(get_config_file_path(conf), 50, resize=(1280, 720), exposure_params=None)
        assert c is not None
