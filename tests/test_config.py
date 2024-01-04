import pytest

from src.depthai_wrappers.cam_config import CamConfig


def test_config():
    c = CamConfig("./config_files/CONFIG_CUSTOM_SR.json", 50, resize=(1280, 720), exposure_params=None)
    assert c is not None
