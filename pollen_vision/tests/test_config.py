from pathlib import Path

import depthai as dai
import numpy as np
import pytest
from pollen_vision.camera_wrappers.depthai.cam_config import CamConfig
from pollen_vision.camera_wrappers.depthai.utils import (
    get_config_file_path,
    get_config_files_names,
)
from pollen_vision.perception.utils import get_checkpoint_path, get_checkpoints_names


def test_config() -> None:
    list_config = get_config_files_names()
    for conf in list_config:
        c = CamConfig(get_config_file_path(conf), 50, resize=(1280, 720), exposure_params=None)
        assert c is not None

    with pytest.raises(AssertionError):
        c = CamConfig(get_config_file_path(conf), 50, resize=(1280, 720), exposure_params=(None, 10))

    with pytest.raises(AssertionError):
        c = CamConfig(get_config_file_path(conf), 50, resize=(1280, 720), exposure_params=(10, 10))

    with pytest.raises(AssertionError):
        c = CamConfig(get_config_file_path(conf), 50, resize=(1280, 720), exposure_params=(10, 2000))


def test_config_file() -> None:
    c = CamConfig(get_config_file_path("CONFIG_OAK_D_PRO"), 60, resize=(1280, 720), exposure_params=None)
    c.resize_resolution = None
    c.set_sensor_resolution((1280, 720))
    assert c.sensor_resolution[0] == 1280
    assert c.sensor_resolution[1] == 720
    assert c.resize_resolution[0] == 1280
    assert c.resize_resolution[1] == 720
    c.set_resize_resolution((940, 720))
    assert c.resize_resolution[0] == 940
    assert c.resize_resolution[1] == 720

    path = str((Path(__file__).parent / "data" / Path("calibration_unit_tests.json")).resolve().absolute())
    c.calib = dai.CalibrationHandler(path)
    K_ref = np.zeros((3, 3))
    K_ref[2, 2] = 1
    assert np.array_equal(c.get_K_left(), K_ref)
    c.set_calib(c.calib)
    assert np.array_equal(c.get_K_right(), K_ref)

    str_msg = (
        "Camera Config: \nFPS: 60\nSensor resolution: (1280, 720)\nResize resolution: (940, 720)\n"
        "Inverted: False\nFisheye: False\nMono: True\nMX ID: \nrectify: False\nforce_usb2: False\nExposure params: auto\nnot set"
    )

    assert c.to_string() == str_msg

    assert c.get_device_info().name == ""
    assert c.get_device_info().mxid == ""


def test_config_file_path() -> None:
    assert get_config_file_path(".") is None


def test_get_checkpoints_names() -> None:
    valid_names = get_checkpoints_names()

    assert get_checkpoint_path("dummy") is None
    assert "mobile_sam" in valid_names
