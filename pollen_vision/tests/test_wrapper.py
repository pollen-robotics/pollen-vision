import pytest


def test_wrapper() -> None:
    from pollen_vision.camera_wrappers.depthai.wrapper import Wrapper

    with pytest.raises(TypeError):
        Wrapper("dummy", 50, resize=(1280, 720), rectify=True)


def test_get_connected_devices() -> None:
    from pollen_vision.camera_wrappers.depthai.utils import get_connected_devices

    devices = get_connected_devices()
    assert devices == {}
