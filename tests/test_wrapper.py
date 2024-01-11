import pytest

from src.depthai_wrappers.wrapper import Wrapper


def test_wrapper():
    with pytest.raises(TypeError):
        w = Wrapper("dummy", 50, resize=(1280, 720), rectify=True)


def test_get_connected_devices():
    from src.depthai_wrappers.utils import get_connected_devices

    devices = get_connected_devices()
    assert devices == {}
