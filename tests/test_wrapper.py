import pytest

from src.depthai_wrappers.wrapper import Wrapper


def test_wrapper():
    with pytest.raises(TypeError):
        w = Wrapper("dummy", 50, resize=(1280, 720), rectify=True)
