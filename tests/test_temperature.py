import pytest

from src.example.celcius import Celsius


def test_temperature():
    temp = Celsius(37)
    assert temp.temperature == 37
    with pytest.raises(ValueError, match="Temperature below -273 is not possible"):
        temp.temperature = -300
