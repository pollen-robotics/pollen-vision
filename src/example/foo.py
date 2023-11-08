"""Example class."""

import logging


class Foo:
    """This is a template class"""

    def __init__(self) -> None:
        """Set up empty slots."""
        self._logger = logging.getLogger(__name__)
        self._logger.info("Constructor")
        self._private_variable = "private"
        self.public_variable = "public"

    @property
    def private_variable(self) -> str:
        return self._private_variable

    @private_variable.setter
    def private_variable(self, value: str) -> None:
        self._private_variable = value

    def __del__(self) -> None:
        self._logger.info("Destructor")
