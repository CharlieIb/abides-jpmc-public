from abc import abstractmethod, ABC
from collections import deque

from abides_core import Agent


class CoreGymAgent(Agent, ABC):
    """
    Abstract class to inherit from to create usable specific ABIDES Gym Experiemental Agents
    Nothing has changed in this class and it could be deleted in the future.
    """

    @abstractmethod
    def update_raw_state(self) -> None:
        raise NotImplementedError

    @abstractmethod
    def get_raw_state(self) -> deque:
        raise NotImplementedError
