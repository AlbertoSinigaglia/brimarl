from collections import deque
import random


class ReplayMemory(object):
    """Implements an experience replay memory"""

    def __init__(self, capacity: int) -> None:
        """
        :param capacity: capacity of the experience replay memory
        """
        self.capacity = capacity
        self.memory = deque(maxlen=capacity)

    def __len__(self) -> int:
        """
        :returns: size of the experience replay memory
        """
        return len(self.memory)

    def push(self, transitions) -> None:
        """Stores a sequence of transitions (s, a, r, s', done)"""
        self.memory.append(transitions)

    def sample(self, batch_size: int):
        """
        :returns: a sample of size batch_size
        """
        batch_size = min(batch_size, len(self.memory))
        return random.sample(self.memory, batch_size)