from abc import ABC, abstractmethod


class BaseExecutor(ABC):

    @abstractmethod
    def execute(self, encoding, architecture, x, weights, bias):
        pass