from abc import ABC, abstractmethod
import numpy as np

class AbstractModel(ABC):
    def __init__(self, config):
        self.config = config
    
    @abstractmethod
    def train(self, *args, **kwargs):
        pass

    @abstractmethod
    def evaluate(self, *args, **kwargs):
        pass

    @abstractmethod
    def predict(self, *args, **kwargs):
        pass

    @abstractmethod
    def step(self,*args, **kwargs):
        pass

    @abstractmethod
    def save(self):
        pass
    
