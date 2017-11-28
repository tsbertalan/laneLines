from collections import deque
import numpy as np

class Smoother:

    def __init__(self, historySize=10):
        self.history = deque(maxlen=historySize)

    def __call__(self, x):
        raise NotImplementedError


class WindowSmoother(Smoother):

    def __call__(self, x):
        self.history.append(x)
        l = len(self.history)
        normalizer = sum(self.window[-l:])
        return sum([
            self.window[-i] * self.history[-i] for i in range(l)[::-1]
        ]) / normalizer

class BoxSmoother(WindowSmoother):

    def __init__(self, historySize=10):
        WindowSmoother.__init__(self, historySize=historySize)
        self.window = np.ones((historySize,))
