from collections import deque
import numpy as np

class Smoother:
    """Running average"""

    def __init__(self, historySize=10):
        self.history = deque(maxlen=historySize)

    def __call__(self, x):
        raise NotImplementedError


class WindowSmoother(Smoother):
    """Running average weighted by some window function"""

    def __call__(self, x):
        self.history.append(x)
        l = len(self.history)
        indices = range(0, -l, -1)
        normalizer = sum([self.window[i] for i in indices])
        return sum([
            self.window[i] * self.history[i] for i in indices
        ]) / normalizer

class BoxSmoother(WindowSmoother):
    """Running average weighted by a uniform window function"""

    def __init__(self, historySize=10):
        WindowSmoother.__init__(self, historySize=historySize)
        self.window = np.ones((historySize,))

class WeightedSmoother(WindowSmoother):
    """Running average weighted by (renormalized) given weights"""

    def __init__(self, historySize=10):
        WindowSmoother.__init__(self, historySize=historySize)
        self.window = deque(maxlen=historySize)

    def __call__(self, x, weight=1):
        self.window.append(weight)
        return WindowSmoother.__call__(self, x)
