import numpy as np

class ExponentialMovingStdDev:
    def __init__(self, decay=0.75):
        self.decay = decay
        self.mean = None
        self.variance = None

    def update(self, data):
        if self.mean is None:
            self.mean = np.mean(data)
            self.variance = np.var(data)
        else:
            current_mean = np.mean(data)
            current_variance = np.var(data)
            self.variance = self.decay * self.variance + (1 - self.decay) * current_variance + \
                            self.decay * (self.mean - current_mean) ** 2
            self.mean = self.decay * self.mean + (1 - self.decay) * current_mean

    def get_stddev(self):
        return np.sqrt(self.variance)