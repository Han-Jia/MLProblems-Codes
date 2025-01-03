import numpy as np
import mini_torch as mtorch


class MLP:
    def __init__(self, input_size, output_size):
        self.W1 = mtorch.Tensor(np.random.uniform(-1, 1,
                                                  size=(input_size, 128)))
        self.b1 = mtorch.Tensor(np.zeros((1, 128)))
        self.W2 = mtorch.Tensor(
            np.random.uniform(-1, 1, size=(128, output_size)))
        self.b2 = mtorch.Tensor(np.zeros((1, output_size)))
        self.params = [self.W1, self.W2, self.b1, self.b2]

    def forward(self, x):
        x1 = x @ self.W1 + self.b1
        x1 = mtorch.relu(x1)
        x2 = x1 @ self.W2 + self.b2
        return x2
