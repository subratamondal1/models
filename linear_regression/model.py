import torch.nn as nn
import torch

# Build the Architecture of the Model
class LinearRegression(nn.Module):
    def __init__(self, input_features:int, output_features:int) -> None:
        """
        Initialize the Linear Regression model.

        Args:
            input_features (int): The number of features in the input data.
            output_features (int): The number of features in the output data.
        """
        super(LinearRegression, self).__init__()
        self.linear:nn.Linear = nn.Linear(in_features=input_features, out_features=output_features)

    def forward(self, x:torch.Tensor) -> torch.Tensor:
        return self.linear(x)
