import torch
# Prepare the Data
x_train: torch.Tensor = torch.Tensor(
    [
        [3.3],
        [4.4],
        [5.5],
        [6.71],
        [6.93],
        [4.168],
        [9.779],
        [6.182],
        [7.59],
        [2.167],
        [7.042],
        [10.791],
        [5.313],
        [7.997],
        [3.1],
    ]
)

print(f"X Train: \n{x_train}")
print(f"X_train shape: {x_train.shape}")

y_train: torch.Tensor = torch.Tensor(
    [
        [1.7],
        [2.76],
        [2.09],
        [3.19],
        [1.694],
        [1.573],
        [3.366],
        [2.596],
        [2.53],
        [1.221],
        [2.827],
        [3.465],
        [1.65],
        [2.904],
        [1.3],
    ]
)

x_test: torch.Tensor = torch.Tensor(
    [
        [8.0],
        [9.0],
        [10.0],
        [11.0],
        [12.0],
    ]
)

y_test: torch.Tensor = torch.Tensor(
    [
        [3.5],
        [3.8],
        [4.1],
        [4.4],
        [4.7],
    ]
)