import torch
import torch.nn as nn

from utils.utils import save_model

from .data import x_train, y_train
from .model import LinearRegression

# Device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using {device} device")
print(f"Threads: torch.get_num_threads(): {torch.get_num_threads()}")
print(f"Cuda Cores: torch.cuda.device_count(): {torch.cuda.device_count()}")

# Initialize the Model
model = LinearRegression(input_features=1, output_features=1)

# Loss Function and Optimizer
criterion = nn.MSELoss()
optimizer = torch.optim.SGD(model.parameters(), lr=0.01)

# Training Loop
EPOCH: int = 1000

for epoch in range(EPOCH):
    # set the model to training mode
    model.train()
    # do the Forward Pass and get the output
    out = model(x_train)
    # calculate the loss
    loss = criterion(out, y_train)
    # zero the gradients before backpropagation
    optimizer.zero_grad()
    # do the backward pass (backpropagation)
    loss.backward()
    # update the weights
    optimizer.step()
    # print the loss
    if (epoch + 1) % 50 == 0:
        print(f"Epoch [{epoch+1}/{EPOCH}], Loss: {loss.item():.4f}")


save_model(model=model, model_version="v1", dir="checkpoints")
