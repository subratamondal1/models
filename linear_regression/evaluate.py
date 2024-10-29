import torch

from utils.utils import get_newest_filename

from .data import x_test, y_test
from .trainer import criterion, model

# Set the model to evaluation mode
model_name = get_newest_filename(directory="checkpoints")
print(f"Loading model: {model_name}")
model.load_state_dict(state_dict=torch.load(f=model_name))
model.eval()
# Disable gradient calculation
with torch.no_grad():
    # Make prediction on the test data
    y_pred = model(x_test)
# Calculate the loss
loss = criterion(y_pred, y_test)
print(f"\nLoss: {loss}")
