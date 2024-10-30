import torch

from .data import x_test, y_test
from .trainer import criterion, model, saved_model_name

if saved_model_name is None:
    raise ValueError(f"No suitable model {saved_model_name} file found")
model.load_state_dict(state_dict=torch.load(f=saved_model_name))

# Set the model to evaluation mode
model.eval()
# Disable gradient calculation
with torch.no_grad():
    # Make prediction on the test data
    y_pred = model(x_test)
# Calculate the loss
loss = criterion(y_pred, y_test)
print(f"\nLoss: {loss}")
