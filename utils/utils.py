import datetime
import os

import torch


def save_model(
    model, model_version: str = "v1", dir: str = "linear_regression/checkpoints"
) -> str:
    checkpoint_dir = os.path.join(dir, model_version)
    os.makedirs(
        name=checkpoint_dir, exist_ok=True
    )  # Create the directory if it doesn't exist
    checkpoint_filename: str = f"{checkpoint_dir}/model{model_version}-{datetime.datetime.now().strftime('%Y-%m-%d-%H-%M-%S')}.pth"
    torch.save(obj=model.state_dict(), f=checkpoint_filename)
    print(f"Model saved to {checkpoint_filename}")
    return checkpoint_filename
