import datetime
import os

import torch


def save_model(model, model_version: str = "v1", dir: str = ".checkpoints") -> None:
    """
    Saves the model to the specified directory with the given model version.

    The filename format is "model{model_version}-{timestamp}.pth" where timestamp is the current time.

    Args:
        model: The model to be saved.
        model_version (str): Model version. Defaults to "v1".
        dir (str): The directory to save the model. Defaults to "checkpoints".
    """
    now = datetime.datetime.now().strftime("%Y-%m-%d-%H-%M-%S")
    checkpoint_filename = f"{dir}/model{model_version}-{now}.pth"
    torch.save(model.state_dict(), checkpoint_filename)
    print(f"Model saved to {checkpoint_filename}")


def get_newest_filename(directory):
    """
    Returns the newest filename in the given directory based on the timestamp in the filename.

    Args:
        directory (str): The name of the directory to search for files.

    Returns:
        str: The newest filename in the directory.
    """
    # Get a list of all files in the directory
    files = os.listdir(directory)

    # Filter the list to only include files that have a timestamp in the filename
    timestamp_files = [file for file in files if len(file) > 19 and file[-4:] == ".pth"]

    # Parse the timestamp from each filename and store it in a dictionary
    timestamp_dict = {}
    for file in timestamp_files:
        timestamp_str = file[-19:-4]  # Extract the timestamp from the filename
        try:
            timestamp = datetime.datetime.strptime(timestamp_str, "%Y-%m-%d-%H-%M-%S")
            timestamp_dict[file] = timestamp
        except ValueError:
            pass

    # Get the latest checkpoint file
    if timestamp_dict:
        newest_filename = max(timestamp_dict, key=timestamp_dict.get)
    else:
        newest_filename = None

    return newest_filename
