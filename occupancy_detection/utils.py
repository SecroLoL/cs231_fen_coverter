"""
General utility functions for all code in this dir
"""
import os 

def generate_checkpoint_path(save_path: str):

    """
    Given a model path, create a checkpoint path by appending '_ckpt' to the filename.

    Args:
    model_path (str): The original path to the model file.

    Returns:
    str: The new path with '_ckpt' appended to the filename.
    """
    # Extract the directory and filename
    dir_name = os.path.dirname(save_path)
    base_name = os.path.basename(save_path)
    
    # Split the filename into name and extension
    name, ext = os.path.splitext(base_name)
    
    # Create the new filename with '_ckpt' appended
    new_base_name = f"{name}_ckpt{ext}"
    
    # Construct the new path
    new_path = os.path.join(dir_name, new_base_name)
    
    return new_path