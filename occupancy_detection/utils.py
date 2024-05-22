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

def load_dataset(model_type: ModelType, data_path: str, batch_size: int = 32, subset_size: int = None):

    """
    TODO
    """

    if model_type is None:
        raise ValueError(f"ModelType {model_type} is invalid.")
    
    transform = transforms.Compose([
        transforms.ToTensor()
    ])

    if model_type == ModelType.INCEPTION:
        transform = transforms.Compose([
        transforms.Resize((299, 299)),  
        transforms.ToTensor(),
        ])  # InceptionV3 requires images to be size 299 x 299

    dataset = datasets.ImageFolder(root=data_path, transform=transform)

    if subset_size is not None:
        indices = np.random.choice(len(dataset), subset_size, replace=False)
        final_dataset = Subset(dataset, indices)
        dataset = final_dataset
    
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=True)
    return loader