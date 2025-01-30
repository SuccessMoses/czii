import pandas as pd
import numpy as np
import os
import shutil
import matplotlib.pyplot as plt
import copick
import tensorflow as tf
import numpy as np
from tensorflow.keras.utils import to_categorical

import constants
from typing import List, Tuple, Union

def dict_to_df(coord_dict, experiment_name):
    """
    Convert dictionary of coordinates to pandas DataFrame.
    
    Parameters:
    -----------
    coord_dict : dict
        Dictionary where keys are labels and values are Nx3 coordinate arrays
        
    Returns:
    --------
    pd.DataFrame
        DataFrame with columns ['x', 'y', 'z', 'label']
    """
    # Create lists to store data
    all_coords = []
    all_labels = []
    
    # Process each label and its coordinates
    for label, coords in coord_dict.items():
        all_coords.append(coords)
        all_labels.extend([label] * len(coords))
    
    # Concatenate all coordinates
    all_coords = np.vstack(all_coords)
    
    df = pd.DataFrame({
        'experiment': experiment_name,
        'particle_type': all_labels,
        'x': all_coords[:, 0],
        'y': all_coords[:, 1],
        'z': all_coords[:, 2]
    })

    
    return df


def calculate_patch_starts(dimension_size: int, patch_size: int) -> List[int]:
    """
    Calculate the starting positions of patches along a single dimension
    with minimal overlap to cover the entire dimension.
    
    Parameters:
    -----------
    dimension_size : int
        Size of the dimension
    patch_size : int
        Size of the patch in this dimension
        
    Returns:
    --------
    List[int]
        List of starting positions for patches
    """
    if dimension_size <= patch_size:
        return [0]
        
    # Calculate number of patches needed
    n_patches = np.ceil(dimension_size / patch_size)
    
    if n_patches == 1:
        return [0]
    
    # Calculate overlap
    total_overlap = (n_patches * patch_size - dimension_size) / (n_patches - 1)
    
    # Generate starting positions
    positions = []
    for i in range(int(n_patches)):
        pos = int(i * (patch_size - total_overlap))
        if pos + patch_size > dimension_size:
            pos = dimension_size - patch_size
        if pos not in positions:  # Avoid duplicates
            positions.append(pos)
    
    return positions

def extract_3d_patches_minimal_overlap(arrays: List[np.ndarray], patch_size: int) -> Tuple[List[np.ndarray], List[Tuple[int, int, int]]]:
    """
    Extract 3D patches from multiple arrays with minimal overlap to cover the entire array.
    
    Parameters:
    -----------
    arrays : List[np.ndarray]
        List of input arrays, each with shape (m, n, l)
    patch_size : int
        Size of cubic patches (a x a x a)
        
    Returns:
    --------
    patches : List[np.ndarray]
        List of all patches from all input arrays
    coordinates : List[Tuple[int, int, int]]
        List of starting coordinates (x, y, z) for each patch
    """
    if not arrays or not isinstance(arrays, list):
        raise ValueError("Input must be a non-empty list of arrays")
    
    # Verify all arrays have the same shape
    shape = arrays[0].shape
    if not all(arr.shape == shape for arr in arrays):
        raise ValueError("All input arrays must have the same shape")
    
    if patch_size > min(shape):
        raise ValueError(f"patch_size ({patch_size}) must be smaller than smallest dimension {min(shape)}")
    
    m, n, l = shape
    patches = []
    coordinates = []
    
    # Calculate starting positions for each dimension
    x_starts = calculate_patch_starts(m, patch_size)
    y_starts = calculate_patch_starts(n, patch_size)
    z_starts = calculate_patch_starts(l, patch_size)
    
    # Extract patches from each array
    for arr in arrays:
        for x in x_starts:
            for y in y_starts:
                for z in z_starts:
                    patch = arr[
                        x:x + patch_size,
                        y:y + patch_size,
                        z:z + patch_size
                    ]
                    patches.append(patch)
                    coordinates.append((x, y, z))
    
    return patches, coordinates

# Note: I should probably averge the overlapping areas, 
# but here they are just overwritten by the most recent one. 

def reconstruct_array(patches: List[np.ndarray], 
                     coordinates: List[Tuple[int, int, int]], 
                     original_shape: Tuple[int, int, int]) -> np.ndarray:
    """
    Reconstruct array from patches.
    
    Parameters:
    -----------
    patches : List[np.ndarray]
        List of patches to reconstruct from
    coordinates : List[Tuple[int, int, int]]
        Starting coordinates for each patch
    original_shape : Tuple[int, int, int]
        Shape of the original array
        
    Returns:
    --------
    np.ndarray
        Reconstructed array
    """
    reconstructed = np.zeros(original_shape, dtype=np.int64)  # To track overlapping regions
    
    patch_size = patches[0].shape[0]
    
    for patch, (x, y, z) in zip(patches, coordinates):
        reconstructed[
            x:x + patch_size,
            y:y + patch_size,
            z:z + patch_size
        ] = patch
        
    
    return reconstructed
                         

INFERENCE_TRNASFORMS = Compose([
    EnsureChannelFirstd(keys=["image"], channel_dim="no_channel"),
    NormalizeIntensityd(keys="image"),
    Orientationd(keys=["image"], axcodes="RAS")
])

id_to_name = {1: "apo-ferritin", 
              2: "beta-amylase",
              3: "beta-galactosidase", 
              4: "ribosome", 
              5: "thyroglobulin", 
              6: "virus-like-particle"}

BLOB_THRESHOLD = 500
CERTAINTY_THRESHOLD = 0.5

CLASSES = [1, 2, 3, 4, 5, 6]

def validation(model, valid_id):
    with torch.no_grad():
        location_df = []
        for run in ROOT.runs:
            print(run)

            tomo = run.get_voxel_spacing(10)
            tomo = tomo.get_tomogram(TOMOGRAM_ALGORITHM).numpy()

            tomo_patches, coordinates  = extract_3d_patches_minimal_overlap([tomo], 96)

            tomo_patched_data = [{"image": img} for img in tomo_patches]

            tomo_ds = CacheDataset(data=tomo_patched_data, transform=INFERENCE_TRNASFORMS, cache_rate=1.0)

            pred_masks = []

            for i in range(len(tomo_ds)):
                input_tensor = tomo_ds[i]['image'].unsqueeze(0).to("cuda")
                model_output = model(input_tensor)

                probs = torch.softmax(model_output[0], dim=0)
                thresh_probs = probs > CERTAINTY_THRESHOLD
                _, max_classes = thresh_probs.max(dim=0)

                pred_masks.append(max_classes.cpu().numpy())
                

            reconstructed_mask = reconstruct_array(pred_masks, coordinates, tomo.shape)
            
            location = {}

            for c in CLASSES:
                cc = cc3d.connected_components(reconstructed_mask == c)
                stats = cc3d.statistics(cc)
                zyx=stats['centroids'][1:]*10.012444 #https://www.kaggle.com/competitions/czii-cryo-et-object-identification/discussion/544895#3040071
                zyx_large = zyx[stats['voxel_counts'][1:] > BLOB_THRESHOLD]
                xyz =np.ascontiguousarray(zyx_large[:,::-1])

                location[id_to_name[c]] = xyz


            df = dict_to_df(location, run.name)
            location_df.append(df)
        
        location_df = pd.concat(location_df)
        location_df.insert(loc=0, column='id', value=np.arange(len(location_df)))

    _, score = compute_lb(location_df, valid_id)
    return score


