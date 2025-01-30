import pandas as pd
import numpy as np
import os
import shutil
from deepfindET.entry_points import step1
from deepfindET.utils import copick_tools
import matplotlib.pyplot as plt
import copick
import tensorflow as tf
import numpy as np
from deepfindET.utils import core, augmentdata
from tensorflow.keras.utils import to_categorical


import constants


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


def create_working_overlay(source_dir=constants.TRAIN_OVERLAY, destination_dir=constants.WORKING_OVERLAY, prefix="curation_0_"):
    """
    Walks through the source directory, creates corresponding subdirectories in the destination, 
    and copies files while ensuring they have the specified prefix.

    Args:
        source_dir (str): The path to the source directory.
        destination_dir (str): The path to the destination directory.
        prefix (str, optional): The prefix to add to files if not already present. Defaults to "curation_0_".
    """
    for root, dirs, files in os.walk(source_dir):
        # Create corresponding subdirectories in the destination
        relative_path = os.path.relpath(root, source_dir)
        target_dir = os.path.join(destination_dir, relative_path)
        os.makedirs(target_dir, exist_ok=True)

        # Copy and rename each file
        for file in files:
            new_filename = file if file.startswith(prefix) else f"{prefix}{file}"
            
            # Define full paths for the source and destination files
            source_file = os.path.join(root, file)
            destination_file = os.path.join(target_dir, new_filename)
            
            # Copy the file with the new name
            shutil.copy2(source_file, destination_file)
            print(f"Copied {source_file} to {destination_file}")


def write_train_targets_segmentation(copick_root, config, voxel_size=constants.VOXEL_SIZE, step1=step1, 
                           tomogram_algorithm=constants.TOMOGRAM_ALGORITHM, out_name=constants.NAME, 
                           out_user_id=constants.USER_ID, out_session_id=constants.USER_ID):
    """
    Generates training targets for protein coordinates and associated segmentations 
    for the 3D U-Net model using CoPick data.

    Args:
        copick_root (object): The root object containing pickable objects and metadata.
        voxel_size (float): The voxel size used in the tomogram data.
        config (str): Configuration file path specifying project settings.
        step1 (module): The module containing `create_train_targets` function.
        tomogram_algorithm (str): The reconstruction algorithm for tomograms, e.g., 'wbp'.
        out_name (str): Output name for the generated segmentation targets.
        out_user_id (str): User ID under which the output targets will be saved.
        out_session_id (str): Session ID associated with the output (for tracking).
        run_ids (list, optional): List of Run-IDs for which to generate targets. Defaults to None.
    
    Returns:
        dict: A dictionary containing train targets for each protein or object.
    """
    # Create working overlay
    create_working_overlay()

    train_targets = {}
    # Define protein targets with their respective radii
    targets = [(obj.name, None, None, (obj.radius / voxel_size)) 
               for obj in copick_root.pickable_objects if obj.is_particle]

    # Generate train target information
    for obj_name, user_id, session_id, radius in targets:
        train_targets[obj_name] = {
            "label": copick_root.get_object(obj_name).label,
            "user_id": user_id,
            "session_id": session_id,
            "radius": radius,
            "is_particle_target": True,
        }

    # Define segmentation target (e.g., membrane)
    seg_targets = [("membrane", None, None)]

    # Generate segmentation target information
    for obj_name, user_id, session_id in seg_targets:
        train_targets[obj_name] = {
            "label": copick_root.get_object(obj_name).label,
            "user_id": user_id,
            "session_id": session_id,
            "radius": None,
            "is_particle_target": False,
        }

    # Call create_train_targets to generate training targets
    step1.create_train_targets(
        config=config,
        train_targets=train_targets,
        voxel_size=voxel_size,
        tomogram_algorithm=tomogram_algorithm,
        out_name=out_name,
        out_user_id=out_user_id,
        out_session_id=out_session_id,
    )

    return train_targets  # Returning this in case it's useful for debugging or further processing


def copick_data_generator(
    TrainInstance,
    input_dataset,  # The dataset from which patches are extracted
    input_target,  # The corresponding ground truth (labels) for the dataset
    batch_size,  # Number of samples to generate per batch
    dim_in,  # Dimension of the input patches
    Ncl,  # Number of classes for categorical labeling
    flag_batch_bootstrap,  # Boolean flag to decide if batch bootstrapping is enabled
    organizedPicksDict,  # Dictionary containing organized picking information
    batch_data,
    batch_target,
):

    # Calculate the padding value for extracting patches (half the dimension size)
    p_in = int(np.floor(dim_in / 2))

      # Get the dimensions of the tomogram from the first tomoID in the organized picks
    tomodim = input_dataset[organizedPicksDict["tomoIDlist"][0]].shape

    # While loop for generating batches of data
    while True:

        # Generate bootstrap indices if bootstrapping is enabled, otherwise set to None
        pool = core.get_copick_boostrap_idx(organizedPicksDict, batch_size) if flag_batch_bootstrap else None
        # pool = range(0, len(objlist))

        # Initialize an empty list to store selected indices
        idx_list = []

        # Loop over the batch size to generate each sample in the batch
        for i in range(batch_size):

            # Randomly select an index from the bootstrap pool
            randomBSidx = np.random.choice(pool["bs_idx"])
            idx_list.append(randomBSidx)

            # Find the original index position of the selected bootstrap index
            index = np.where(pool["bs_idx"] == randomBSidx)[0][0]

            # Determine the patch position (x, y, z) within the tomogram
            x, y, z = core.get_copick_patch_position(
                tomodim,
                p_in,
                TrainInstance.Lrnd,
                TrainInstance.voxelSize,
                pool["protein_coords"][index],
            )

            # Extract the data patch from the input dataset based on the calculated position
            patch_data = input_dataset[pool["tomoID_idx"][index]][
                z - p_in : z + p_in,
                y - p_in : y + p_in,
                x - p_in : x + p_in,
            ]

            # Extract the corresponding target patch (ground truth labels)
            patch_target = input_target[pool["tomoID_idx"][index]][
                z - p_in : z + p_in,
                y - p_in : y + p_in,
                x - p_in : x + p_in,
            ]

            # Convert the target patch to categorical format based on the number of classes
            patch_target = to_categorical(patch_target, Ncl)

              # Normalize the data patch by subtracting the mean and dividing by the standard deviation
            patch_data = (patch_data - np.mean(patch_data)) / np.std(patch_data)

            # Apply data augmentations (e.g., rotation, flipping) to both data and target patches
            patch_data, patch_target = TrainInstance.data_augmentor.apply_augmentations(patch_data, patch_target)

            # Store the processed target patch in the batch target array
            batch_target[i] = patch_target

            # Store the processed data patch in the batch data array (assuming single channel data)
            batch_data[i, :, :, :, 0] = patch_data

        # Yield the batch data and targets as output to the calling function
        yield batch_data, batch_target



def get_dataset(
    batch_size,
    sample_size,
    dim_in,
    n_class=constants.N_CLASS,
    flag_batch_bootstrap=constants.FLAG_BATCH_BOOTSTRAP,
    labelName=constants.NAME,  # Define labelName, labelUserID, sessionID as arguments
    labelUserID=constants.USER_ID,
    sessionID=constants.SESSION_ID,
    Lrnd=15,
    voxelSize=constants.VOXEL_SIZE
):
    """
    Function to retrieve and prepare the copick dataset for training.

    Parameters:
    - root_file: Path to the root dataset file.
    - batch_size: Number of samples per batch.
    - dim_in: Input patch dimension.
    - n_class: Number of output classes.
    - sample_size: Size of the sample data to use.
    - flag_batch_bootstrap: Boolean flag for batch bootstrapping.
    - labelName: The label name for the training instance.
    - labelUserID: The user ID associated with the training instance.
    - sessionID: The session ID for tracking.
    - Lrnd: The learning radius, default value is 15.
    - voxelSize: The size of the voxel, default value is 10.

    Returns:
    - dataset: A TensorFlow Dataset object ready for training.
    """

    # Define TrainInstance class inside the function
    class TrainInstance:
        def __init__(self, labelName, labelUserID, sessionID, Lrnd, voxelSize):
            self.labelName = labelName
            self.labelUserID = labelUserID
            self.sessionID = sessionID
            self.Lrnd = Lrnd
            self.voxelSize = voxelSize
            # Assuming augmentdata.DataAugmentation() is a class you want to initialize
            self.data_augmentor = augmentdata.DataAugmentation()

    # Create an instance of TrainInstance
    train_instance = TrainInstance(labelName, labelUserID, sessionID, Lrnd, voxelSize)

    tomo_ids = [r.name for r in copick.from_file(constants.ROOT_FILE).runs]
    # Load training data
    train_tomo_ids = tomo_ids[:sample_size]
    print(f'Train Tomo IDs : {train_tomo_ids}')
    print(f'Valid Tomo IDs : {tomo_ids[sample_size:]}')
    (trainData, trainTarget) = core.load_copick_datasets(constants.ROOT_FILE, train_instance, train_tomo_ids)

    # Query organized picks
    copickRoot = copick.from_file(constants.ROOT_FILE)
    organizedPicksDict = core.query_available_picks(copickRoot, train_tomo_ids, None)

    batch_data = np.zeros((batch_size, dim_in, dim_in, dim_in, 1))
    batch_target = np.zeros((batch_size, dim_in, dim_in, dim_in, n_class))


    # Create dataset
    dataset = tf.data.Dataset.from_generator(
        lambda: copick_data_generator(
            train_instance,  # Pass the train_instance instead of TrainInstance
            trainData,
            trainTarget,
            batch_size,
            dim_in,
            n_class,
            flag_batch_bootstrap,
            organizedPicksDict,
            batch_data,
            batch_target,
        ),
        output_signature=(
            tf.TensorSpec(shape=(batch_size, dim_in, dim_in, dim_in, 1), dtype=tf.float32),
            tf.TensorSpec(shape=(batch_size, dim_in, dim_in, dim_in, n_class), dtype=tf.float32),
        ),
    )
    dataset = dataset.prefetch(buffer_size=tf.data.experimental.AUTOTUNE)

    return dataset

