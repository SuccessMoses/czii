import json
import copick

from typing import List, Tuple, Union
import numpy as np
import torch
from monai.data import DataLoader, Dataset, CacheDataset, decollate_batch
from monai.transforms import (
    Compose, 
    EnsureChannelFirstd, 
    Orientationd,  
    AsDiscrete,  
    RandFlipd, 
    RandRotate90d, 
    NormalizeIntensityd,
    RandCropByLabelClassesd,
)

import cc3d
import pandas as pd


CONFIG_BLOB = """{
    "name": "czii_cryoet_mlchallenge_2024",
    "description": "2024 CZII CryoET ML Challenge training data.",
    "version": "1.0.0",

    "pickable_objects": [
        {
            "name": "apo-ferritin",
            "is_particle": true,
            "pdb_id": "4V1W",
            "label": 1,
            "color": [  0, 117, 220, 128],
            "radius": 60,
            "map_threshold": 0.0418
        },
        {
            "name": "beta-amylase",
            "is_particle": true,
            "pdb_id": "1FA2",
            "label": 2,
            "color": [153,  63,   0, 128],
            "radius": 65,
            "map_threshold": 0.035
        },
        {
            "name": "beta-galactosidase",
            "is_particle": true,
            "pdb_id": "6X1Q",
            "label": 3,
            "color": [ 76,   0,  92, 128],
            "radius": 90,
            "map_threshold": 0.0578
        },
        {
            "name": "ribosome",
            "is_particle": true,
            "pdb_id": "6EK0",
            "label": 4,
            "color": [  0,  92,  49, 128],
            "radius": 150,
            "map_threshold": 0.0374
        },
        {
            "name": "thyroglobulin",
            "is_particle": true,
            "pdb_id": "6SCJ",
            "label": 5,
            "color": [ 43, 206,  72, 128],
            "radius": 130,
            "map_threshold": 0.0278
        },
        {
            "name": "virus-like-particle",
            "is_particle": true,
            "label": 6,
            "color": [255, 204, 153, 128],
            "radius": 135,
            "map_threshold": 0.201
        },
        {
            "name": "membrane",
            "is_particle": false,
            "label": 8,
            "color": [100, 100, 100, 128]
        },
        {
            "name": "background",
            "is_particle": false,
            "label": 9,
            "color": [10, 150, 200, 128]
        }
    ],

    "overlay_root": "/kaggle/working/overlay",

    "overlay_fs_args": {
        "auto_mkdir": true
    },

    "static_root": "/kaggle/input/czii-cryo-et-object-identification/train/static"
}"""

TRAIN_OVERLAY = '/kaggle/input/czii-cryo-et-object-identification/train/overlay'
WORKING_OVERLAY = '/kaggle/working/overlay'

VOXEL_SIZE = 10
TOMOGRAM_ALGORITHM = 'denoised'

# Output Name for the Segmentation Targets
NAME = "copickUtils"
USER_ID = "paintedPicks"
SESSION_ID = '0'

FLAG_BATCH_BOOTSTRAP = 1

N_CLASS = 7

TRAIN_DATA_DIR = "/kaggle/input/create-numpy-dataset-exp-name"
TEST_DATA_DIR = "/kaggle/input/czii-cryo-et-object-identification"
VALID_DIR ='/kaggle/input/czii-cryo-et-object-identification/train'



ROOT_FILE = TRAIN_DATA_DIR + "/copick.config"

with open(ROOT_FILE) as f:
    copick_config = json.load(f)

#copick_config['static_root'] = '/kaggle/input/czii-cryo-et-object-identification/test/static'

#copick_test_config_path = 'copick_test.config'

#with open(copick_test_config_path, 'w') as outf#ile:
#    json.dump(copick_config, outfile)


ROOT = copick.from_file(ROOT_FILE)
