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
from helper import extract_3d_patches_minimal_overlap, reconstruct_array, dict_to_df
import pandas as pd
from callback import compute_lb

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