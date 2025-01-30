from czii_helper import dotdict
from dataset import PARTICLE, read_one_truth
from scipy.optimize import linear_sum_assignment

from deepfindET.inference import Segment
from deepfindET.utils import core

import cc3d
import tensorflow as tf
from tensorflow.keras.utils import Sequence
import os
import constants
import copick
from helper import dict_to_df
from tensorflow.keras.callbacks import Callback


import pandas as pd
import numpy as np
from constants import VALID_DIR


id_to_name = {1: "apo-ferritin", 
              2: "beta-amylase",
              3: "beta-galactosidase", 
              4: "ribosome", 
              5: "thyroglobulin", 
              6: "virus-like-particle"}

def do_one_eval(truth, predict, threshold):
    P=len(predict)
    T=len(truth)

    if P==0:
        hit=[[],[]]
        miss=np.arange(T).tolist()
        fp=[]
        metric = [P,T,len(hit[0]),len(miss),len(fp)]
        return hit, fp, miss, metric

    if T==0:
        hit=[[],[]]
        fp=np.arange(P).tolist()
        miss=[]
        metric = [P,T,len(hit[0]),len(miss),len(fp)]
        return hit, fp, miss, metric

    #---
    distance = predict.reshape(P,1,3)-truth.reshape(1,T,3)
    distance = distance**2
    distance = distance.sum(axis=2)
    distance = np.sqrt(distance)
    p_index, t_index = linear_sum_assignment(distance)

    valid = distance[p_index, t_index] <= threshold
    p_index = p_index[valid]
    t_index = t_index[valid]
    hit = [p_index.tolist(), t_index.tolist()]
    miss = np.arange(T)
    miss = miss[~np.isin(miss,t_index)].tolist()
    fp = np.arange(P)
    fp = fp[~np.isin(fp,p_index)].tolist()

    metric = [P,T,len(hit[0]),len(miss),len(fp)] #for lb metric F-beta copmutation
    return hit, fp, miss, metric


def compute_lb(submit_df, overlay_dir=f'{VALID_DIR}/overlay/ExperimentRuns', valid_id = ['TS_99_9']):
    #valid_id = list(submit_df['experiment'].unique())

    eval_df = []
    for id in valid_id:
        truth = read_one_truth(id, overlay_dir) #=f'{valid_dir}/overlay/ExperimentRuns')
        id_df = submit_df[submit_df['experiment'] == id]
        for p in PARTICLE:
            p = dotdict(p)
            xyz_truth = truth[p.name]
            xyz_predict = id_df[id_df['particle_type'] == p.name][['x', 'y', 'z']].values
            hit, fp, miss, metric = do_one_eval(xyz_truth, xyz_predict, p.radius* 0.5)
            eval_df.append(dotdict(
                id=id, particle_type=p.name,
                P=metric[0], T=metric[1], hit=metric[2], miss=metric[3], fp=metric[4],
            ))
    print('')
    eval_df = pd.DataFrame(eval_df)
    gb = eval_df.groupby('particle_type').agg('sum').drop(columns=['id'])
    gb.loc[:, 'precision'] = gb['hit'] / gb['P']
    gb.loc[:, 'precision'] = gb['precision'].fillna(0)
    gb.loc[:, 'recall'] = gb['hit'] / gb['T']
    gb.loc[:, 'recall'] = gb['recall'].fillna(0)
    gb.loc[:, 'f-beta4'] = 17 * gb['precision'] * gb['recall'] / (16 * gb['precision'] + gb['recall'])
    gb.loc[:, 'f-beta4'] = gb['f-beta4'].fillna(0)

    gb = gb.sort_values('particle_type').reset_index(drop=False)
    # https://www.kaggle.com/competitions/czii-cryo-et-object-identification/discussion/544895
    gb.loc[:, 'weight'] = [1, 0, 2, 1, 2, 1]
    lb_score = (gb['f-beta4'] * gb['weight']).sum() / gb['weight'].sum()
    return gb, lb_score



class CustomSegment(Segment):
    def __init__(self, Ncl, model_name=None, path_weights=None, patch_size=96, 
                 model_filters = [48, 64, 128], model_dropout = 0, gpuID = None):
        core.DeepFindET.__init__(self)

        self.Ncl = Ncl

        # Segmentation, parameters for dividing data in patches:
        self.P = patch_size  # patch length (in pixels) /!\ has to a multiple of 4 (because of 2 pooling layers), so that dim_in=dim_out
        self.pcrop = 5 #25,how many pixels to crop from border (net model dependent)
        self.poverlap = 0 #55, patch overlap (in pixels) (2*pcrop + 5)

        self.path_weights = path_weights
        self.check_attributes()
        
        # Initialize Empty network:
        #model_loader.load_model(patch_size, Ncl, model_name, path_weights, 
                                          # model_filters, model_dropout)[0]
        # Set GPU configuration
        gpus = tf.config.experimental.list_physical_devices('GPU')
        if gpus and gpuID is not None:
            try:
                # Restrict TensorFlow to only use the first GPU
                tf.config.experimental.set_visible_devices(gpus[gpuID], 'GPU')
                tf.config.experimental.set_memory_growth(gpus[gpuID], True)
            except RuntimeError as e:
                # Visible devices must be set at program startup
                print(e)
    
    # Override the check_attributes method
    def check_attributes(self):
        self.is_positive_int(self.Ncl, 'Ncl')
        #self.is_h5_path(self.path_weights, 'path_weights')
        self.is_multiple_4_int(self.P, 'patch_size')


#BLOB_THRESHOLD = 500
#CERTAINTY_THRESHOLD = 0.5

BLOB_THRESHOLD = 200
CERTAINTY_THRESHOLD = 0.05

classes = [1, 2, 3, 4, 5, 6]

from contextlib import redirect_stdout

def process_tomo_to_location_df(model, valid_id):
    # Redirect stdout
    copickRoot = copick.from_file(constants.ROOT_FILE)
    run = copickRoot.get_run(valid_id[0])
    tomo = run.get_voxel_spacing(10)
    VALID_TOMO = tomo.get_tomogram().numpy()

    with open(os.devnull, 'w') as f, redirect_stdout(f):
        # Extract patches from the tomogram
        seg = CustomSegment(constants.N_CLASS, patch_size=72)
        seg.net = model
        scoremaps = seg.launch(tomo[:])
        labelmap = seg.to_labelmap(scoremaps)
        
        location = {}
        for c in classes:
            cc = cc3d.connected_components(labelmap == c)
            stats = cc3d.statistics(cc)
            zyx = stats['centroids'][1:] * 10.012444  # Adjusting centroid positions based on scale
            zyx_large = zyx[stats['voxel_counts'][1:] > BLOB_THRESHOLD]
            xyz = np.ascontiguousarray(zyx_large[:, ::-1])  # Flip zyx to xyz
    
            location[id_to_name[c]] = xyz
    
        # Convert the location dictionary to a DataFrame
        df = dict_to_df(location, valid_id[0])  # Replace 'tomo_run_name' with the actual run name
        df.insert(loc=0, column='id', value=np.arange(len(df)))
    
        return df



class MetricCallback(Callback):
    def __init__(self, compute_lb, tomo_data, valid_id, model_save_path="best_model.keras", patience=10):
        """
        Custom callback to compute a metric, save the best model, and implement early stopping.

        Args:
            compute_lb: Function to compute the metric.
            tomo_data: Data used for processing the tomogram.
            model_save_path (str): Path to save the best model.
            patience (int): Number of epochs to wait before stopping if no improvement.
        """
        super().__init__()
        self.compute_lb = compute_lb
        self.tomo_data = tomo_data
        self.model_save_path = model_save_path
        self.patience = patience
        self.best_metric = float('-inf')  # Best metric value (bigger is better)
        self.wait = 0  # Counter for early stopping patience
        self.valid_id = valid_id

    def on_epoch_end(self, epoch, logs=None):
        print(f"Epoch {epoch + 1} end: Computing metric...")

        # Process tomogram to dataframe using the model
        location_df = process_tomo_to_location_df(self.model, self.valid_id)

        # Compute the metric using the compute_lb function
        _, metric = self.compute_lb(location_df)

        # Check if the metric improved
        if metric > self.best_metric:
            print(f"New best metric: {metric:.4f}. Saving model...")
            self.best_metric = metric
            self.model.save(self.model_save_path)
            self.wait = 0  # Reset patience counter
        else:
            print(f"Metric {metric:.4f} did not improve from {self.best_metric:.4f}.")
            self.wait += 1

        # Implement early stopping
        if self.wait >= self.patience:
            print(f"Early stopping at epoch {epoch + 1}: No improvement in {self.patience} epochs.")
            self.model.stop_training = True

        # Print the metric for logging purposes
        print(f"Computed Metric for Epoch {epoch + 1}: {metric:.4f}")
