import torch
import sys
import pickle
import numpy as np

sys.path.append("MTR/tools")

from train import parse_config

from mtr.models import model as model
from mtr.utils import common_utils
from mtr.datasets import build_dataloader

# --cfg_file MTR/tools/cfgs/waymo/mtr_single_file_vis.yaml

# checkpoint_path = "MTR\output\tools\cfgs\waymo\mtr_weak\my_first_exp"
checkpoint_path = "../checkpoint_epoch_1.pth"
data_path = "../data"
data_filename = "sample_867dd000677d389.pkl"

def load_model(checkpoint_path=checkpoint_path):
    # checkpoint = torch.load(checkpoint_path)
    args, cfg = parse_config()
    logger = common_utils.create_logger("log.txt", rank=cfg.LOCAL_RANK)  # create logger
    model_ = model.MotionTransformer(config=cfg.MODEL)
    model_.load_params_from_file(checkpoint_path, logger=logger, to_cpu=False)
    model_.eval()
    return model_

def dataloader():
    args, cfg = parse_config()
    logger = common_utils.create_logger("log_vis.txt", rank=cfg.LOCAL_RANK)  # create logger
    dataset, dataloader_, sampler = build_dataloader(
        dataset_cfg=cfg.DATA_CONFIG,
        batch_size=cfg.OPTIMIZATION.BATCH_SIZE_PER_GPU,
        dist=False, workers=1, logger=logger,
        training=False, merge_all_iters_to_one_epoch=False, total_epochs=cfg.OPTIMIZATION.NUM_EPOCHS,
        add_worker_init_fn=True
    )
    return dataset, dataloader_, sampler

def create_pickle(filename):
    data_file = open(data_path + "/processed_scenarios_training/" + filename, 'rb')
    data = pickle.load(data_file)
    data_file.close()
    # create a dict with these keys 'scenario_id', 'timestamps_seconds', 'current_time_index', 'sdc_track_index', 'objects_of_interest', 'tracks_to_predict'
    # and values from data
    dict = {}
    dict['scenario_id'] = data['scenario_id']
    dict['timestamps_seconds'] = data['timestamps_seconds']
    dict['current_time_index'] = data['current_time_index']
    dict['sdc_track_index'] = data['sdc_track_index']
    dict['objects_of_interest'] = data['objects_of_interest']
    dict['tracks_to_predict'] = data['tracks_to_predict']
    list = [dict]
    pickle.dump(list, open(data_path + "/processed_scenarios_single_file_vis.pkl", 'wb'))



def apply_model(model_):
    dataset, dataloader_, sampler = dataloader()
    i, batch_dict = next(enumerate(dataloader_))
    with torch.no_grad():
        batch_pred_dicts = model_(batch_dict)
        final_pred_dicts = dataset.generate_prediction_dicts(batch_pred_dicts, output_path=None)
    return final_pred_dicts

def apply(data_filename=data_filename, checkpoint_path=checkpoint_path):
    model_ = load_model(checkpoint_path=checkpoint_path).cuda()
    create_pickle(data_filename)
    result = apply_model(model_)
    return result