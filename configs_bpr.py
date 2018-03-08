from model.constants import *
from os import path


configs = {
    'embedding_size': 256,
    'training_epochs': 200,
    'self.batch_size': 256,
    'emb_init_value': 1,
    'lambda_value': 0.001,
    'display_step': 100,
    "neg_sample_nums": 10,
    "pos_item_threshold": 300,
    "max_size": 30,
    'model_name': "bpr_model",
    "save_path": path.join(tmp_dir, "bpr"),
    "rec_dict": path.join(data_dir, "bpr_rec_list"),
}
