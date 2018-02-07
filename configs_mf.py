from model.constants import *
from os import path


configs = {
    'embedding_size': 128,
    'learning_rate': 0.01,
    'training_epochs': 10,
    'self.batch_size': 256,
    'emb_init_value': 1,
    'lambda_value': 0.001,
    'display_step': 100,
    'optimize_method': 'adam',
    "max_size": 30,
    'model_name': "mf_model",
    "save_path": path.join(tmp_dir, "mf"),
    "rec_dict": path.join(data_dir, "mf_rec_list"),
}
