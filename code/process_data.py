

from utils import *

data_path = '/Users/patrick/Downloads/SIGMOD2023-dataset/'
dataset_name ='googleplus'

adj, features, y_train, y_val, y_test, train_mask, val_mask, test_mask, y_train_active, train_active_mask=load_unprocessed_data(data_path,dataset_name)