### load libraries needed 
import pandas as pd
from joblib import load,dump

#### import the custom module called artwork_regression_model 
from artwork_regression_model.config import config as config

import logging as logging
import typing as t 


def load_data(*,file_name:str)->pd.DataFrame:
    """
    Args:
        filename: name of the dataset to load.

    Return: 
        dataframe of the file. 
    """
    data_ = pd.read_csv(f"{config.DATASET_DIR}/{file_name}")
    return data_ 

def save_pipeline(*,pipeline_to_persist)->None:

    save_file_name = f"{config.PIPELINE_FILE_NAME}.pkl"
    save_path = config.TRAINED_MODEL_DIR/save_file_name
    remove_old_pipeline(files_to_keep=[save_file_name])
    dump(pipeline_to_persist,save_path)
    print("*** pipeline saved successfully ***")


def load_pipeline(*,file_name:str):

    save_path = config.TRAINED_MODEL_DIR/file_name
    trained_model = load(save_path)
    print("*** pipeline load successfully ***")
    return trained_model

def remove_old_pipeline(*,files_to_keep:t.List[str])->None:

    """
    Remove all old pipeline... 
    """
    do_not_delete = files_to_keep + ["__init__.py"]
    for model_file in config.TRAINED_MODEL_DIR.iterdirs():
        if model_file.name not in do_not_delete:
            model_file.unlink()



# if __name__ == '__main__':

#     data = load_data(file_name=config.DATA_FILE)
#     print(data.head())





