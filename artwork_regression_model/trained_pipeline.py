import numpy as np
from sklearn.model_selection import train_test_split
import logging


#................................................................

from artwork_regression_model.processing import data_management,preprocessors as pp 
from artwork_regression_model.processing.data_management import save_pipeline,load_data
from artwork_regression_model import pipeline
from artwork_regression_model.config import config




logger = logging.getLogger(__name__)



### define the training functions

def run_training()->None:

    trainset = load_data(file_name = config.DATA_FILE)
    xtrainset, ytrainset = trainset[config.FEATURES] , trainset[config.TARGET]

    print("Nature of dataset:  \n")
    print(xtrainset.head(),"\n")
    print(ytrainset.head())


if __name__ == "__main__":

    run_training()