import numpy as np
from sklearn.model_selection import train_test_split
import logging


#................................................................

from artwork_regression_model.processing import data_management,preprocessors as pp 
from artwork_regression_model.processing.data_management import save_pipeline,load_data
from artwork_regression_model import pipeline
from artwork_regression_model.pipeline import model_pipeline
from artwork_regression_model.config import config




logger = logging.getLogger(__name__)



### define the training functions

def run_training()->None:


    trainset = load_data(file_name = config.DATA_FILE)
    xtrainset, ytrainset = trainset[config.FEATURES] , trainset[config.TARGET]
    ytrainset = pp.targetCapping(ytrainset)
    
    model_pipeline.fit(X = xtrainset,y = ytrainset)
    save_pipeline(pipeline_to_persist=model_pipeline)


if __name__ == "__main__":

    run_training()