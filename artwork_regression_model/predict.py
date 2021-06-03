import numpy as np 
import pandas as pd 
import json

################################################################
from artwork_regression_model.processing.data_management import load_pipeline,load_data
from artwork_regression_model.processing import preprocessors as pp 
from artwork_regression_model.config import config

import logging
import typing as t 


_logger = logging.getLogger(__name__)


pipeline_file_name = f"{config.PIPELINE_FILE_NAME}.pkl"
_model = load_pipeline(file_name=pipeline_file_name)


def make_predictions(*,input_data:t.Union[pd.DataFrame,dict])-> dict:

    data = pd.DataFrame(input_data)


    prediction = _model.predict(data)

    result = {"price of artwork: ":prediction}
    return result



if __name__ == '__main__':

    data_ = load_data(file_name = config.TESTSET)
    Xtest = data_[config.FEATURES]
    preds = _model.predict(Xtest)
    target = pp.targetCapping(data_[config.TARGET])
    from sklearn.metrics import r2_score
    print("Check r2 score: ",r2_score(preds,target))