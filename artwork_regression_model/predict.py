import numpy as np 
import pandas as pd 
import json

################################################################
from artwork_regression_model.processing.data_management import load_pipeline,load_data
from artwork_regression_model.processing import preprocessors as pp ,validation
from artwork_regression_model.config import config

import logging
import typing as t 


_logger = logging.getLogger(__name__)


pipeline_file_name = f"{config.PIPELINE_FILE_NAME}.pkl"
_model = load_pipeline(file_name=pipeline_file_name)


def make_predictions(*,input_data:t.Union[pd.DataFrame,dict])-> dict:

    data = pd.DataFrame(input_data)
    data = validation.validate_input(data)  ### validate the data. 

    prediction = _model.predict(data)

    result = {"price":prediction}
    return result



if __name__ == '__main__':
    ### run test prediction for the model saved 
    data_ = load_data(file_name = config.TESTSET)
    Xtest = data_[config.FEATURES]
    ytest = data_[config.TARGET]
    print(f"check the two: {ytest[0:1]}")
    pred = make_predictions(input_data = Xtest[0:1])["price"]
    print(f"price: ${pred[0][0]:.2f}")
    #preds = _model.predict(Xtest)
    # target = pp.targetCapping(data_[config.TARGET])
    # from sklearn.metrics import r2_score,mean_squared_error as mse, mean_absolute_error as mae 
    # print(f"Check r2 score: \n \t \t {r2_score(preds,target):.2f} \n")
    # print(f"check mean square error: \n \t \t {np.sqrt(mse(preds,target))} \n")
    # print(f"check mean absolute error: \n \t \t {mae(preds,target)} \n")