from artwork_regression_model.predict import make_predictions
from artwork_regression_model.processing.data_management import  load_data
from artwork_regression_model.config import config

import math 


def test_make_single_prediction():

    #Given the data set
    data = load_data(file_name = config.TESTSET)
    data = data[config.FEATURES]
    single_test_input = data[0:1]


    #When 
    prediction = make_predictions(input_data=single_test_input)

    #then 
    assert prediction is not None 
    assert isinstance(prediction.get("price")[0][0],float)
    assert math.floor(prediction.get("price")[0][0]) == 2684


def test_make_multi_predictions():

    data = load_data(file_name= config.TESTSET)
    data = data[config.FEATURES]
    num_elements, _ = data.shape

    predictions = make_predictions(input_data=data)

    assert predictions is not None
    assert len(predictions.get("price")) == num_elements



if __name__ == '__main__':
    test_make_single_prediction()
    test_make_multi_predictions()
