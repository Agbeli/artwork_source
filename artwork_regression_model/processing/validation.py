from artwork_regression_model.config import config 
import pandas as pd 



def validate_input(data_input:pd.DataFrame)->pd.DataFrame:

    """
    Validate the input data if there is any form missing information in the 
    Input:  
        data_input: pd.DataFrame
    return:
        pd.DataFrame
    """
    validated_data = data_input.copy()

    if data_input[config.CATEGORICAL_FEATURES].isnull().any().any():
        validated_data = data_input[config.CATEGORICAL_FEATURES].dropna(axis=1,subset=config.FLOAT_FEATURES)
    
    if data_input[config.FLOAT_FEATURES].isnull().any().any():
        validated_data = data_input[config.FLOAT_FEATURES].dropna(axis=1,subset=config.CATEGORICAL_FEATURES)
    
    return validated_data