import pathlib as pathlib
from pathlib import Path
import artwork_regression_model 
import os as os
import pandas as pd 


### Define the path to the package....
PACKAGE_ROOT = Path(artwork_regression_model.__file__).resolve().parent   ### set path to the package 
TRAINED_MODEL_DIR = PACKAGE_ROOT / "trained_models"   ### path to store the trained model
DATASET_DIR = PACKAGE_ROOT / "datasets"               ### data storage 


### name of the dataset.
DATA_FILE = "trainset.csv"
TESTSET = "testset.csv"

FEATURES = ["index","artist", "country","country_group", "artwork", "type",
       "auction_house", "auction_location", "cm_x", "cm_y", "year_sold",
       "month_sold", "year_made"]

TARGET = ["price_usd"]

CATEGORICAL_FEATURES =  ["artist","country","country_group","artwork", "type","auction_house","auction_location"]

SELECTED_FEATURES = ["artist","country","country_group","artwork", "type","auction_house","auction_location","cm_x","cm_y","month_sold"]

FLOAT_FEATURES = ["cm_x","cm_y"]

NUMERIC_FEATURES = ["month_sold"]

SCALE_FEATURES = ["cm_x","cm_y","month_sold"]

DROP_FEATURES = ["index","year_sold","year_made"]

PIPELINE_FILE_NAME = "artwork_regression_model"
PIPELINE_SAVE_FILE = f"{PIPELINE_FILE_NAME}_output"


# if __name__ == "__main__":
#     print(f"This is parent path \n \t \t {PACKAGE_ROOT}")
#     dataset = pd.read_csv(DATASET_DIR/DATA_FILE)
#     print(dataset.shape)



    
    
    
    
    #export PYTHONPATH=/home/wallace07/workshop/sentiment:${PYTHONPATH}
    #export $PYTHONPATH=/home/wallace07/Credit_rating/credit_rating/packages:${PYTHONPATH}