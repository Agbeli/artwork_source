import numpy as np 
import pandas as pd 
import json
import argparse, sys
import warnings

warnings.filterwarnings('ignore')

parser = argparse.ArgumentParser()

parser.add_argument('--index', help='Unique row identifier', type=int)
parser.add_argument('--artist', help='Artist name')
parser.add_argument('--country', help='Country of artist')
parser.add_argument('--country_group', help='Region of country e.g. Middle East')
parser.add_argument('--artwork', help='Title of artwork')
parser.add_argument('--type', help='Medium')
parser.add_argument('--cm_x', help='Width in centimeters', type=float)
parser.add_argument('--cm_y', help='Height in centimeters', type=float)
parser.add_argument('--year_made', help='Year made', type=int)

args = parser.parse_args()

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
    data_ = {
        'index': args.index,
        'artist': args.artist,
        'country': args.country,
        'country_group': args.country_group,
        'artwork': args.artwork,
        'type': args.type,
        'cm_x': args.cm_x,
        'cm_y': args.cm_y,
        'year_made': args.year_made,
        ## empty features
        'auction_house': 'dsfds',
        'auction_location': 'sdfsdf',
        'year_sold': 2021,
        'month_sold': 4
    }

    data_ = pd.DataFrame(data_, index=[0])

    pred = make_predictions(input_data = data_[config.FEATURES])["price"]
    print(f"price: ${pred[0][0]:.2f}")
