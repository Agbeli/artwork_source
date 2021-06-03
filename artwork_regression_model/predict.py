import numpy as np 
import pandas as pd 
import json

################################################################
from artwork_regression_model.processing.data_management import load_pipeline
from artwork_regression_model.config import config

import logging
import typing as t 


_logger = logging.getLogger(__name__)


pipeline_file_name = f"{config.PIPELINE_FILE_NAME}.pkl"
_model = load_pipeline(file_name=pipeline_file_name)

