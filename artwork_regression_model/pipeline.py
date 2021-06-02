from sklearn.pipeline import Pipeline,FeatureUnion
from catboost import CatBoostRegressor
from sklearn.compose import TransformedTargetRegressor
import numpy as np
from sklearn.feature_selection import SelectFromModel
import pandas as pd 


### custom modules 
from artwork_regression_model.processing import preprocessors as pp 
from artwork_regression_model.config import  config

import logging as logging


categorical_features = config.CATEGORICAL_FEATURES

class CustomSelectionFeature(SelectFromModel):

    pass  

class CustomCatboostRegressor(CatBoostRegressor):

    def fit(self,X,y=None,**fit_params):

        dataframe = pd.DataFrame(X)
        dataframe = dataframe.infer_objects()
        cat_features_new = [dataframe.columns.get_loc(col) for col in dataframe.select_dtypes(include=['object', 'bool']).columns]
        print(cat_features_new)

        return super().fit(X ,y = y,cat_features=cat_features_new,**fit_params)


UnionFeature = FeatureUnion(
    [
        (
            "outlier_capping",pp.OutlierCapping(config.FLOAT_FEATURES)
            ),
            (
                "frequent_labels",pp.RareImputation(config.CATEGORICAL_FEATURES)

            ),
            (
                "encoding_feature",pp.FrequencyEncoding(config.NUMERIC_FEATURES)

            ),
            (
                "log_transform",pp.LogTransformer(config.FLOAT_FEATURES)

            ),
            (
                "scalar",pp.ScaleFeatues(config.SCALE_FEATURES)
            ),
            (
                "dropout_features",pp.DropFeature(config.DROP_FEATURES)
            )

        ])


List = list(range(len(config.CATEGORICAL_FEATURES)))
model_union = Pipeline([
                            ("union", UnionFeature), 
                            ("catboost_model",CustomCatboostRegressor( learning_rate = 0.1, 
                            iterations = 250, 
                            l2_leaf_reg = 0.5, 
                            depth = 10,
                            random_seed = 110,silent = True))])



model_pipeline = TransformedTargetRegressor(model_union,func=np.log,inverse_func=np.exp)