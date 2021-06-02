from sklearn.pipeline import Pipeline,FeatureUnion
from catboost import CatboostRegressor


### custom modules 
from artwork_regression_model.processing import preprocessors as pp 
from artwork_regression_model.config import  config

import logging as logging



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



model_pipeline = Pipeline([
                            ("union", UnionFeature), 
                            ("catboost_model",CatboostRegressor( learning_rate = 0.1, iteration = 250, l2_leaf_reg = 0.5, depth = 10,
                            cat_features=config.CATEGORICAL_FEATURES,random_state=110,silent=True))
                          ])