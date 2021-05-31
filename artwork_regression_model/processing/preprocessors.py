##### import the modules needed 
import numpy as np
import pandas as pd
from sklearn.base import TransformerMixin,BaseEstimator




class LogTransformer(TransformerMixin,BaseEstimator):

    """
    The goal of this class is to log transform the floating features in the dataset. 
    To scale the feature to follow a gaussian distribution. 
    Args: 
        Variables: list of features to log transform 

    """

    def __init__(self,variables=None):

        if not isinstance(variables,list):
            self.variables = [variables]
        else:
            self.variables = variables

    def fit(self,X,y=None):
        return self 

    def transform(self,X=None):

        X = X.copy()
        for variable in self.variables:
            X[variable] = np.log(X[variable])
        
        return X  


class DropFeature(TransformerMixin,BaseEstimator):

    """
    Drop all selected features not required in the model. 
    Args: 
        Input: lists of variables to drop from the dataframe.  
    """

    def __init__(self,variables):

        if not isinstance(variables,list):

            self.variables = [variables]

        else:
            self.variables = variables

    def fit(self,X,y=None):
        return self

    def transform(self,X):

        X = X.copy()

        X = X.drop(self.variables,axis=1)
        return X 


class FrequencyEncoding(TransformerMixin,BaseEstimator):

    """
    Encode the frequency of a given feature in the dataframe. 

    """
    def __init__(self,variables):

        if not isinstance(variables,list):
            self.variables = [variables]

        else:
            self.variables = variables 

    def fit(self,X,y=None):

        X = X.copy()
        self.val_encode = {}
        for val in self.variables:
            self.val_encode[val] = X[val].value_counts()
        return self

    def transform(self,X):

        X = X.copy()
        for var in self.variables:
            X[var] = X[var].map(np.log(self.val_encode[var]))

        return X 





