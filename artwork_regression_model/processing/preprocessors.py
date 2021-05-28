##### import the modules needed 
import numpy as np
import pandas as pd
from sklearn.base import TransformerMixin,BaseEstimator




class LogTransformer(TransformerMixin,BaseEstimator):

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


