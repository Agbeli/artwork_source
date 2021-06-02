##### import the modules needed 
import numpy as np
import pandas as pd
from sklearn.base import TransformerMixin,BaseEstimator
from sklearn.preprocessing import StandardScaler




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
        ### nothing to learn from the train set of the data. 
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

        ### learn categorical frequency in the train data. 
        
        self.val_encode = {}
        for val in self.variables:
            self.val_encode[val] = X[val].value_counts()
        return self

    def transform(self,X):

        X = X.copy()
        for var in self.variables:
            X[var] = X[var].map(np.log(self.val_encode[var]))

        return X 


class RareImputation(TransformerMixin,BaseEstimator):

    """
    Args:
        Input: lists of variables
        rate : rate of frequency of labels in a given feature.   

    """

    def __init__(self,variables,rate=0.001):

        self.rate = rate 
        if not isinstance(variables,list):

            self.variables = [variables]
        else:
            self.variables = variables

    def fit(self,X,y=None):

        self.find_frequent_list = {}

        for variable in self.variables:

            temp = pd.Series(X[variable].value_counts() / np.float(X.shape[0]))
            self.find_frequent_list[variable] = list(temp[temp>self.rate].index)

        return self 

    def transform(self,X):

        X = X.copy()

        for variable in self.variables:

            X[variable] = np.where(X[variable].isin(self.find_frequent_list[variable]),X[variable],"Rare")

        return X



class ScaleFeatues(TransformerMixin,BaseEstimator):


    def __init__(self,variables):

        if not isinstance(variables,list):

            self.variables = [variables]

        else:

            self.variables = variables

    def fit(self,X,y=None):

        
        self.scale = {}
        for variable in self.variables:
            scalar = StandardScaler()
            self.scale[variable] = scalar.fit(X[[variable]])

        return self

    def transform(self,X):

        X = X.copy()

        for variable in self.variables:

            X[variable] = self.scale[variable].transform(X[[variable]]) 

        return X  



class OutlierCapping(TransformerMixin,BaseEstimator):

    def __init__(self,variables):

        if not isinstance(variables,list):

            self.variables = variables
        
        else:
            self.variables = variables


    def fit(self,X,y = None):

        
        self.features_capping = {}   ### store the lower and upper bound of each floating feature 
        for variable in self.variables:
            self.features_capping[variable] = (X[variable].quantile(0.1),X[variable].quantile(0.9))

        return self 

    def transform(self,X):

        X = X.copy()

        for variable in self.variables:

            X[variable] = np.where(X[variable] > self.features_capping[variable][1],self.features_capping[variable][1],X[variable])
            X[variable] = np.where(X[variable] < self.features_capping[variable][0],self.features_capping[variable][0],X[variable])

        return X 


def targetCapping(y):

    y_ = y.copy()

    lower_bound = y.quantile(0.1)
    upper_bound = y.quantile(0.9)

    y_ = np.where(y_ < lower_bound, lower_bound, y_)
    y_ = np.where(y_ > upper_bound, upper_bound, y_)

    return y_ 







            





