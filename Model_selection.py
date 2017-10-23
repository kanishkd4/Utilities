import pandas as pd
import numpy as np
from sklearn.feature_extraction import DictVectorizer
from sklearn.model_selection import train_test_split


__authors__ = "Kanishk Dogar", "Arya Poddar"
__version__ = "0.0.1"

class MS:
    """
    Create and summarise different scikit learn models
    MS stores data and applies multiple models to a training and test set based on models and hyperparameters selected.

    Parameters
    ----------
    target: string
        name of the target column
    modelBase: pandas dataframe
        the name of the dataframe in which all the data is stored
    """
    def __init__(self, modelBase, target="target", small_sample_floor=0.01, combine_small_samples=True, test_size=0.2, random_state=12345, univariate_test=True):
        self.target = target
        self.modelBase = modelBase
        self.small_sample_floor = small_sample_floor
        self.combine_small_samples = combine_small_samples
        self.test_size = test_size
        self.random_state = random_state
        self.univariate_test = univariate_test

        if self.univariate_test:
            self.univariate()
        self.train_test_transform()

    def univariate(self):
        """
        """
        var_start_list = pd.DataFrame(self.modelBase.dtypes, index=None)

        uniquecnt = self.modelBase.apply(pd.Series.nunique)
        desc = self.modelBase.describe().transpose()
        cor = self.modelBase.select_dtypes(include=["Int64", "float64"]).apply(lambda x: x.corr(self.modelBase[self.target])) # watch out for other numeric data types
        zeros = self.modelBase.apply(lambda x: (x[x == 0].shape[0]/x.shape[0]))
        null = self.modelBase.apply(lambda x: (x[x.isnull()].shape[0]/x.shape[0]))

        var_start_list = var_start_list.merge(pd.DataFrame(uniquecnt), how="left", left_index=True, right_index=True)
        var_start_list.rename(columns={"0_x": "type", "0_y": "var_vals"}, inplace=True)

        var_start_list = var_start_list.merge(desc[["min", "max", "mean", "50%"]], how="left", left_index=True, right_index=True)

        var_start_list = var_start_list.merge(pd.DataFrame(cor), how="left", left_index=True, 
                                              right_index=True)

        var_start_list = var_start_list.merge(pd.DataFrame(zeros), how="left", left_index=True, right_index=True)
        var_start_list = var_start_list.merge(pd.DataFrame(null), how="left", left_index=True, right_index=True)

        var_start_list.rename(columns = {0: "percentNull", "0_x": "CorrelationWithTarget","0_y": "percentZeros" , "min": "var_min", 
                                         "max": "var_max", "50%": "var_median", "mean": "var_mean"}, inplace=True)
        self.var_start_list = var_start_list

        return self

    def train_test_transform(self):
        if self.univariate:
            self.modelBase.drop(list(self.var_start_list[(self.var_start_list["var_vals"] < 2)].index), axis=1, inplace=True)

        if self.combine_small_samples:
            for column in self.modelBase.select_dtypes(["object"]).columns:
                cnt = pd.value_counts(self.modelBase[column], normalize=True)
                self.modelBase[column] = np.where(self.modelBase[column].isin(cnt[cnt < self.small_sample_floor].index), "small_samples_combined", self.modelBase[column])

        self.X, self.y = self.modelBase.loc[:, self.modelBase.columns != self.target], self.modelBase[self.target]
        self.dvec = DictVectorizer(sparse=False)
        X_dict = self.dvec.fit_transform(self.X.transpose().to_dict().values())
        self.X = pd.DataFrame(X_dict, columns=self.dvec.get_feature_names(), index=self.modelBase.index)

        self.X_train, self.X_test, self.y_train, self.y_test = train_test_split(self.X, self.y, test_size=self.test_size, random_state=self.random_state)




