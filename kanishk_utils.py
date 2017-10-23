__version__: "0.0.1"

from IPython.core.interactiveshell import InteractiveShell
InteractiveShell.ast_node_interactivity = "all"
import pandas as pd
import numpy as np
pd.set_option("display.max_columns",100)

from sklearn.model_selection import train_test_split


def feature_importance(data, model):
    a = pd.concat([pd.DataFrame(data.columns), pd.DataFrame(model.feature_importances_)], axis=1)
    a.columns = ["Variable", "Gini_importance"]
    return a.sort_values(by="Gini_importance", ascending=False)


def univariate(modelBase, target="target"):
    """
    """
    var_start_list = pd.DataFrame(modelBase.dtypes, index=None)

    uniquecnt = modelBase.apply(pd.Series.nunique)
    desc = modelBase.describe().transpose()
    cor = modelBase.select_dtypes(include=["Int64", "float64"]).apply(lambda x: x.corr(modelBase[target])) # watch out for other numeric data types
    zeros = modelBase.apply(lambda x: (x[x == 0].shape[0]/x.shape[0]))
    null = modelBase.apply(lambda x: (x[x.isnull()].shape[0]/x.shape[0]))

    var_start_list = var_start_list.merge(pd.DataFrame(uniquecnt), how="left", left_index=True, right_index=True)
    var_start_list.rename(columns={"0_x": "type", "0_y": "var_vals"}, inplace=True)

    var_start_list = var_start_list.merge(desc[["min", "max", "mean", "50%"]], how="left", left_index=True, right_index=True)

    var_start_list = var_start_list.merge(pd.DataFrame(cor), how="left", left_index=True, 
                                          right_index=True)

    var_start_list = var_start_list.merge(pd.DataFrame(zeros), how="left", left_index=True, right_index=True)
    var_start_list = var_start_list.merge(pd.DataFrame(null), how="left", left_index=True, right_index=True)

    var_start_list.rename(columns = {0: "percentNull", "0_x": "CorrelationWithTarget","0_y": "percentZeros" , "min": "var_min", 
                                     "max": "var_max", "50%": "var_median", "mean": "var_mean"}, inplace=True)
    
    return var_start_list


def train_test_transform(modelBase, target="target", small_sample_floor=0.01, combine_small_samples=True, test_size=0.2, random_state=12345):
    if combine_small_samples:
        for column in modelBase.select_dtypes(["object"]).columns:
            cnt = pd.value_counts(modelBase[column], normalize=True)
            modelBase[column] = np.where(modelBase[column].isin(cnt[cnt < small_sample_floor].index), "small_samples_combined", modelBase[column])

    X, y = modelBase.loc[:, modelBase.columns != target], modelBase[target]
    dvec = DictVectorizer(sparse=False)
    X_dict = dvec.fit_transform(X.transpose().to_dict().values())
    X = pd.DataFrame(X_dict, columns=dvec.get_feature_names(), index=modelBase.index)

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size, random_state=random_state)

    return X_train, X_test, y_train, y_test
