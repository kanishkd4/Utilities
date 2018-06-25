
from IPython.core.interactiveshell import InteractiveShell
InteractiveShell.ast_node_interactivity = "all"
import pandas as pd
import numpy as np
pd.set_option("display.max_columns",100)

from sklearn.model_selection import train_test_split
from sklearn.feature_extraction import DictVectorizer



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


def performance_proba(estimator, X, y, test_set=False, train_bins=0, bad=1, n_bins=10, target="target", event=1):
    performance = pd.concat([pd.DataFrame(estimator.predict_proba(X), index=X.index), pd.DataFrame(y)], axis=1)
    if test_set:
        performance["Probability_of_bad"] = pd.cut(performance.loc[:, event], bins=train_bins)
    else:
        seg, train_bins = pd.qcut(performance.loc[:, 1].round(12), n_bins, retbins=True, duplicates="drop")
        performance["Probability_of_bad"] = pd.cut(performance.loc[:, event], bins=train_bins)

    performance = performance.pivot_table(index="Probability_of_bad", values=target, aggfunc=[np.sum, len], margins=True).reset_index()
    performance.columns = ["Probability_of_bad", "Bad", "Total"]
    performance = performance[performance.Total.notnull()]

    performance = performance[performance.Probability_of_bad != "All"].sort_values(by="Probability_of_bad", ascending=False).append(performance[performance.Probability_of_bad == "All"])
    performance["Good"] = performance.Total - performance.Bad
    performance["Cumulative_good"] = performance.Good.cumsum()
    performance.loc[performance[performance.Probability_of_bad == "All"].index, "Cumulative_good"] = performance.loc[performance[performance.Probability_of_bad == "All"].index, "Good"]
    performance["Cumulative_bad"] = performance.Bad.cumsum()
    performance.loc[performance[performance.Probability_of_bad == "All"].index, "Cumulative_bad"] = performance.loc[performance[performance.Probability_of_bad == "All"].index, "Bad"]
    performance["Population_%"] = performance.Total/performance[performance.Probability_of_bad == "All"].Total.values
    performance["Cumulative_good_%"] = performance.Cumulative_good/performance[performance.Probability_of_bad == "All"].Cumulative_good.values
    performance["Cumulative_bad_%"] = performance.Cumulative_bad/performance[performance.Probability_of_bad == "All"].Cumulative_bad.values
    performance["Difference"] = performance["Cumulative_bad_%"] - performance["Cumulative_good_%"]
    performance["Bad_rate"] = performance.Bad/performance.Total
    performance["Gini"] = ((performance["Cumulative_bad_%"]+performance["Cumulative_bad_%"].shift(1).fillna(0))/2)*(performance["Cumulative_good_%"]-performance["Cumulative_good_%"].shift(1).fillna(0))
    performance.loc[performance[performance.Probability_of_bad == "All"].index, "Gini"] = np.nan
    model_KS = np.max(performance[performance.Probability_of_bad != "All"].Difference)*100
    model_Gini = (2*(np.sum(performance[performance.Probability_of_bad != "All"].Gini))-1)*100
    if test_set:
        return performance, model_KS, model_Gini
    else:
        return performance, model_KS, model_Gini, train_bins

def performance_proba_train_test(estimator, X, y, n_bins=10, bad=1):
    train_performance, train_KS, train_Gini, bins = performance_proba(estimator=estimator, X=X[0], y=y[0], n_bins=n_bins)
    train_bins = bins
    test_performance, test_KS, test_Gini = performance_proba(estimator=estimator, X=X[1], y=y[1], test_set=True, train_bins=train_bins[estimator])
    return train_performance_proba, train_KS, train_Gini, test_performance_proba, test_KS, test_Gini


def convert_mutli_to_single_index(data, filler="_"):
    cols = []
    for i in range(len(data.columns.get_level_values(0))):
        name = ""
        for j in range(len(data.columns.levels)-1, -1, -1):
            nametemp = data.columns.get_level_values(j)[i]
            name = nametemp+filler+name
        cols = cols+[name]
    data.columns = cols
    return data.head(2)
