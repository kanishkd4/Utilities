
from IPython.core.interactiveshell import InteractiveShell
InteractiveShell.ast_node_interactivity = "all"
import pandas as pd
import numpy as np
pd.set_option("display.max_columns",100)

from sklearn.model_selection import train_test_split
from sklearn.feature_extraction import DictVectorizer

import openpyxl
from openpyxl.utils.dataframe import dataframe_to_rows



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


def proba_score_performance(actual, prediction, test_set=False, train_bins=0, event=1, target="target"):
    performance = pd.concat([pd.DataFrame(prediction, columns=["Probability_of_Event"]), pd.DataFrame(actual)], axis=1)
    performance.loc[:, target] = np.where(performance.loc[:, target] == event, 1, 0)

    if test_set:
        performance["Probability_of_Event"] = pd.cut(performance.loc[:, "Probability_of_Event"], bins=train_bins, include_lowest=True)
    else:
        seg, train_bins = pd.qcut(performance.loc[:, "Probability_of_Event"].round(12), 10, retbins=True, duplicates="drop")
        train_bins[0] = np.min([0.0, performance.loc[:, "Probability_of_Event"].min()])
        train_bins[train_bins.shape[0]-1] = np.max([1.0, performance.loc[:, "Probability_of_Event"].max()])
        performance["Probability_of_Event"] = pd.cut(performance.loc[:, "Probability_of_Event"], bins=train_bins, include_lowest=True)        

    performance = pd.concat([performance.groupby(by="Probability_of_Event")[target].sum(), 
                     performance.groupby(by="Probability_of_Event")[target].count()], axis=1)
    performance["Probability_of_Event"] = performance.index
    performance.columns = ["Event", "Total", "Probability_of_Event"]
    performance = performance[performance.Total.notnull()]
    performance = performance.append(pd.DataFrame(data={"Event": np.sum(performance.Event), "Total": np.sum(performance.Total), "Probability_of_Event": "All"},index=["All"]), ignore_index=True)

    performance = performance[performance.Probability_of_Event != "All"].sort_values(by="Probability_of_Event", ascending=False).append(performance[performance.Probability_of_Event == "All"])
    performance["Non_event"] = performance.Total - performance.Event
    performance["Cumulative_Non_event"] = performance.Non_event.cumsum()
    performance.loc[performance[performance.Probability_of_Event == "All"].index, "Cumulative_Non_event"] = performance.loc[performance[performance.Probability_of_Event == "All"].index, "Non_event"]
    performance["Cumulative_Event"] = performance.Event.cumsum()
    performance.loc[performance[performance.Probability_of_Event == "All"].index, "Cumulative_Event"] = performance.loc[performance[performance.Probability_of_Event == "All"].index, "Event"]
    performance["Population_%"] = performance.Total/performance[performance.Probability_of_Event == "All"].Total.values
    performance["Cumulative_Non_event_%"] = performance.Cumulative_Non_event/performance[performance.Probability_of_Event == "All"].Cumulative_Non_event.values
    performance["Cumulative_Event_%"] = performance.Cumulative_Event/performance[performance.Probability_of_Event == "All"].Cumulative_Event.values
    performance["Difference"] = performance["Cumulative_Event_%"] - performance["Cumulative_Non_event_%"]
    performance["Event_rate"] = performance.Event/performance.Total
    performance["Gini"] = ((performance["Cumulative_Event_%"]+performance["Cumulative_Event_%"].shift(1).fillna(0))/2)*(performance["Cumulative_Non_event_%"]-performance["Cumulative_Non_event_%"].shift(1).fillna(0))
    performance.loc[performance[performance.Probability_of_Event == "All"].index, "Gini"] = np.nan
    model_KS = np.max(performance[performance.Probability_of_Event != "All"].Difference)*100
    model_Gini = (2*(np.sum(performance[performance.Probability_of_Event != "All"].Gini))-1)*100
    if test_set:
        return performance, model_KS, model_Gini
    else:
        return performance, model_KS, model_Gini, train_bins

def estimator_performance(estimator, X, y, test_set=False, train_bins=0, event=1, target="target"):
    performance = pd.concat([pd.DataFrame(estimator.predict_proba(X), index=X.index), pd.DataFrame(y)], axis=1)
    performance.loc[:, "Probability_of_non_Event"] = 1 - performance.loc[:, event]
    performance.loc[:, target] = np.where(performance.loc[:, target] == event, 1, 0)
    performance.rename(columns={event: "Probability_of_Event"}, inplace=True)
    performance = performance.loc[:, ["Probability_of_Event", "Probability_of_non_Event", target]]

    if test_set:
        performance["Probability_of_Event"] = pd.cut(performance.loc[:, "Probability_of_Event"], bins=train_bins, include_lowest=True)
    else:
        seg, train_bins = pd.qcut(performance.loc[:, "Probability_of_Event"].round(12), 10, retbins=True, duplicates="drop")
        train_bins[0] = np.min([0.0, performance.loc[:, "Probability_of_Event"].min()])
        train_bins[train_bins.shape[0]-1] = np.max([1.0, performance.loc[:, "Probability_of_Event"].max()])
        performance["Probability_of_Event"] = pd.cut(performance.loc[:, "Probability_of_Event"], bins=train_bins, include_lowest=True)        

    performance = pd.concat([performance.groupby(by="Probability_of_Event")[target].sum(), 
                     performance.groupby(by="Probability_of_Event")[target].count()], axis=1)
    performance["Probability_of_Event"] = performance.index
    performance.columns = ["Event", "Total", "Probability_of_Event"]
    performance = performance[performance.Total.notnull()]
    performance = performance.append(pd.DataFrame(data={"Event": np.sum(performance.Event), "Total": np.sum(performance.Total), "Probability_of_Event": "All"},index=["All"]), ignore_index=True)

    performance = performance[performance.Probability_of_Event != "All"].sort_values(by="Probability_of_Event", ascending=False).append(performance[performance.Probability_of_Event == "All"])
    performance["Non_event"] = performance.Total - performance.Event
    performance["Cumulative_Non_event"] = performance.Non_event.cumsum()
    performance.loc[performance[performance.Probability_of_Event == "All"].index, "Cumulative_Non_event"] = performance.loc[performance[performance.Probability_of_Event == "All"].index, "Non_event"]
    performance["Cumulative_Event"] = performance.Event.cumsum()
    performance.loc[performance[performance.Probability_of_Event == "All"].index, "Cumulative_Event"] = performance.loc[performance[performance.Probability_of_Event == "All"].index, "Event"]
    performance["Population_%"] = performance.Total/performance[performance.Probability_of_Event == "All"].Total.values
    performance["Cumulative_Non_event_%"] = performance.Cumulative_Non_event/performance[performance.Probability_of_Event == "All"].Cumulative_Non_event.values
    performance["Cumulative_Event_%"] = performance.Cumulative_Event/performance[performance.Probability_of_Event == "All"].Cumulative_Event.values
    performance["Difference"] = performance["Cumulative_Event_%"] - performance["Cumulative_Non_event_%"]
    performance["Event_rate"] = performance.Event/performance.Total
    performance["Gini"] = ((performance["Cumulative_Event_%"]+performance["Cumulative_Event_%"].shift(1).fillna(0))/2)*(performance["Cumulative_Non_event_%"]-performance["Cumulative_Non_event_%"].shift(1).fillna(0))
    performance.loc[performance[performance.Probability_of_Event == "All"].index, "Gini"] = np.nan
    model_KS = np.max(performance[performance.Probability_of_Event != "All"].Difference)*100
    model_Gini = (2*(np.sum(performance[performance.Probability_of_Event != "All"].Gini))-1)*100
    if test_set:
        return performance, model_KS, model_Gini
    else:
        return performance, model_KS, model_Gini, train_bins

def performance_proba_train_test(estimator, X, y, n_bins=10, bad=1):
    train_performance, train_KS, train_Gini, bins = estimator_performance(estimator=estimator, X=X[0], y=y[0], n_bins=n_bins)
    train_bins = bins
    test_performance, test_KS, test_Gini = estimator_performance(estimator=estimator, X=X[1], y=y[1], test_set=True, train_bins=train_bins[estimator])
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


def write_to_excel(df, excel_path, sheet_name="Sheet1", start_cell=(1, 1), header=True, index=False):
    """
    Write a data frame to an existing excel file. The file must have a sheet with the specific name present
    Parameters
    ----------
    df: pd.DataFrame
        A pandas dataframe
    excel path: raw string literal 
        Give the path as a raw string literal; r"path to file"
    sheet_name: string
        Name of the excel sheet to add the dataframe to
    start_cell: tuple
        The (row, column) of the sheet cell to write the dataframe values to
    header: bool
        If True, the dataframe will be written with headers
    index: book
        If True, the dataframe will be written with the index
    """
    rows = dataframe_to_rows(df, index=index, header=header)
    wb = openpyxl.load_workbook(excel_path)
    sheetx = wb[sheet_name]
    for r_idx, row in enumerate(rows, 1):
        for c_idx, value in enumerate(row, 1):
            sheetx.cell(row=r_idx+start_cell[0]-1, column=c_idx+start_cell[1]-1, value=value)
    wb.save(excel_path)
    wb.close()


