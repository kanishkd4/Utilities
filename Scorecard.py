import pandas as pd
import numpy as np
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import train_test_split
from scipy.stats import chi2_contingency

__authors__ = "Kanishk Dogar"; "Arya Poddar"
__version__ = "0.0.2"


class scorecard:
    """
    Create a credit scorecard using machine learning, supervised binning and
    """

    def __init__(self, modelBase, target="target"):
        """
        """
        self.target = target
        self.modelBase = modelBase

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

    def train_test_split(self, random_state=1, test_size=0.2):
        self.modelBase.drop(list(self.var_start_list[(self.var_start_list["var_vals"] < 2)].index), axis=1, inplace=True)
        self.X, self.y = self.modelBase.loc[:, self.modelBase.columns != self.target], self.modelBase[self.target]
        self.X_train, self.X_test, self.y_train, self.y_test = train_test_split(self.X, self.y, test_size=test_size, random_state=random_state)
        return self


    def WOE_Categorical_fit_transform(self, min_percentage_bin=0.06, small_sample_floor=0.0025):
        WOE_Categorical = pd.DataFrame()
        for column in self.modelBase.select_dtypes(include=["object"]).columns:
            grp = pd.concat([self.X_train.loc[:, column], self.y_train], axis=1)
            test = grp.pivot_table(index=column, values=self.target, aggfunc=[np.sum, len], margins=True)
            grp.rename(columns={column: "col"}, inplace=True)
            test.columns = ["sum", "len"]
            test.index = test.index.map(str)
            test["popPercentage"] = test["len"]/test.loc["All", "len"]
            test["map"] = np.where(test["popPercentage"] < small_sample_floor, "Others", test.index)
            grp[column] = grp.col.map(str).map(test["map"].to_dict())
            test = grp.pivot_table(index=column, values=self.target, aggfunc=[np.sum, len], margins=True)

            test.columns = ["badCnt", "totalCnt"]
            test["goodCnt"] = test["totalCnt"] - test["badCnt"]
            test["popPercentage"] = test["totalCnt"]/test.loc["All", "totalCnt"]
            test["badRate"] = test["badCnt"]/test["totalCnt"]

            test["deleteFlag"] = 0
            test["var"] = column

            test.sort_values(by="badRate", inplace=True)
            test = test.loc[test.index != "All", :].append(test.loc[test.index == "All", :])
            for i in range(len(test.index)-2):
                if test.loc[test.index[i], "popPercentage"] < min_percentage_bin:
                    test.loc[test.index[i+1], "badCnt"] = test.loc[test.index[i], "badCnt"]+test.loc[test.index[i+1], "badCnt"]
                    test.loc[test.index[i+1], "goodCnt"] = test.loc[test.index[i], "goodCnt"]+test.loc[test.index[i+1], "goodCnt"]
                    test.loc[test.index[i+1], "totalCnt"] = test.loc[test.index[i], "totalCnt"]+test.loc[test.index[i+1], "totalCnt"]
                    test.loc[test.index[i], "deleteFlag"] = 1
                    test.rename(index={test.index[i+1]: test.index[i]+","+ test.index[i+1]}, inplace=True)
                    test["popPercentage"] = test["totalCnt"]/test.loc["All", "totalCnt"]
            test = test[test["deleteFlag"] == 0]

            i = len(test.index)-2
            if test.loc[test.index[i], "popPercentage"] < min_percentage_bin:
                test.loc[test.index[i-1], "badCnt"] = test.loc[test.index[i], "badCnt"]+test.loc[test.index[i-1], "badCnt"]
                test.loc[test.index[i-1], "goodCnt"] = test.loc[test.index[i], "goodCnt"]+test.loc[test.index[i-1], "goodCnt"]
                test.loc[test.index[i-1], "totalCnt"] = test.loc[test.index[i], "totalCnt"]+test.loc[test.index[i-1], "totalCnt"]
                test.loc[test.index[i], "deleteFlag"] = 1
                test.rename(index={test.index[i-1]: test.index[i-1]+","+ test.index[i]}, inplace=True)
                test["popPercentage"] = test["totalCnt"]/test.loc["All", "totalCnt"]
            test = test[test["deleteFlag"] == 0]
            
            for i in range(len(test.index)-2):
                if test.loc[test.index[i], "badCnt"] == 0.0:
                    test.loc[test.index[i+1], "badCnt"] = test.loc[test.index[i], "badCnt"]+test.loc[test.index[i+1], "badCnt"]
                    test.loc[test.index[i+1], "goodCnt"] = test.loc[test.index[i], "goodCnt"]+test.loc[test.index[i+1], "goodCnt"]
                    test.loc[test.index[i+1], "totalCnt"] = test.loc[test.index[i], "totalCnt"]+test.loc[test.index[i+1], "totalCnt"]
                    test.loc[test.index[i], "deleteFlag"] = 1
                    test.rename(index={test.index[i+1]: test.index[i]+","+ test.index[i+1]}, inplace=True)
                    test["popPercentage"] = test["totalCnt"]/test.loc["All", "totalCnt"]
            test = test[test["deleteFlag"] == 0]
            
            test["popPercentage"] = test["totalCnt"]/test.loc["All", "totalCnt"]
            test["badRate"] = test["badCnt"]/test["totalCnt"]

            test["badDistribution"] = test["badCnt"]/test.loc["All", "badCnt"]
            test["goodDistribution"] = test["goodCnt"]/test.loc["All", "goodCnt"]
            test["distributedGoodBad"] = test["goodDistribution"] - test["badDistribution"]
            test["WOE"] = np.log(test["goodDistribution"]/test["badDistribution"])
            test["IV"] = test["WOE"]*test["distributedGoodBad"]
            test.loc["All", "IV"] = np.sum(test["IV"])
            WOE_Categorical = WOE_Categorical.append(test)

        WOE_Categorical.drop("deleteFlag", axis=1, inplace=True)
        WOE_Categorical = WOE_Categorical[["var", "badCnt", "totalCnt", "goodCnt","popPercentage", "badDistribution", "goodDistribution", "distributedGoodBad","WOE", "IV", "badRate"]]
        WOE_Categorical["cuts"] = WOE_Categorical.index
        WOE_Categorical["cuts"] = WOE_Categorical.cuts.str.split(",")
        self.WOE_Categorical = WOE_Categorical

        trainC = self.X_train.select_dtypes(include=["object"]).copy()
        for column in trainC.columns:
            cut = WOE_Categorical[WOE_Categorical["var"] == column] # subset data for one variable
            cut = cut.loc[cut.index != "All", :]                    # remove row All
            cut["cuts"] = cut.index                                 # create a column for all cuts
            cut = cut.loc[:, ["var", "cuts"]]                       # subset only 2 columns
            cut["cuts"] = cut.cuts.str.split(",")                   # convert cuts to a list of cuts from a string
            for i in trainC.index:                                  # override trainC with the bins whereever the value is in the bin in cut
                for j in cut.cuts:
                    if str(trainC.loc[i, column]) in j:
                        trainC.loc[i, column] = str(j)
                    if "Others" in j:
                        oth = j
                if trainC.loc[i, column] not in list(cut.cuts.map(str)):
                    trainC.loc[i, column] = str(oth)
        self.trainC = trainC
        return self

    def treeBinning(self, trainN, i=6, column="column", cutPoint=0.0):
        tree = DecisionTreeClassifier(max_leaf_nodes=i, min_samples_leaf=np.int(np.rint(trainN.shape[0]*0.06)))
        X_select = pd.DataFrame(trainN[trainN[column] > cutPoint][column]) # this filter assumes all values < 0 as special and bins separately
        tree.fit(X_select, self.y_train[X_select.index])
        X_select["Node"] = tree.apply(X_select)
        X_select = X_select.append(pd.DataFrame({column: trainN[trainN[column] <= cutPoint][column], "Node": -1}))
        X_select = pd.concat([X_select, self.y_train], axis=1)
        test = pd.concat([X_select.pivot_table(index="Node", values=column, aggfunc=[np.min, np.max], margins=True),
                         X_select.pivot_table(index="Node", values=self.target, aggfunc=[np.sum, len], margins=True)], axis=1)
        test.rename(columns={"sum": "badCnt", "len": "totalCnt"}, inplace=True)
        test["goodCnt"] = test["totalCnt"] - test["badCnt"]
        test["popPercentage"] = test["totalCnt"]/test.loc["All", "totalCnt"] * 100
        test["badRate"] = test["badCnt"]/test["totalCnt"] * 100
        test.columns = ["amin", "amax", "badCnt", "totalCnt", "goodCnt", "popPercentage", "badRate"]
        test = test.sort_values(by="amax")
        test["badSign"] = np.sign(test.badRate - test.badRate.shift(-1))
        test.iloc[test.shape[0]-2, 7] = np.nan
        test["badDistribution"] = test["badCnt"]/test.loc["All", "badCnt"]
        test["goodDistribution"] = test["goodCnt"]/test.loc["All", "goodCnt"]
        test["distributedGoodBad"] = test["goodDistribution"] - test["badDistribution"]
        test["WOE"] = np.log(test["goodDistribution"]/test["badDistribution"])
        test["IV"] = test["WOE"]*test["distributedGoodBad"] * 100
        test.loc["All", "IV"] = np.sum(test["IV"])    
        test["column"] = column
        return test[["amin", "amax", "popPercentage", "IV", "badRate", "badSign", "column"]]

    def WOE_Numeric_fit_transform(self, bins=6, special_values=[-1], treeCut=0.0):
        trainN = self.X_train.select_dtypes(include=["int64", "float64"]).copy()
        WOE_N = pd.DataFrame()
        for column in trainN.columns:
            i = bins
            if (trainN[column].nunique() >= 2) & (trainN[trainN[column] > treeCut].shape[0] > 0):
                while i > 1:
                    test = self.treeBinning(i=i, column=column, trainN=trainN, cutPoint=treeCut)
                    if test[test.index != -1]["badSign"].nunique() == 1:
                        break
                    else:
                        i = i-1
                WOE_N = WOE_N.append(test)

        for column in trainN.columns:
            w = WOE_N[(WOE_N["column"] == column) & (WOE_N.index != "All") & (WOE_N["badRate"] > 0) & (WOE_N["badRate"] < 100)]
            w = w.iloc[range(w.shape[0] - 1), :]
            cutPoints = sorted(list(set([-np.inf] + special_values + list(w.amax) + [np.inf])))
            trainN.loc[:, column] = pd.cut(trainN[column], [x for x in cutPoints if x <= np.min(special_values) or x >= np.max(special_values)], right=True).astype("object")

        WOE_Numerical = pd.DataFrame()
        for column in trainN.columns:
            test = pd.concat([trainN.loc[:, column], self.y_train], axis=1).pivot_table(index=column, values=self.target, aggfunc=[np.sum, len], margins=True)
            test.columns = ["badCnt", "totalCnt"]
            test["goodCnt"] = test["totalCnt"] - test["badCnt"]
            test["var"] = column
            test["popPercentage"] = test["totalCnt"]/test.loc["All", "totalCnt"]
            test["badRate"] = test["badCnt"]/test["totalCnt"]
            test["badDistribution"] = test["badCnt"]/test.loc["All", "badCnt"]
            test["goodDistribution"] = test["goodCnt"]/test.loc["All", "goodCnt"]
            test["distributedGoodBad"] = test["goodDistribution"] - test["badDistribution"]
            test["WOE"] = np.log(test["goodDistribution"]/test["badDistribution"])
            test["IV"] = test["WOE"]*test["distributedGoodBad"]
            test.loc["All", "IV"] = np.sum(test[np.isfinite(test["IV"])]["IV"])
            WOE_Numerical = WOE_Numerical.append(test)

        WOE_Numerical = WOE_Numerical[["var", "badCnt", "totalCnt", "goodCnt","popPercentage", "badDistribution", "goodDistribution", "distributedGoodBad","WOE", "IV", "badRate"]]
        WOE_Numerical["cuts"] = WOE_Numerical.index
        self.WOE_N = WOE_N
        self.WOE_Numerical = WOE_Numerical
        self.trainN = trainN
        return self

    def WOE_Final_transform(self):
        self.WOE_Final = self.WOE_Numerical.append(self.WOE_Categorical)
        self.WOE_All = self.WOE_Final[self.WOE_Final.index == "All"].set_index("var")
        return self

    def IV_filter_transform(self, IV_min=0.02):
        X_train_binned = pd.concat([self.trainC, self.trainN], axis=1)
        X_train_binned.drop(list(self.WOE_Final[(self.WOE_Final.index == "All") & (self.WOE_Final["IV"] < IV_min)]["var"]), axis=1, inplace=True)
        self.X_train_binned = X_train_binned
        return self

    def cramersV(self, cv_max=0.6):
        cramersV = pd.DataFrame()
        for c1 in self.X_train_binned.columns:
            for c2 in self.X_train_binned.columns:
                if (c1 != c2) & (self.X_train_binned.columns.get_loc(c2) > self.X_train_binned.columns.get_loc(c1)):
                    a = chi2_contingency(pd.crosstab(self.X_train_binned[c1], self.X_train_binned[c2]), correction=False)[0]
                    b = len(self.X_train_binned[c1]) * (np.min([self.X_train_binned[c1].nunique(), self.X_train_binned[c2].nunique()])-1)
                    d = {"var1": c1, "var2": c2, "CramersV": np.sqrt(a/b)}
                    cramersV = cramersV.append(pd.DataFrame(d, index=[0]), ignore_index=True)

        cramersV["var1_IV"] = cramersV.var1.map(self.WOE_All["IV"].to_dict())
        cramersV["var2_IV"] = cramersV.var2.map(self.WOE_All["IV"].to_dict())
        s = cramersV[cramersV["CramersV"] > 0.6].sort_values(by="CramersV", ascending=False)

        var_drop = []
        while len(s) > 0:
            if s.iloc[0, 3] > s.iloc[0, 4]:
                var_drop = var_drop + [s.iloc[0, 2]]
            else:
                var_drop = var_drop + [s.iloc[0, 1]]
            s = s[(s["var1"] != var_drop[-1]) & (s["var2"] != var_drop[-1])]
        self.cramersV_var_drop = var_drop
        self.fSelection = self.X_train_binned.drop(var_drop, axis=1)


    def Modify_Categorical(self, structure):
        struct = pd.DataFrame(pd.DataFrame.from_dict(structure, orient="index").stack()).reset_index()[["level_0", 0]]
        struct.columns = ["var", "cuts"]
        
        cat_mod = self.X_train[list(structure.keys())]
        for column in cat_mod.columns:
            look = struct[struct["var"] == column]
            for i in cat_mod.index:
                for j in look.cuts:
                    if str(cat_mod.loc[i, column]) in j:
                        cat_mod.loc[i, column] = str(j)
                    if "Others" in j:
                        oth = j
                if cat_mod.loc[i, column] not in list(look.cuts.map(str)):
                    cat_mod.loc[i, column] = str(oth)

        self.fSelection.drop(list(structure.keys()), axis=1, inplace=True)
        self.fSelection = pd.concat([self.fSelection, cat_mod], axis=1)


    def WOE_Scaling(self):
        WOE_Scaling = pd.DataFrame()
        for column in self.fSelection.columns:
            test = pd.concat([self.fSelection.loc[:, column], self.y_train], axis=1).pivot_table(index=column, values=self.target, aggfunc=[np.sum, len], margins=True)
            test.columns = ["badCnt", "totalCnt"]
            test["goodCnt"] = test["totalCnt"] - test["badCnt"]
            test["var"] = column
            test["popPercentage"] = test["totalCnt"]/test.loc["All", "totalCnt"]
            test["badRate"] = test["badCnt"]/test["totalCnt"]
            test["badDistribution"] = test["badCnt"]/test.loc["All", "badCnt"]
            test["goodDistribution"] = test["goodCnt"]/test.loc["All", "goodCnt"]
            test["distributedGoodBad"] = test["goodDistribution"] - test["badDistribution"]
            test["WOE"] = np.log(test["goodDistribution"]/test["badDistribution"])
            test["IV"] = test["WOE"]*test["distributedGoodBad"]
            test.loc["All", "IV"] = np.sum(test[np.isfinite(test["IV"])]["IV"])
            WOE_Scaling = WOE_Scaling.append(test)

        WOE_Scaling = WOE_Scaling[["var", "badCnt", "totalCnt", "goodCnt","popPercentage", "badDistribution", "goodDistribution", "distributedGoodBad","WOE", "IV", "badRate"]]
        WOE_Scaling["cuts"] = WOE_Scaling.index
        self.WOE_Scaling = WOE_Scaling












