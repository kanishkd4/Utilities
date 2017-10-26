import pandas as pd
import numpy as np
from sklearn.feature_extraction import DictVectorizer
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, confusion_matrix, precision_score, f1_score
from sklearn.model_selection import learning_curve
from sklearn.ensemble import GradientBoostingClassifier, RandomForestClassifier
from sklearn.svm import SVC
from sklearn.naive_bayes import GaussianNB
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import RandomizedSearchCV, GridSearchCV
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import make_pipeline
from sklearn.ensemble import VotingClassifier

import bokeh
from bokeh.plotting import figure, show, output_notebook, gridplot
from bokeh.models import Range1d, Legend, Title, Label


__authors__ = "Kanishk Dogar", "Arya Poddar"
__version__ = "0.0.4"

# option to add scorer added
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
    def __init__(self, modelBase, target="target", small_sample_floor=0.01, combine_small_samples=True, test_size=0.2, random_state=12345, univariate_test=True,
                n_jobs=-1, learning_curve_train_sizes=[0.05, 0.1, 0.2, 0.4, 0.75, 1], fast=True, CV=5, verbose=1, automated=False, scoring="accuracy"):
        self.target = target
        self.modelBase = modelBase
        self.small_sample_floor = small_sample_floor
        self.combine_small_samples = combine_small_samples
        self.test_size = test_size
        self.random_state = random_state
        self.univariate_test = univariate_test
        self.n_jobs = n_jobs
        self.train_sizes = learning_curve_train_sizes
        self.fast = fast
        self.CV = CV
        self.verbose = verbose
        self.scoring = scoring


        if self.univariate_test:
            self.univariate()
        self.train_test_transform()

        if automated:
            self.fit_models()
            self.apply_models()
            self.evaluate_models()


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

        return self.var_start_list

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

    def fit_models(self, models=["gbm", "random_forest", "svm", "naive_bayes", "logistic_regression"]):
        self.models = models
        self.curve = {}
        estimators = {}
        curve = {}
        n_cv = {}
        param_grid = {}
        clf = {}
        for estimator in models:
            if estimator == "gbm":
                estimators[estimator] = GradientBoostingClassifier()
            elif estimator == "random_forest":
                estimators[estimator] = RandomForestClassifier()
            elif estimator == "svm":
                estimators[estimator] = SVC(random_state=1, probability=True)
            elif estimator == "naive_bayes":
                estimators[estimator] = GaussianNB()
            elif estimator == "logistic_regression":
                estimators[estimator] = LogisticRegression()
            else:
                print("method not applicable for model")

        for estimator in models:
            if estimator in ["gbm", "random_forest", "svm", "logistic_regression"]:
                train_sizes, train_scores, test_scores = learning_curve(estimators[estimator], self.X_train, self.y_train, cv=5, n_jobs=self.n_jobs, train_sizes=self.train_sizes)
                curve = pd.concat([pd.DataFrame(train_sizes), pd.DataFrame(np.mean(train_scores, axis=1)), 
                           pd.DataFrame(np.mean(test_scores, axis=1)), pd.DataFrame(np.std(train_scores, axis=1)), 
                           pd.DataFrame(np.std(test_scores, axis=1))], axis=1)
                curve.columns = ["training_sample_size", "mean_train_accuracy", "mean_test_accuracy", "std_train_accuracy", "std_test_accuracy"]
                self.curve[estimator] = curve
        
                if self.fast:
                    n_cv[estimator] = np.min([self.X_train.shape[0], 50000])
                else:
                    n_cv[estimator] = self.X_train.shape[0]

                X_train_cv = self.X_train.sample(n=np.int(np.rint(n_cv[estimator])))
                y_train_cv = self.y_train[self.y_train.index.isin(X_train_cv.index)]
                y_train_cv = y_train_cv.reindex(X_train_cv.index)
                for estimator in models:
                    if estimator == "gbm":
                        param_grid[estimator] = {"learning_rate":[0.001, 0.01, 0.1, 1], "n_estimators":[10, 100, 500], "min_weight_fraction_leaf":[0.06, 0.1], "max_depth":[3, 10, 20, 40, 80]}
                    elif estimator == "random_forest":
                        param_grid[estimator] = {"n_estimators":[10, 100, 200, 500], "min_weight_fraction_leaf":[0.06, 0.1], "max_depth":[3, 10, 20, 40, 80], "max_features":["auto", "sqrt", "log2", None]}
                    elif estimator == "svm":
                        param_grid[estimator+"1"] = {"svc__kernel": ["rbf", "sigmoid", "linear", "poly"]}
                    elif estimator == "logistic_regression":
                        param_grid[estimator+"1"] = {"logisticregression__C":[0.001, 0.01, 0.1, 1, 10]}

        for estimator in models:
            if estimator in ["gbm", "random_forest"]:
                clf[estimator] = RandomizedSearchCV(estimators[estimator], param_grid[estimator], verbose=self.verbose, cv=self.CV, n_jobs=-1, n_iter=100, return_train_score=False, scoring=self.scoring)
                clf[estimator].fit(X_train_cv, y_train_cv)
                print(estimator+" best CV score:", clf[estimator].best_score_)
                if estimator == "gbm":
                    estimators[estimator].set_params(learning_rate=clf[estimator].best_params_["learning_rate"], n_estimators=clf[estimator].best_params_["n_estimators"], max_depth=clf[estimator].best_params_["max_depth"], min_weight_fraction_leaf=clf[estimator].best_params_["min_weight_fraction_leaf"])
                elif estimator == "random_forest":
                    estimators[estimator].set_params(n_estimators=clf[estimator].best_params_["n_estimators"], max_depth=clf[estimator].best_params_["max_depth"], min_weight_fraction_leaf=clf[estimator].best_params_["min_weight_fraction_leaf"], max_features=clf[estimator].best_params_["max_features"])

                estimators[estimator].fit(self.X_train, self.y_train)
            
            elif estimator in ["svm", "logistic_regression"]:
                estimators[estimator] = make_pipeline(StandardScaler(), estimators[estimator])
                clf[estimator+"1"] = GridSearchCV(estimators[estimator], param_grid[estimator+"1"], verbose=self.verbose, cv=self.CV, n_jobs=self.n_jobs, return_train_score=False, scoring=self.scoring)
                clf[estimator+"1"].fit(X_train_cv, y_train_cv)

                if estimator == "logistic_regression":
                    estimators[estimator].set_params(logisticregression__C=clf[estimator+"1"].best_params_["logisticregression__C"])
                    estimators[estimator].fit(self.X_train, self.y_train)

                elif estimator == "svm":
                    estimators[estimator] = make_pipeline(StandardScaler(), SVC(random_state=1, kernel=clf[estimator+"1"].best_params_["svc__kernel"], probability=True))
                    if clf[estimator+"1"].best_params_["svc__kernel"] == "linear":
                        param_grid[estimator+"2"] = {"svc__C": [0.001, 0.01, 0.1, 1.0, 10.0, 100]}
                    else:
                        param_grid[estimator+"2"] = {"svc__C": [0.001, 0.01, 0.1, 1.0, 10.0, 100], "svc__gamma": [0.001, 0.01, 0.1, 1.0, 10.0, 100]}
                    
                    clf[estimator+"2"] = GridSearchCV(estimators[estimator], param_grid[estimator+"2"], verbose=self.verbose, cv=self.CV, n_jobs=self.n_jobs, return_train_score=False, scoring=self.scoring)
                    clf[estimator+"2"].fit(X_train_cv, y_train_cv)

                    if clf[estimator+"1"].best_params_["svc__kernel"] == "linear":
                        estimators[estimator] = make_pipeline(StandardScaler(), SVC(random_state=1, kernel=clf[estimator+"1"].best_params_["svc__kernel"], 
                                                                       C=clf[estimator+"2"].best_params_["svc__C"], probability=True))
                    else:
                        estimators[estimator] = make_pipeline(StandardScaler(), SVC(random_state=1, kernel=clf[estimator+"1"].best_params_["svc__kernel"], 
                                                                       C=clf[estimator+"2"].best_params_["svc__C"], gamma=clf[estimator+"2"].best_params_["svc__gamma"], probability=True))

                    print(estimator+" best CV score:", clf[estimator+"2"].best_score_)
                    estimators[estimator].fit(self.X_train, self.y_train)

            elif estimator == "naive_bayes":
                estimators[estimator].fit(self.X_train, self.y_train)


            print(f"{estimator} fitting complete")

        self.estimators = estimators
        self.n_cv = n_cv
        self.clf = clf

    def apply_models(self):
        trainPred = {}
        testPred = {}
        
        for estimator in self.models:
            trainPred[estimator] = self.estimators[estimator].predict(self.X_train)
            testPred[estimator] = self.estimators[estimator].predict(self.X_test)

            train_accuracy = accuracy_score(self.y_train, trainPred[estimator])*100
            print(f"{estimator} Training accuracy:{train_accuracy:{4}.{4}}%")
            test_accuracy = accuracy_score(self.y_test, testPred[estimator])*100
            print(f"{estimator} Test accuracy:{test_accuracy:{4}.{4}}%")

            train_precision = precision_score(self.y_train, trainPred[estimator])*100
            print(f"{estimator} Training precision:{train_precision:{4}.{4}}%")
            test_precision = precision_score(self.y_test, testPred[estimator])*100
            print(f"{estimator} Test precision:{test_precision:{4}.{4}}%")

            self.trainPred, self.testPred = trainPred, testPred

    def evaluate_models(self):
        Gini_table = {}
        KS = {}
        Gini = {}
        plots = {}
        self.train_bins = {}

        for estimator in self.models:
            Gini_table[estimator+"_train"], KS[estimator+"_train"], Gini[estimator+"_train"], Gini_table[estimator+"_test"], KS[estimator+"_test"], Gini[estimator+"_test"] = self.performance_train_test(estimator=self.estimators[estimator], X=[self.X_train, self.X_test], y=[self.y_train, self.y_test])
            plots[estimator] = self.plot_performance([Gini_table[estimator+"_train"], Gini_table[estimator+"_test"]], [KS[estimator+"_train"], KS[estimator+"_test"]], [Gini[estimator+"_train"], Gini[estimator+"_test"]], name=estimator.upper())
            show(plots[estimator])
        self.Gini_table, self.KS, self.Gini, self.plots = Gini_table, KS, Gini, plots

    def performance(self, estimator, X, y, test_set=False, train_bins=0):
        performance = pd.concat([pd.DataFrame(estimator.predict_proba(X), index=X.index), pd.DataFrame(y)], axis=1)
        if test_set:
            performance["Probability_of_1"] = pd.cut(performance.loc[:, 1], bins=train_bins)
        else:
            seg, train_bins = pd.qcut(performance.loc[:, 1].round(12), 10, retbins=True, duplicates="drop")
            performance["Probability_of_1"] = pd.cut(performance.loc[:, 1], bins=train_bins)        

        performance = performance.pivot_table(index="Probability_of_1", values=self.target, aggfunc=[np.sum, len], margins=True).reset_index()
        performance.columns = ["Probability_of_1", "Bad", "Total"]
        performance = performance[performance.Total.notnull()]

        performance = performance[performance.Probability_of_1 != "All"].sort_values(by="Bad", ascending=False).append(performance[performance.Probability_of_1 == "All"])
        performance["Good"] = performance.Total - performance.Bad
        performance["Cumulative_good"] = performance.Good.cumsum()
        performance.loc[performance[performance.Probability_of_1 == "All"].index, "Cumulative_good"] = performance.loc[performance[performance.Probability_of_1 == "All"].index, "Good"]
        performance["Cumulative_bad"] = performance.Bad.cumsum()
        performance.loc[performance[performance.Probability_of_1 == "All"].index, "Cumulative_bad"] = performance.loc[performance[performance.Probability_of_1 == "All"].index, "Bad"]
        performance["Population_%"] = performance.Total/performance[performance.Probability_of_1 == "All"].Total.values
        performance["Cumulative_good_%"] = performance.Cumulative_good/performance[performance.Probability_of_1 == "All"].Cumulative_good.values
        performance["Cumulative_bad_%"] = performance.Cumulative_bad/performance[performance.Probability_of_1 == "All"].Cumulative_bad.values
        performance["Difference"] = performance["Cumulative_bad_%"] - performance["Cumulative_good_%"]
        performance["Bad_rate"] = performance.Bad/performance.Total
        performance["Gini"] = ((performance["Cumulative_bad_%"]+performance["Cumulative_bad_%"].shift(1).fillna(0))/2)*(performance["Cumulative_good_%"]-performance["Cumulative_good_%"].shift(1).fillna(0))
        performance.loc[performance[performance.Probability_of_1 == "All"].index, "Gini"] = np.nan
        model_KS = np.max(performance[performance.Probability_of_1 != "All"].Difference)*100
        model_Gini = (2*(np.sum(performance[performance.Probability_of_1 != "All"].Gini))-1)*100
        if test_set:
            return performance, model_KS, model_Gini
        else:
            return performance, model_KS, model_Gini, train_bins

    def performance_train_test(self, estimator, X, y):
        train_performance, train_KS, train_Gini, bins = self.performance(estimator=estimator, X=X[0], y=y[0])
        self.train_bins[estimator] = bins
        test_performance, test_KS, test_Gini = self.performance(estimator=estimator, X=X[1], y=y[1], test_set=True, train_bins=self.train_bins[estimator])
        return train_performance, train_KS, train_Gini, test_performance, test_KS, test_Gini

    def plot_performance(self, performance, ks, gini, name=""):
        plot_data_train = performance[0][performance[0].Probability_of_1 != "All"][["Cumulative_good_%", "Cumulative_bad_%"]]
        plot_data_train = pd.DataFrame(data={"Cumulative_good_%": 0, "Cumulative_bad_%": 0}, index=[100]).append(plot_data_train)
        plot_data_test = performance[1][performance[1].Probability_of_1 != "All"][["Cumulative_good_%", "Cumulative_bad_%"]]
        plot_data_test = pd.DataFrame(data={"Cumulative_good_%": 0, "Cumulative_bad_%": 0}, index=[100]).append(plot_data_test)

        p = figure(plot_width=600, plot_height=400, title=f"Gini curve for {name}", )
        p.line(x=plot_data_train["Cumulative_good_%"], y=plot_data_train["Cumulative_bad_%"], legend = "Delinquent distribution - training", line_width=2, line_color="#053C6D", line_alpha=0.5)
        p.line(x=plot_data_test["Cumulative_good_%"], y=plot_data_test["Cumulative_bad_%"], legend = "Delinquent distribution - test", line_width=2, line_color="#97291E", line_alpha=0.5)
        p.line(x=plot_data_train["Cumulative_good_%"], y=plot_data_train["Cumulative_good_%"], line_color="#E46713", line_width=2)

        p.x_range = Range1d(0, 1)
        p.y_range = Range1d(0, 1)
        p.legend.location = "bottom_right"
        p.xaxis.axis_label = "Cumulative Good"
        p.yaxis.axis_label = "Cumulative Bad"

        KS_train = Label(x=70, y=-20, x_units='screen', y_units='screen',
                         text=f"training KS: {ks[0]:{4}.{4}}%", render_mode='css',border_line_alpha=0.0,
                         background_fill_color='white', background_fill_alpha=1.0)
        Gini_train = Label(x=250, y=-20, x_units='screen', y_units='screen',
                           text=f"training Gini: {gini[0]:{4}.{4}}%", render_mode='css',border_line_alpha=0.0,
                           background_fill_color='white', background_fill_alpha=1.0)
        
        KS_test = Label(x=70, y=-40, x_units='screen', y_units='screen',
                         text=f"test KS: {ks[1]:{4}.{4}}%", render_mode='css',border_line_alpha=0.0,
                         background_fill_color='white', background_fill_alpha=1.0)
        Gini_test = Label(x=250, y=-40, x_units='screen', y_units='screen',
                           text=f"test Gini: {gini[1]:{4}.{4}}%", render_mode='css',border_line_alpha=0.0,
                           background_fill_color='white', background_fill_alpha=1.0)
        p.add_layout(KS_train)
        p.add_layout(Gini_train)
        p.add_layout(KS_test)
        p.add_layout(Gini_test)
        return p

    def test_on_new_data(self, X, y, models):
        for i in range(len(X)):
            for column in X[i].select_dtypes(["object"]).columns:
                X[i][column] = np.where(X[i][column].isin(self.modelBase[column]), X[i][column], "small_samples_combined")

            X_vec = self.dvec.transform(X[i].transpose().to_dict().values())
            X[i] = pd.DataFrame(X_vec,columns=self.dvec.get_feature_names(), index=X[i].index)
            i += 1

            for estimator in models:
                    self.Gini_table[estimator+"_test"+str(i)], self.KS[estimator+"_test"+str(i)], self.Gini[estimator+"_test"+str(i)] = self.performance(estimator=self.estimators[estimator], X=X[i-1], y=y[i-1], test_set=True, train_bins=self.train_bins[self.estimators[estimator]])
                    plot_data_new = self.Gini_table[estimator+"_test"+str(i)][self.Gini_table[estimator+"_test"+str(i)].Probability_of_1 != "All"][["Cumulative_good_%", "Cumulative_bad_%"]]
                    plot_data_new = pd.DataFrame(data={"Cumulative_good_%": 0, "Cumulative_bad_%": 0}, index=[100]).append(plot_data_new)
                    self.plots[estimator].line(x=plot_data_new["Cumulative_good_%"], y=plot_data_new["Cumulative_bad_%"], legend = "Delinquent distribution - test"+str(i), line_width=2, line_color="black", line_alpha=0.5)
                    ks = self.KS[estimator+"_test"+str(i)]
                    gini = self.Gini[estimator+"_test"+str(i)]
                    KS_new = Label(x=70, y=-20*(i+2), x_units='screen', y_units='screen',
                                     text="test"+str(i)+f" KS: {ks:{4}.{4}}%", render_mode='css',border_line_alpha=0.0,
                                     background_fill_color='white', background_fill_alpha=1.0)
                    Gini_new = Label(x=250, y=-20*(i+2), x_units='screen', y_units='screen',
                                       text="test"+str(i)+f" Gini: {gini:{4}.{4}}%", render_mode='css',border_line_alpha=0.0,
                                       background_fill_color='white', background_fill_alpha=1.0)
                    self.plots[estimator].add_layout(KS_new)
                    self.plots[estimator].add_layout(Gini_new)

                    show(self.plots[estimator])







