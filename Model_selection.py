import pandas as pd
import numpy as np
from sklearn.feature_extraction import DictVectorizer
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, confusion_matrix, precision_score, f1_score
from sklearn.model_selection import learning_curve
from sklearn.ensemble import GradientBoostingClassifier, RandomForestClassifier
from sklearn.svm import SVC
from sklearn.naive_bayes import GaussianNB
from sklearn.model_selection import RandomizedSearchCV, GridSearchCV
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import make_pipeline
from sklearn.ensemble import VotingClassifier

import bokeh
from bokeh.plotting import figure, show, output_notebook, gridplot
from bokeh.models import Range1d, Legend, Title, Label


__authors__ = "Kanishk Dogar", "Arya Poddar"
__version__ = "0.0.2"

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
                n_jobs=-1, learning_curve_train_sizes=[0.05, 0.1, 0.2, 0.4, 0.75, 1], fast=True, CV=5, verbose=1, automated=False):
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

        if self.univariate_test:
            self.univariate()
        self.train_test_transform()

        if automated:
            self.fit_gbm()
            self.fit_Random_forest()
            self.fit_SVM()
            self.fit_Naive_Bayes()
            output_notebook()
            self.evaluate_GBM()
            self.evaluate_Random_forest()
            self.evaluate_SVM()
            self.evaluate_Naive_Bayes()

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

    def fit_gbm(self):
        gbc = GradientBoostingClassifier()
        train_sizes, train_scores, test_scores = learning_curve(gbc, self.X_train, self.y_train, cv=5, n_jobs=self.n_jobs, train_sizes=self.train_sizes)
        curve = pd.concat([pd.DataFrame(train_sizes), pd.DataFrame(np.mean(train_scores, axis=1)), 
                   pd.DataFrame(np.mean(test_scores, axis=1)), pd.DataFrame(np.std(train_scores, axis=1)), 
                   pd.DataFrame(np.std(test_scores, axis=1))], axis=1)
        curve.columns = ["training_sample_size", "mean_train_accuracy", "mean_test_accuracy", "std_train_accuracy", "std_test_accuracy"]
        self.gbm_curve = curve
        
        if self.fast:
            n_gbm = np.min([self.X_train.shape[0], 50000])
        else:
            n_gbm = self.X_train.shape[0]

        X_train_cv = self.X_train.sample(n=np.int(np.rint(n_gbm)))
        y_train_cv = self.y_train[self.y_train.index.isin(X_train_cv.index)]
        y_train_cv = y_train_cv.reindex(X_train_cv.index)

        param_grid_gbm = {"learning_rate":[0.001, 0.01, 0.1, 1], "n_estimators":[10, 100, 500], "min_weight_fraction_leaf":[0.06, 0.1], "max_depth":[3, 10, 20, 40, 80]}

        gbm_clf = RandomizedSearchCV(gbc, param_grid_gbm, verbose=self.verbose, cv=self.CV, n_jobs=-1, n_iter=100, return_train_score=False)
        gbm_clf.fit(X_train_cv, y_train_cv)
        print("GBM best CV score:", gbm_clf.best_score_)
        gbc.set_params(learning_rate=gbm_clf.best_params_["learning_rate"], n_estimators=gbm_clf.best_params_["n_estimators"], max_depth=gbm_clf.best_params_["max_depth"], min_weight_fraction_leaf=gbm_clf.best_params_["min_weight_fraction_leaf"])

        gbc.fit(self.X_train, self.y_train)
        trainPredGBM = gbc.predict(self.X_train)
        testPredGBM = gbc.predict(self.X_test)
        train_accuracy = accuracy_score(self.y_train, trainPredGBM)*100
        print(f"GBM Training accuracy:{train_accuracy:{4}.{4}}%")
        test_accuracy = accuracy_score(self.y_test, testPredGBM)*100
        print(f"GBM Test accuracy:{test_accuracy:{4}.{4}}%")

        train_precision = precision_score(self.y_train, trainPredGBM)*100
        print(f"GBM Training precision:{train_precision:{4}.{4}}%")
        test_precision = precision_score(self.y_test, testPredGBM)*100
        print(f"GBM Test precision:{test_precision:{4}.{4}}%")

        self.gbc = gbc
        self.gbc_best_model = gbm_clf
        self.trainPredGBM, self.testPredGBM = trainPredGBM, testPredGBM
        
        return self.gbc

    def evaluate_GBM(self):
        self.GBM_train_performance, GBM_train_KS, GBM_train_Gini, self.GBM_test_performance, GBM_test_KS, GBM_test_Gini = self.performance_train_test(estimator=self.gbc, X=[self.X_train, self.X_test], y=[self.y_train, self.y_test])
        self.plot_GBM = self.plot_performance([self.GBM_train_performance, self.GBM_test_performance], [GBM_train_KS, GBM_test_KS], [GBM_train_Gini, GBM_test_Gini], name="Gradient Boosting")
        return(show(self.plot_GBM))

    def fit_Random_forest(self):
        rfc = RandomForestClassifier()
        train_sizes, train_scores, test_scores = learning_curve(rfc, self.X_train, self.y_train, cv=5, n_jobs=self.n_jobs, train_sizes=self.train_sizes)
        curve = pd.concat([pd.DataFrame(train_sizes), pd.DataFrame(np.mean(train_scores, axis=1)), 
                   pd.DataFrame(np.mean(test_scores, axis=1)), pd.DataFrame(np.std(train_scores, axis=1)), 
                   pd.DataFrame(np.std(test_scores, axis=1))], axis=1)
        curve.columns = ["training_sample_size", "mean_train_accuracy", "mean_test_accuracy", "std_train_accuracy", "std_test_accuracy"]
        self.RF_curve = curve
        
        if self.fast:
            n_rfc = np.min([self.X_train.shape[0], 50000])
        else:
            n_rfc = self.X_train.shape[0]

        X_train_cv = self.X_train.sample(n=np.int(np.rint(n_rfc)))
        y_train_cv = self.y_train[self.y_train.index.isin(X_train_cv.index)]
        y_train_cv = y_train_cv.reindex(X_train_cv.index)

        param_grid_rf = {"n_estimators":[10, 100, 200, 500], "min_weight_fraction_leaf":[0.06, 0.1], "max_depth":[3, 10, 20, 40, 80], "max_features":["auto", "sqrt", "log2", None]}

        rfc = RandomForestClassifier(n_jobs=self.n_jobs)
        rfc_clf = RandomizedSearchCV(rfc, param_grid_rf, verbose=self.verbose, cv=self.CV, n_jobs=self.n_jobs, n_iter=100, return_train_score=False)
        rfc_clf.fit(X_train_cv, y_train_cv)
        print("Random Forest best CV score:", rfc_clf.best_score_)
        rfc.set_params(n_estimators=rfc_clf.best_params_["n_estimators"], max_depth=rfc_clf.best_params_["max_depth"], min_weight_fraction_leaf=rfc_clf.best_params_["min_weight_fraction_leaf"], max_features=rfc_clf.best_params_["max_features"])

        rfc.fit(self.X_train, self.y_train)
        trainPredRF = rfc.predict(self.X_train)
        testPredRF = rfc.predict(self.X_test)
        train_accuracy = accuracy_score(self.y_train, trainPredRF)*100
        print(f"Random Forest Training accuracy:{train_accuracy:{4}.{4}}%")
        test_accuracy = accuracy_score(self.y_test, testPredRF)*100
        print(f"Random Forest Test accuracy:{test_accuracy:{4}.{4}}%")

        train_precision = precision_score(self.y_train, trainPredRF)*100
        print(f"Random Forest Training precision:{train_precision:{4}.{4}}%")
        test_precision = precision_score(self.y_test, testPredRF)*100
        print(f"Random Forest Test precision:{test_precision:{4}.{4}}%")

        self.rfc = rfc
        self.rfc_best_model = rfc_clf
        self.trainPredRF, self.testPredRF = trainPredRF, testPredRF
        return self.rfc

    def evaluate_Random_forest(self):
        self.RF_train_performance, RF_train_KS, RF_train_Gini, self.RF_test_performance, RF_test_KS, RF_test_Gini = self.performance_train_test(self.rfc, [self.X_train, self.X_test], [self.y_train, self.y_test])
        self.plot_RF = self.plot_performance([self.RF_train_performance, self.RF_test_performance], [RF_train_KS, RF_test_KS], [RF_train_Gini, RF_test_Gini], name="Random Forest")
        return show(self.plot_RF)

    def fit_SVM(self):
        svc = SVC(probability=True)
        train_sizes, train_scores, test_scores = learning_curve(svc, self.X_train, self.y_train, cv=5, n_jobs=self.n_jobs, train_sizes=self.train_sizes)
        curve = pd.concat([pd.DataFrame(train_sizes), pd.DataFrame(np.mean(train_scores, axis=1)), 
                   pd.DataFrame(np.mean(test_scores, axis=1)), pd.DataFrame(np.std(train_scores, axis=1)), 
                   pd.DataFrame(np.std(test_scores, axis=1))], axis=1)
        curve.columns = ["training_sample_size", "mean_train_accuracy", "mean_test_accuracy", "std_train_accuracy", "std_test_accuracy"]
        self.svm_curve = curve
        
        if self.fast:
            n_svm = np.min([self.X_train.shape[0], 50000])
        else:
            n_svm = self.X_train.shape[0]

        X_train_cv = self.X_train.sample(n=np.int(np.rint(n_svm)))
        y_train_cv = self.y_train[self.y_train.index.isin(X_train_cv.index)]
        y_train_cv = y_train_cv.reindex(X_train_cv.index)

        pipe_svc = make_pipeline(StandardScaler(), SVC(random_state=1, probability=True))
        param_svm1 = {"svc__kernel": ["rbf", "sigmoid", "linear", "poly"]}

        svc_clf1 = GridSearchCV(pipe_svc, param_svm1, verbose=self.verbose, cv=self.CV, n_jobs=self.n_jobs, return_train_score=False)
        svc_clf1.fit(X_train_cv, y_train_cv)

        pipe_svc = make_pipeline(StandardScaler(), SVC(random_state=1, kernel=svc_clf1.best_params_["svc__kernel"], probability=True))

        param_range = [0.001, 0.01, 0.1, 1.0, 10.0, 100]

        if svc_clf1.best_params_["svc__kernel"] == "linear":
            param_svm2 = {"svc__C": param_range}
        else:
            param_svm2 = {"svc__C": param_range, "svc__gamma": param_range}

        svc_clf2 = GridSearchCV(pipe_svc, param_svm2, verbose=self.verbose, cv=self.CV, n_jobs=self.n_jobs, return_train_score=False)
        svc_clf2.fit(X_train_cv, y_train_cv)

        print("SVM best CV score:", svc_clf2.best_score_)

        if svc_clf1.best_params_["svc__kernel"] == "linear":
            pipe_svc = make_pipeline(StandardScaler(), SVC(random_state=1, kernel=svc_clf1.best_params_["svc__kernel"], 
                                                           C=svc_clf2.best_params_["svc__C"], probability=True))
        else:
            pipe_svc = make_pipeline(StandardScaler(), SVC(random_state=1, kernel=svc_clf1.best_params_["svc__kernel"], 
                                                           C=svc_clf2.best_params_["svc__C"], gamma=svc_clf2.best_params_["svc__gamma"], probability=True))

        pipe_svc.fit(self.X_train, self.y_train)
        trainPredSVM = pipe_svc.predict(self.X_train)
        testPredSVM = pipe_svc.predict(self.X_test)
        train_accuracy = accuracy_score(self.y_train, trainPredSVM)*100
        print(f"SVM Training accuracy:{train_accuracy:{4}.{4}}%")
        test_accuracy = accuracy_score(self.y_test, testPredSVM)*100
        print(f"SVM Test accuracy:{test_accuracy:{4}.{4}}%")

        train_precision = precision_score(self.y_train, trainPredSVM)*100
        print(f"SVM Training precision:{train_precision:{4}.{4}}%")
        test_precision = precision_score(self.y_test, testPredSVM)*100
        print(f"SVM Test precision:{test_precision:{4}.{4}}%")

        self.pipe_svc = pipe_svc
        self.svm_best_model = svc_clf2
        self.trainPredSVM, self.testPredSVM = trainPredSVM, testPredSVM

        return self.pipe_svc

    def evaluate_SVM(self):
        self.SVM_train_performance, SVM_train_KS, SVM_train_Gini, self.SVM_test_performance, SVM_test_KS, SVM_test_Gini = self.performance_train_test(self.pipe_svc, [self.X_train, self.X_test], [self.y_train, self.y_test])
        self.plot_SVM = self.plot_performance([self.SVM_train_performance, self.SVM_test_performance], [SVM_train_KS, SVM_test_KS], [SVM_train_Gini, SVM_test_Gini], name="Support Vector Machine")
        return show(self.plot_SVM)

    def fit_Naive_Bayes(self):
        nb = GaussianNB()
        nb.fit(self.X_train, self.y_train)
        trainPredNB = nb.predict(self.X_train)
        testPredNB = nb.predict(self.X_test)

        train_accuracy = accuracy_score(self.y_train, trainPredNB)*100
        print(f"Naive Bayes training accuracy:{train_accuracy:{4}.{4}}%")
        test_accuracy = accuracy_score(self.y_test, testPredNB)*100
        print(f"Naive Bayes Test accuracy:{test_accuracy:{4}.{4}}%")

        train_precision = precision_score(self.y_train, trainPredNB)*100
        print(f"Naive Bayes Training precision:{train_precision:{4}.{4}}%")
        test_precision = precision_score(self.y_test, testPredNB)*100
        print(f"Naive Bayes Test precision:{test_precision:{4}.{4}}%")

        self.nb = nb
        self.trainPredNB, self.testPredNB = trainPredNB, testPredNB    
        return self.nb

    def evaluate_Naive_Bayes(self):
        self.Naive_Bayes_train_performance, Naive_Bayes_train_KS, Naive_Bayes_train_Gini, self.Naive_Bayes_test_performance, Naive_Bayes_test_KS, Naive_Bayes_test_Gini = self.performance_train_test(self.nb, [self.X_train, self.X_test], [self.y_train, self.y_test])
        self.plot_Naive_Bayes = self.plot_performance([self.Naive_Bayes_train_performance, self.Naive_Bayes_test_performance], [Naive_Bayes_train_KS, Naive_Bayes_test_KS], [Naive_Bayes_train_Gini, Naive_Bayes_test_Gini], name="Naive Bayes")
        return show(self.plot_Naive_Bayes)

    def performance(self, estimator, X, y, test_set=False, train_bins=0):
        performance = pd.concat([pd.DataFrame(estimator.predict_proba(X), index=X.index), pd.DataFrame(y)], axis=1)
        if test_set:
            performance["Probability_of_1"] = pd.cut(performance.loc[:, 1], bins=train_bins)
        else:
            seg, train_bins = pd.qcut(performance.loc[:, 1].round(12), 10, retbins=True, duplicates="drop")
            performance["Probability_of_1"] = pd.cut(performance.loc[:, 1], bins=train_bins)        

        performance = performance.pivot_table(index="Probability_of_1", values=self.target, aggfunc=[np.sum, len], margins=True).reset_index()
        performance.columns = ["Probability_of_1", "Bad", "Total"]

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
        test_performance, test_KS, test_Gini = self.performance(estimator=estimator, X=X[1], y=y[1], test_set=True, train_bins=bins)
        return train_performance, train_KS, train_Gini, test_performance, test_KS, test_Gini

    def plot_performance(self, performance, ks, gini, name=""):
        plot_data_train = performance[0][performance[0].Probability_of_1 != "All"][["Cumulative_good_%", "Cumulative_bad_%"]]
        plot_data_train = pd.DataFrame(data={"Cumulative_good_%": 0, "Cumulative_bad_%": 0}, index=[100]).append(plot_data_train)
        plot_data_test = performance[1][performance[1].Probability_of_1 != "All"][["Cumulative_good_%", "Cumulative_bad_%"]]
        plot_data_test = pd.DataFrame(data={"Cumulative_good_%": 0, "Cumulative_bad_%": 0}, index=[100]).append(plot_data_test)

        p = figure(plot_width=600, plot_height=400, title=f"Gini curve for {name}", )
        p.line(x=plot_data_train["Cumulative_good_%"], y=plot_data_train["Cumulative_bad_%"], legend = "Delinquent distribution - training", line_width=2, line_color="#053C6D", line_alpha=0.5)
        p.line(x=plot_data_test["Cumulative_good_%"], y=plot_data_test["Cumulative_bad_%"], legend = "Delinquent distribution - test", line_width=2, line_color="#97291E", line_alpha=0.5)
        p.line(x=plot_data_train["Cumulative_good_%"], y=plot_data_train["Cumulative_good_%"], line_color="#E46713", legend = "Random line", line_width=2)

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

