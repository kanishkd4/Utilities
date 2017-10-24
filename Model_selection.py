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
from bokeh.plotting import figure, show, output_notebook
from bokeh.models import Range1d, Legend


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
    def __init__(self, modelBase, target="target", small_sample_floor=0.01, combine_small_samples=True, test_size=0.2, random_state=12345, univariate_test=True,
                n_jobs=-1, learning_curve_train_sizes=[0.05, 0.1, 0.2, 0.4, 0.75, 1], fast=True, CV=5, verbose=1):
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

    def fit_SVM(self):
        svc = SVC()
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

        pipe_svc = make_pipeline(StandardScaler(), SVC(random_state=1))
        param_svm1 = {"svc__kernel": ["rbf", "sigmoid", "linear", "poly"]}

        svc_clf1 = GridSearchCV(pipe_svc, param_svm1, verbose=self.verbose, cv=self.CV, n_jobs=self.n_jobs, return_train_score=False)
        svc_clf1.fit(X_train_cv, y_train_cv)

        pipe_svc = make_pipeline(StandardScaler(), SVC(random_state=1, kernel=svc_clf1.best_params_["svc__kernel"]))

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
                                                           C=svc_clf2.best_params_["svc__C"]))
        else:
            pipe_svc = make_pipeline(StandardScaler(), SVC(random_state=1, kernel=svc_clf1.best_params_["svc__kernel"], 
                                                           C=svc_clf2.best_params_["svc__C"], gamma=svc_clf2.best_params_["svc__gamma"]))

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

    def fit_Naive_Bayes(self):
        nb.fit(self.X_train, self.y_train)
        trainPredNB = nb.predict(self.X_train)
        testPredNB = nb.predict(self.X_test)

        train_accuracy = accuracy_score(self.y_train, trainPredNB)*100
        print(f"Naive Bayes accuracy:{train_accuracy:{4}.{4}}%")
        test_accuracy = accuracy_score(self.y_test, testPredNB)*100
        print(f"Naive Bayes Test accuracy:{test_accuracy:{4}.{4}}%")

        train_precision = precision_score(self.y_train, trainPredNB)*100
        print(f"Naive Bayes Training precision:{train_precision:{4}.{4}}%")
        test_precision = precision_score(self.y_test, testPredNB)*100
        print(f"Naive Bayes Test precision:{test_precision:{4}.{4}}%")

        self.nb = nb
        self.trainPredNB, self.testPredNB = trainPredNB, testPredNB
        return self.nb








