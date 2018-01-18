import pandas as pd
import numpy as np
from scipy.stats import ttest_ind   
from sklearn.feature_extraction import DictVectorizer
from sklearn.preprocessing import LabelEncoder
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
from sklearn.externals import joblib

import bokeh
from bokeh.plotting import figure, show, output_notebook, gridplot
from bokeh.models import Range1d, Legend, Title, Label


__author__ = "Kanishk Dogar"
__version__ = "0.0.8"

"""
Tested for the following versions
pandas version: 0.22.0
numpy version: 1.13.3
scikit-learn version: 0.19.1
scipy version: 1.0.0
bokeh version: 0.12.13
"""

class model_train:
    """
    Create and summarise different scikit learn models
    MS stores data and applies multiple models to a training and test set based on models and hyperparameters selected.

    Parameters
    ----------
    target: string
        name of the target column
    modelBase: pandas dataframe
        the name of the dataframe in which all the data is stored
    categorical_columns: list of column names
        Select the columns that have to be treated as categories
    models: list
        list of models to fit- ["gbm", "random_forest", "svm", "naive_bayes", "logistic_regression"]
    export_path: string
        path to export all model objects and performance tables
    scoring: string
        scoring parameter for hyperparameter optimization
    combine_small_samples: bool
        if true, small categorical samples in every variables are combined to create one category. The minimum percentage is given by small_samples_floor
    small_sample_floor: numeric
        small samples in categorical variables are combined to form one category. e.g. if value is 0.01, all categorical variables
    random_state: integer
        select the random state
    univariate_test: bool
        if true, create a dataframe var_start_list within the class to describe the data
    n_jobs: integer
        number of processors to use for the models
    learning_curve: bool
        use a learning curve to determine the training data size for hyperparameter optimization
    learning_curve_train_sizes: list of floats
        the proportions of the training data to use to test the data size for hyperparameter optimization
    CV: integer
        value of k for k fold cross vaidation
    automated: boolean
        run the automated code on the data.
    classes: string
        the parameter "average in scikit learn's f1_score" [None, 'binary' (default), 'micro', 'macro', 'samples', 'weighted']
    """
    def __init__(self, modelBase, target="target", categorical_columns=[], export_path="", models=["gbm", "random_forest", "svm", "logistic_regression"],
        scoring="roc_auc", combine_small_samples=True, small_sample_floor=0.01, test_size=0.2, random_state=12345, 
        univariate_test=True, n_jobs=2, learning_curve=False, learning_curve_train_sizes=[0.05, 0.1, 0.2, 0.4, 0.75, 1], 
        CV=5, verbose=1, automated=False, classes="binary", num_na = 0, cat_na="."):
        self.target = target
        self.models = models
        self.categorical_columns = categorical_columns
        self.small_sample_floor = small_sample_floor
        self.combine_small_samples = combine_small_samples
        self.test_size = test_size
        self.random_state = random_state
        self.univariate_test = univariate_test
        self.n_jobs = n_jobs
        self.train_sizes = learning_curve_train_sizes
        self.CV = CV
        self.verbose = verbose
        self.scoring = scoring
        self.learning_curve = learning_curve
        self.export_path = export_path
        self.classes = classes
        self.num_na = num_na
        self.cat_na = cat_na

        if modelBase[target].dtype != "int64":
            raise TypeError(f"target is not an integer - target type = {modelBase[target].dtype}. Use sklearn's label encoder to encode target")
        

        self.modelBase = data_process(data=modelBase, categorical_columns=categorical_columns, cat_na=cat_na, num_na=num_na)

        if self.univariate_test:
            self.univariate()
        self.train_test_transform()

        if automated:
            self.fit_models()
            self.apply_models()
            self.evaluate_models()


    def univariate(self):
        """
        Summarize the model base data of the starting list of all variables
        The summary data frame is stored in self.var_start_list
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
        """
        Split the data into a training and test set; combine small samples and use scikit-learn's DictVectorizer to one hot encode the features
        """

        if self.univariate_test:
            self.modelBase.drop(list(self.var_start_list[(self.var_start_list["var_vals"] < 2)].index), axis=1, inplace=True)

        self.X, self.y = self.modelBase.loc[:, self.modelBase.columns != self.target], self.modelBase.loc[:, self.target]
        X_train, X_test, y_train, y_test = train_test_split(self.X, self.y, test_size=self.test_size, random_state=self.random_state)

        if self.combine_small_samples:
            for column in X_train.select_dtypes(["object"]).columns:
                cnt = pd.value_counts(X_train[column], normalize=True)
                X_train.loc[:, column] = np.where(X_train[column].isin(cnt[cnt < self.small_sample_floor].index), "small_samples_combined", X_train[column])

            for column in X_test.select_dtypes(["object"]).columns:
                X_test.loc[:, column] = np.where(X_test[column].isin(X_train[column]), X_test[column], "small_samples_combined")

        
        self.dvec = DictVectorizer(sparse=False)
        X_dict = self.dvec.fit_transform(X_train.transpose().to_dict().values())
        X_train = pd.DataFrame(X_dict, columns=self.dvec.get_feature_names(), index=X_train.index)

        X_test_dict = self.dvec.transform(X_test.transpose().to_dict().values())
        X_test = pd.DataFrame(X_test_dict, columns=self.dvec.get_feature_names(), index=X_test.index)

        self.X_train, self.X_test, self.y_train, self.y_test = X_train, X_test, y_train, y_test


    def fit_models(self, gbm_params={"learning_rate":[0.001, 0.01, 0.1, 1], "n_estimators":[10, 100, 500], "min_weight_fraction_leaf":[0.06, 0.1], "max_depth":[3, 10, 20, 40, 80]},
        random_forest_params={"n_estimators":[10, 100, 200, 500], "min_weight_fraction_leaf":[0.06, 0.1], "max_depth":[3, 10, 20, 40, 80], "max_features":["auto", "sqrt", "log2", None]},
        svm_kernels={"svc__kernel": ["rbf", "sigmoid", "linear", "poly"]}, logistic_params={"logisticregression__C":[0.001, 0.01, 0.1, 1, 10]}, n_iter=100):
        """
        Fit the selected models using a learning curve to estimate the optimum training sample size and cross validation for hyperparameter optimization
        Parameters
        ----------
        gbm_params: dictionary
            The parameters to test for a Gradient Boosting Classifier
        random_forest_params: dictionary
            The parameters to test for a Random Forest Classifier
        n_iter: int
            number of random samples to check for the gbm and random forest
        svm_kernels: dictionary
            The kernels to test for the svm - using a grid search instead of a random search
        logistic_params: dictionary
            Logistic regression costs to test - using a grid search instead of a random search
        """
        self.gbm_params, self.random_forest_params, self.svm_kernels, self.logistic_params = gbm_params, random_forest_params, svm_kernels, logistic_params
        self.n_iter = n_iter

        self.init_models()
        self.fit_learning_curve()
        self.parameter_search()


    def init_models(self):
        estimators = {}
        for estimator in self.models:
            if estimator == "gbm":
                estimators[estimator] = GradientBoostingClassifier()
            elif estimator == "random_forest":
                estimators[estimator] = RandomForestClassifier()
            elif estimator == "svm":
                estimators[estimator] = SVC(probability=True)
            elif estimator == "naive_bayes":
                estimators[estimator] = GaussianNB()
            elif estimator == "logistic_regression":
                estimators[estimator] = LogisticRegression()
            else:
                print("method not applicable for model")

        self.estimators = estimators

    def fit_learning_curve(self):
        self.curve = {}
        self.learning_sample = {}
        for estimator in self.models:
            if estimator in ["gbm", "random_forest", "svm", "logistic_regression"]:
                if self.learning_curve:
                    train_sizes, train_scores, test_scores = learning_curve(self.estimators[estimator], self.X_train, self.y_train, cv=5, n_jobs=self.n_jobs, train_sizes=self.train_sizes)
                    curve = pd.concat([pd.DataFrame(train_sizes), pd.DataFrame(np.mean(train_scores, axis=1)), 
                               pd.DataFrame(np.mean(test_scores, axis=1)), pd.DataFrame(np.std(train_scores, axis=1)), 
                               pd.DataFrame(np.std(test_scores, axis=1))], axis=1)
                    curve.columns = ["training_sample_size", "mean_train_accuracy", "mean_test_accuracy", "std_train_accuracy", "std_test_accuracy"]
                    self.curve[estimator] = curve
                    temp = self.curve[estimator]
                    i = temp.shape[0]
                    while i > 1:
                        a = np.random.normal(temp.loc[i-1, "mean_test_accuracy"], temp.loc[i-1, "std_test_accuracy"], 1000)
                        b = np.random.normal(temp.loc[i-2, "mean_test_accuracy"], temp.loc[i-2, "std_test_accuracy"], 1000)
                        check = ttest_ind(a, b).pvalue < 0.05 # if p value is lower than 0.05, the averages are not equal
                        if check == True:
                            break
                        i -= 1

                    self.learning_sample[estimator] = temp.loc[i-1, "training_sample_size"]                 
  

    def parameter_search(self):
        
        if self.learning_curve:
            n_cv = np.int(np.floor(max(self.learning_sample.values())*1.25))
            cv = pd.concat([self.X_train, self.y_train], axis=1)
            cv = cv.sample(n=n_cv)
            X_train_cv, y_train_cv = cv.loc[:, cv.columns != self.target], cv[self.target]
        else:
            X_train_cv, y_train_cv = self.X_train, self.y_train  
        

        param_grid = {}
        clf = {}

        for estimator in self.models:
            if estimator == "gbm":
                param_grid[estimator] = self.gbm_params
            elif estimator == "random_forest":
                param_grid[estimator] = self.random_forest_params
            elif estimator == "svm":
                param_grid[estimator+"1"] = self.svm_kernels
            elif estimator == "logistic_regression":
                param_grid[estimator+"1"] = self.logistic_params

        for estimator in self.models:
            if estimator in ["gbm", "random_forest"]:
                clf[estimator] = RandomizedSearchCV(self.estimators[estimator], param_grid[estimator], verbose=self.verbose, cv=self.CV, n_jobs=self.n_jobs, n_iter=self.n_iter, return_train_score=False, scoring=self.scoring)
                clf[estimator].fit(X_train_cv, y_train_cv)
                print(estimator+" best CV score:", clf[estimator].best_score_)
                if estimator == "gbm":
                    self.estimators[estimator].set_params(learning_rate=clf[estimator].best_params_["learning_rate"], n_estimators=clf[estimator].best_params_["n_estimators"], max_depth=clf[estimator].best_params_["max_depth"], min_weight_fraction_leaf=clf[estimator].best_params_["min_weight_fraction_leaf"])
                elif estimator == "random_forest":
                    self.estimators[estimator].set_params(n_estimators=clf[estimator].best_params_["n_estimators"], max_depth=clf[estimator].best_params_["max_depth"], min_weight_fraction_leaf=clf[estimator].best_params_["min_weight_fraction_leaf"], max_features=clf[estimator].best_params_["max_features"])

                self.estimators[estimator].fit(self.X_train, self.y_train)
            
            elif estimator in ["svm", "logistic_regression"]:
                self.estimators[estimator] = make_pipeline(StandardScaler(), self.estimators[estimator])
                clf[estimator+"1"] = GridSearchCV(self.estimators[estimator], param_grid[estimator+"1"], verbose=self.verbose, cv=self.CV, n_jobs=self.n_jobs, return_train_score=False, scoring=self.scoring)
                clf[estimator+"1"].fit(X_train_cv, y_train_cv)

                if estimator == "logistic_regression":
                    self.estimators[estimator].set_params(logisticregression__C=clf[estimator+"1"].best_params_["logisticregression__C"])
                    self.estimators[estimator].fit(self.X_train, self.y_train)
                    print(estimator+" best CV score:", clf[estimator+"1"].best_score_)

                elif estimator == "svm":
                    self.estimators[estimator] = make_pipeline(StandardScaler(), SVC(random_state=1, kernel=clf[estimator+"1"].best_params_["svc__kernel"], probability=True))
                    if clf[estimator+"1"].best_params_["svc__kernel"] == "linear":
                        param_grid[estimator+"2"] = {"svc__C": [0.001, 0.01, 0.1, 1.0, 10.0, 100]}
                    else:
                        param_grid[estimator+"2"] = {"svc__C": [0.001, 0.01, 0.1, 1.0, 10.0, 100], "svc__gamma": [0.001, 0.01, 0.1, 1.0, 10.0, 100]}
                    
                    clf[estimator+"2"] = GridSearchCV(self.estimators[estimator], param_grid[estimator+"2"], verbose=self.verbose, cv=self.CV, n_jobs=self.n_jobs, return_train_score=False, scoring=self.scoring)
                    clf[estimator+"2"].fit(X_train_cv, y_train_cv)

                    if clf[estimator+"1"].best_params_["svc__kernel"] == "linear":
                        self.estimators[estimator] = make_pipeline(StandardScaler(), SVC(random_state=1, kernel=clf[estimator+"1"].best_params_["svc__kernel"], 
                                                                       C=clf[estimator+"2"].best_params_["svc__C"], probability=True))
                    else:
                        self.estimators[estimator] = make_pipeline(StandardScaler(), SVC(random_state=1, kernel=clf[estimator+"1"].best_params_["svc__kernel"], 
                                                                       C=clf[estimator+"2"].best_params_["svc__C"], gamma=clf[estimator+"2"].best_params_["svc__gamma"], probability=True))

                    print(estimator+" best CV score:", clf[estimator+"2"].best_score_)
                    self.estimators[estimator].fit(self.X_train, self.y_train)

            elif estimator == "naive_bayes":
                self.estimators[estimator].fit(self.X_train, self.y_train)


            print(f"{estimator} fitting complete\n")

        self.clf = clf


    def apply_models(self, classes="macro"):
        """
        Apply the fitted models on the full training and validation set to get accuracy and f1_score. This can be used to select the model to implement in production
        """
        trainPred = {}
        testPred = {}

        if (self.y.drop_duplicates().shape[0] > 2) & (self.classes=="binary"):
            self.classes = classes
            print(f"changing f score average parameter from binary to {classes}")
        
        for estimator in self.models:
            trainPred[estimator] = self.estimators[estimator].predict(self.X_train)
            testPred[estimator] = self.estimators[estimator].predict(self.X_test)

            train_accuracy = accuracy_score(self.y_train, trainPred[estimator])*100
            print(f"{estimator} Training accuracy:{train_accuracy:{4}.{4}}%")
            test_accuracy = accuracy_score(self.y_test, testPred[estimator])*100
            print(f"{estimator} Test accuracy:{test_accuracy:{4}.{4}}%")

            train_f1_score = f1_score(self.y_train, trainPred[estimator], average=self.classes)*100
            print(f"{estimator} Training f1_score:{train_f1_score:{4}.{4}}%")
            test_f1_score = f1_score(self.y_test, testPred[estimator], average=self.classes)*100
            print(f"{estimator} Test f1_score:{test_f1_score:{4}.{4}}%\n")

            self.trainPred, self.testPred = trainPred, testPred

    def evaluate_models(self, event=1):
        """
        Generate performance characteristics and the lorenz curve to get more evaluations of the fitted model based on the predicted probabilties of the event class.
        The event by default is assumed to be 1.
        """
        self.event = event
        Gini_table = {}
        KS = {}
        Gini = {}
        plots = {}
        self.train_bins = {}

        for estimator in self.models:
            Gini_table[estimator+"_train"], KS[estimator+"_train"], Gini[estimator+"_train"], Gini_table[estimator+"_test"], KS[estimator+"_test"], Gini[estimator+"_test"] = self.performance_train_test(estimator=self.estimators[estimator], X=[self.X_train, self.X_test], y=[self.y_train, self.y_test])
            plots[estimator] = self.plot_performance([Gini_table[estimator+"_train"], Gini_table[estimator+"_test"]], [KS[estimator+"_train"], KS[estimator+"_test"]], [Gini[estimator+"_train"], Gini[estimator+"_test"]], name=estimator.upper())
        self.Gini_table, self.KS, self.Gini, self.plots = Gini_table, KS, Gini, plots

    def performance(self, estimator, X, y, test_set=False, train_bins=0):
        performance = pd.concat([pd.DataFrame(estimator.predict_proba(X), index=X.index), pd.DataFrame(y)], axis=1)
        performance.loc[:, "Probability_of_non_Event"] = 1 - performance.loc[:, self.event]
        performance.loc[:, self.target] = np.where(performance.loc[:, self.target] == self.event, 1, 0)
        performance.rename(columns={self.event: "Probability_of_Event"}, inplace=True)
        performance = performance.loc[:, ["Probability_of_Event", "Probability_of_non_Event", self.target]]

        if test_set:
            performance["Probability_of_Event"] = pd.cut(performance.loc[:, "Probability_of_Event"], bins=train_bins, include_lowest=True)
        else:
            seg, train_bins = pd.qcut(performance.loc[:, "Probability_of_Event"].round(12), 10, retbins=True, duplicates="drop")
            train_bins[0] = 0.0
            train_bins[train_bins.shape[0]-1] = 1.0
            performance["Probability_of_Event"] = pd.cut(performance.loc[:, "Probability_of_Event"], bins=train_bins, include_lowest=True)        

        performance = performance.pivot_table(index="Probability_of_Event", values=self.target, aggfunc=[np.sum, len], margins=True).reset_index()
        performance.columns = ["Probability_of_Event", "Event", "Total"]
        performance = performance[performance.Total.notnull()]

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

    def performance_train_test(self, estimator, X, y):
        train_performance, train_KS, train_Gini, bins = self.performance(estimator=estimator, X=X[0], y=y[0])
        self.train_bins[estimator] = bins
        test_performance, test_KS, test_Gini = self.performance(estimator=estimator, X=X[1], y=y[1], test_set=True, train_bins=self.train_bins[estimator])
        return train_performance, train_KS, train_Gini, test_performance, test_KS, test_Gini

    def plot_performance(self, performance, ks, gini, name=""):
        plot_data_train = performance[0][performance[0].Probability_of_Event != "All"][["Cumulative_Non_event_%", "Cumulative_Event_%"]]
        plot_data_train = pd.DataFrame(data={"Cumulative_Non_event_%": 0, "Cumulative_Event_%": 0}, index=[100]).append(plot_data_train)
        plot_data_test = performance[1][performance[1].Probability_of_Event != "All"][["Cumulative_Non_event_%", "Cumulative_Event_%"]]
        plot_data_test = pd.DataFrame(data={"Cumulative_Non_event_%": 0, "Cumulative_Event_%": 0}, index=[100]).append(plot_data_test)

        p = figure(plot_width=600, plot_height=400, title=f"Gini curve for {name}", )
        p.line(x=plot_data_train["Cumulative_Non_event_%"], y=plot_data_train["Cumulative_Event_%"], legend = "Event distribution - training", line_width=2, line_color="#053C6D", line_alpha=0.5)
        p.line(x=plot_data_test["Cumulative_Non_event_%"], y=plot_data_test["Cumulative_Event_%"], legend = "Event distribution - test", line_width=2, line_color="#97291E", line_alpha=0.5)
        p.line(x=plot_data_train["Cumulative_Non_event_%"], y=plot_data_train["Cumulative_Non_event_%"], line_color="#E46713", line_width=2)

        p.x_range = Range1d(0, 1)
        p.y_range = Range1d(0, 1)
        p.legend.location = "bottom_right"
        p.xaxis.axis_label = "Cumulative Non_event"
        p.yaxis.axis_label = "Cumulative Event"

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

    def select_retrain_model(self, final_model):
        """
        Select the best model (one or a list of models) and retrain on the full training dataset.
        Final_model can be a string with one model or a list of models
        """
        if isinstance(final_model, list):
            self.final_model = final_model
        else:
            self.final_model = [final_model]
        if self.combine_small_samples:
            for column in self.X.select_dtypes(["object"]).columns:
                cnt = pd.value_counts(self.X[column], normalize=True)
                self.X.loc[:, column] = np.where(self.X[column].isin(cnt[cnt < self.small_sample_floor].index), "small_samples_combined", self.X[column])

        self.dvec = DictVectorizer(sparse=False)
        X_dict = self.dvec.fit_transform(self.X.transpose().to_dict().values())
        self.X = pd.DataFrame(X_dict, columns=self.dvec.get_feature_names(), index=self.X.index)

        for estimator in self.final_model:
            self.estimators[estimator].fit(self.X, self.y)

    def test_on_holdout_data(self, X, y):
        """
        Test the selected models on holdout samples if any have been kept for a final evaluation.
        The selected models will also refit after evaluation as we test on these so that they can be saved to be used later
        Parameters
        ----------
        X: dataframe or a list of dataframes
            Enter as a list if multiple holdout sets have been kept or if the model has to be tested on multiple segments of the data.
        y: dataframe or a list of dataframes
            The target of the holdout sample. Enter as a list of dataframes if mulitple sets have to be used.
        """
        self.feature_levels = {}
        if isinstance(X, list) == False:
            X, y = [X], [y]
        X_levels = self.modelBase.loc[:, self.modelBase.columns != self.target]
        new_X, new_y = self.X.copy(), self.y.copy()
        for column in X_levels.select_dtypes(["object"]).columns:
            self.feature_levels[column] = X_levels.loc[:, column].unique()
        for i in range(len(X)):
            X[i] = data_process(data=X[i], categorical_columns=self.categorical_columns, cat_na=self.cat_na, num_na=self.num_na)
            for column in X[i].select_dtypes(["object"]).columns:
                X[i].loc[:, column] = np.where(X[i][column].isin(self.feature_levels[column]), X[i].loc[:, column], "small_samples_combined")

            

            X_vec = self.dvec.transform(X[i].transpose().to_dict().values())
            X[i] = pd.DataFrame(X_vec,columns=self.dvec.get_feature_names(), index=X[i].index)
            new_X, new_y = new_X.append(X[i], ignore_index=True), new_y.append(y[i], ignore_index=True)
            i += 1

            for estimator in self.final_model:
                    self.Gini_table[estimator+"_test"+str(i)], self.KS[estimator+"_test"+str(i)], self.Gini[estimator+"_test"+str(i)] = self.performance(estimator=self.estimators[estimator], X=X[i-1], y=y[i-1], test_set=True, train_bins=self.train_bins[self.estimators[estimator]])
                    plot_data_new = self.Gini_table[estimator+"_test"+str(i)][self.Gini_table[estimator+"_test"+str(i)].Probability_of_Event != "All"][["Cumulative_Non_event_%", "Cumulative_Event_%"]]
                    plot_data_new = pd.DataFrame(data={"Cumulative_Non_event_%": 0, "Cumulative_Event_%": 0}, index=[100]).append(plot_data_new)
                    self.plots[estimator].line(x=plot_data_new["Cumulative_Non_event_%"], y=plot_data_new["Cumulative_Event_%"], legend = "Event distribution - test"+str(i), line_width=2, line_color="black", line_alpha=0.5)
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

        for estimator in self.final_model:
            print("Re-fitting on the full data...")
            self.estimators[estimator].fit(new_X, new_y)


    def save_model_objects(self):
        """
        Save model objects to use later - save the dictvectorizer, feature_levels and the model object
        """
        for estimator in self.final_model:
            joblib.dump(self.estimators[estimator], self.export_path+f"{estimator}.pkl")
        joblib.dump(self.feature_levels, self.export_path+"feature_levels.pkl")
        joblib.dump(self.dvec, self.export_path+"DictVectorizer.pkl")
        joblib.dump(self.categorical_columns, self.export_path+"Categorical_columns.pkl")

def model_apply(X, path="", model="", DictVectorizer="DictVectorizer.pkl", 
    feature_levels="feature_levels.pkl", categorical_columns="Categorical_columns.pkl",
    predict_proba=False, cat_na=".", num_na=0):
    """
    Apply saved scikit learn model on new data.
    Parameters
    ----------
    X: dataframe
        New data for predictions
    path: string
        path of saved models objects
    model: string
        filename of saved model object
    DictVectorizer: string
        filename of the saved dictvectorizer object
    feature_levels: string
        filename of the dictionary to recategorize object features
    predict_proba: bool
        if true, the model will predict probabilites, else will predict classes
    """
    feature_levels = joblib.load(path+feature_levels)
    dvec = joblib.load(path+DictVectorizer)
    model = joblib.load(path+model)
    categorical_columns = joblib.load(path+categorical_columns)

    
    X = data_process(data=X, categorical_columns=categorical_columns, cat_na=cat_na, num_na=num_na)


    for column in X.select_dtypes(["object"]).columns:
        X.loc[:, column] = np.where(X[column].isin(feature_levels[column]), X.loc[:, column], "small_samples_combined")

    X_vec = dvec.transform(X.transpose().to_dict().values())
    X = pd.DataFrame(X_vec,columns=dvec.get_feature_names(), index=X.index)

    if predict_proba:
        predictions = model.predict_proba(X)
    else:
        predictions = model.predict(X)

    return predictions

def data_process(data, categorical_columns, cat_na, num_na):
    data = data.apply(lambda x: x.fillna(num_na) if x.dtype.kind in 'biufc' else x.fillna(cat_na))
    data.loc[:, categorical_columns] = data.loc[:, categorical_columns].astype("object")
    for column in categorical_columns:
        data.loc[:, column] = data.loc[:, column].apply(str) + "_"

    return data

