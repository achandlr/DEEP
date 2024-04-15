from sklearn.model_selection import GridSearchCV
from sklearn.base import clone
from src.models.ModelBaseClass import ModelBaseClass
import pandas as pd
import numpy as np
import time
# General utilities from sklearn
from sklearn.model_selection import GridSearchCV
from sklearn.base import clone
from sklearn.exceptions import NotFittedError
# Ensemble methods
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier, AdaBoostClassifier
from sklearn.ensemble import VotingClassifier
# Linear models
from sklearn.linear_model import LogisticRegression
# Decision Trees
from sklearn.tree import DecisionTreeClassifier
# Support Vector Machines
from sklearn.svm import SVC
# Naive Bayes classifiers
from sklearn.naive_bayes import MultinomialNB, BernoulliNB, GaussianNB
# Nearest Neighbors
from sklearn.neighbors import KNeighborsClassifier
# Discriminant Analysis
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
# Boosting frameworks outside of sklearn
from xgboost import XGBClassifier
from catboost import CatBoostClassifier

from sklearn.dummy import DummyClassifier
from src.utils.Logger import setup_logger
from sklearn.base import BaseEstimator, ClassifierMixin
from sklearn.model_selection import cross_val_score
from sklearn.utils.validation import check_is_fitted
from sklearn.metrics import balanced_accuracy_score
from lightgbm import LGBMClassifier


def warn(*args, **kwargs):
    pass
import warnings
warnings.warn = warn


class SKLearnModel(ModelBaseClass):
    def __init__(self, base_model, param_grid=None):
        """
        Initializes the SKLearnModel instance with a base model and an optional parameter grid for grid search.

        :param base_model: A scikit-learn model instance.
        :param param_grid: An optional dictionary defining the grid of parameters for grid search.
        """
        self.base_model = base_model
        self.param_grid = param_grid
        self.grid_search = None
        self.logger = setup_logger()

    def _train(self, X_train, Y_train):
        """
        Trains the SKLearnModel instance using the base model or performs grid search if a parameter grid is provided.

        :param X_train: Array-like of shape (n_samples, n_features) containing the training data.
        :param Y_train: Array-like of shape (n_samples,) containing the target labels.
        """
        start_time = time.time()  # Start timing

        model_name = self.base_model.__class__.__name__  # Get the class name of the base model

        if self.param_grid:
            print(f"Starting GridSearchCV for {model_name}...")
            self.logger.info(f"Starting GridSearchCV for {model_name}...")
            self.grid_search = GridSearchCV(clone(self.base_model), self.param_grid, cv=5)
            self.grid_search.fit(X_train, Y_train)
            grid_search_time = time.time() - start_time  # Time taken for grid search
            self.logger.debug(f"GridSearchCV for {model_name} completed. Time taken: {grid_search_time:.2f} seconds.")
        else:
            print(f"Starting training for {model_name}...")
            self.logger.info(f"Starting training for {model_name}...")
            self.base_model.fit(X_train, Y_train)
            base_model_time = time.time() - start_time  # Time taken for base model training
            self.logger.debug(f"Training for {model_name} completed. Time taken: {base_model_time:.2f} seconds.")

    def get_fitted_model(self):
        """
        Returns the fitted model, either the base model or the best estimator from grid search.
        """
        if self.param_grid and self.grid_search:
            return self.grid_search.best_estimator_
        else:
            return self.base_model

    def predict_proba(self, X):
        """
        Predict class probabilities for X.
        """
        model = self.get_fitted_model()  # Get the appropriate model
        try:
            return model.predict_proba(X)
        except NotFittedError as e:
            self.logger.error(f"Model not fitted: {e}")
            raise

    def is_fitted(self):
        """
        Check if the model or grid search model is fitted.

        :return: Boolean indicating if the model is fitted
        """
        try:
            if self.param_grid and self.grid_search:
                check_is_fitted(self.grid_search.best_estimator_)
            else:
                check_is_fitted(self.base_model)
            return True
        except NotFittedError:
            return False

    def predict(self, X):
        """
        Predict class labels for X using either the base model or the grid search model.

        :param X: Array-like of shape (n_samples, n_features)
        :return: Array of shape (n_samples,) with predicted class labels
        """
        return self.grid_search.predict(X) if self.grid_search else self.base_model.predict(X)

    def report_trained_parameters(self):
        """
        Report the trained parameters of the model.

        :return: Dictionary containing the best parameters found by grid search, if applicable.
        """
        return self.grid_search.best_params_ if self.grid_search else {}


class WeightedMajorityVotingClassifier(ModelBaseClass, BaseEstimator, ClassifierMixin):
    def __init__(self, classifiers=None):
        """
        Initializes the WeightedMajorityVotingClassifier instance with a list of classifiers.

        :param classifiers: List of scikit-learn classifier instances. Default is None.
        """
        if classifiers is None:
            classifiers = [LogisticRegression(), RandomForestClassifier()]
        self.classifiers = classifiers
        self.weights = None
    def _train(self, X_train, Y_train):
        """
        Trains the WeightedMajorityVotingClassifier instance by fitting each classifier and calculating the weights.

        :param X_train: Array-like of shape (n_samples, n_features) containing the training data.
        :param Y_train: Array-like of shape (n_samples,) containing the target labels.
        """
        self.weights = []
        for clf in self.classifiers:
            clf.fit(X_train, Y_train)
            # Cross-validation to determine the weight of each classifier
            weight = np.mean(cross_val_score(clf, X_train, Y_train, cv=5))
            self.weights.append(weight)

    def predict(self, X):
        """
        Predict class labels for X using the weighted majority voting scheme.

        :param X: Array-like of shape (n_samples, n_features)
        :return: Array of shape (n_samples,) with predicted class labels
        """
        # Aggregate predictions from all classifiers
        class_labels = list(set.union(*(set(clf.classes_) for clf in self.classifiers)))
        weighted_votes = np.zeros((X.shape[0], len(class_labels)))

        label_to_index = {label: index for index, label in enumerate(class_labels)}

        for clf, weight in zip(self.classifiers, self.weights):
            predictions = clf.predict(X)
            for i, p in enumerate(predictions):
                weighted_votes[i, label_to_index[p]] += weight
        return np.array(class_labels)[np.argmax(weighted_votes, axis=1)]

    def report_trained_parameters(self):
        """
        Report the trained parameters of the model.

        :return: Dictionary containing the names of the classifiers and their corresponding weights.
        """
        return {clf.__class__.__name__: w for clf, w in zip(self.classifiers, self.weights)}


class WeightedMajorityVotingClassifierBalancedAccuracyBased(ModelBaseClass, BaseEstimator, ClassifierMixin):
    def __init__(self):
        self.weights = None

    def _train(self, X_train, Y_train):
        """
        Trains the WeightedMajorityVotingClassifierBalancedAccuracyBased instance by calculating the weights based on
        the balanced accuracy score of each feature.

        :param X_train: Array-like of shape (n_samples, n_features) containing the training data.
        :param Y_train: Array-like of shape (n_samples,) containing the target labels.
        """
        self.weights = []
        n_features = X_train.shape[1]

        # Calculate the balanced accuracy score for each feature as a separate classifier
        for feature_index in range(n_features):
            feature_pred = X_train[:, feature_index]
            weight = balanced_accuracy_score(Y_train, feature_pred)
            self.weights.append(weight)

    def predict(self, X):
        """
        Predict class labels for X using the weighted majority voting scheme based on balanced accuracy scores.

        :param X: Array-like of shape (n_samples, n_features)
        :return: Array of shape (n_samples,) with predicted class labels
        """
        # Weighted voting
        weighted_votes = np.zeros(X.shape[0])

        for feature_index, weight in enumerate(self.weights):
            weighted_votes += X[:, feature_index] * weight

        # Decision based on the sum of weighted votes
        return (weighted_votes >= 0.5).astype(int)

    def report_trained_parameters(self):
        """
        Report the trained parameters of the model.

        :return: Dictionary containing the weights for each feature.
        """
        return {f'Feature_{i}': w for i, w in enumerate(self.weights)}
    
class RandomForestSKLearnModel(SKLearnModel):
    def __init__(self):
        param_grid = {
            'n_estimators': [50, 100, 150],
            'max_features': ['auto', 'sqrt'],
            'max_depth': [None, 20, 40],
            'min_samples_split': [2, 5],
            'min_samples_leaf': [1, 2],
            'bootstrap': [True, False]
        }
        super().__init__(RandomForestClassifier(), param_grid)
     
class GradientBoostingSKLearnModel(SKLearnModel):
    def __init__(self):
        param_grid = {
            'n_estimators': [100, 200],  # Reduced number of estimators
            'learning_rate': [0.1, 0.2],  # Simplified learning rates
            'subsample': [0.7, 1.0],  # Fewer subsampling options
            'max_depth': [3, 4],  # Lower max depth to reduce complexity
            'min_samples_split': [2, 5, 10],
            'min_samples_leaf': [1, 2, 4]
        }
        super().__init__(GradientBoostingClassifier(), param_grid)

class AdaBoostSKLearnModel(SKLearnModel):
    def __init__(self):
        param_grid = {
            'n_estimators': [50, 100, 200, 300],
            'learning_rate': [0.01, 0.1, 0.5, 1.0]
        }
        super().__init__(AdaBoostClassifier(), param_grid)



class DummySKLearnModel(SKLearnModel):
    def __init__(self):
        super().__init__(DummyClassifier(strategy='prior'), None)


class VotingSKLearnModel(SKLearnModel):
    def __init__(self, estimators):
        super().__init__(VotingClassifier(estimators=estimators, voting='hard'), None)

class LogisticRegressionSKLearnModel(SKLearnModel):
    def __init__(self):
        # param_grid = {
        #     'C': np.logspace(-4, 4, 20),
        #     'solver': ['newton-cg', 'lbfgs', 'liblinear', 'sag', 'saga'],
        #     'max_iter': [100, 200, 300, 400, 500]
        # }
        param_grid = {
            'C': np.logspace(-4, 4, 20),
            'solver': ['newton-cg', 'lbfgs', 'liblinear', 'sag', 'saga'],
            'max_iter': [1000, 2000, 3000]  # Increased max_iter
        }
        super().__init__(LogisticRegression(), param_grid)


           
class SVCSKLearnModel(SKLearnModel):
    def __init__(self):
        # Note: This param grid takes hours to train on our data. If SVC shows promise, then we can add back a larger grid search
        super().__init__(SVC(C=1, kernel='rbf', gamma='scale', probability = True), None)

class DecisionTreeSKLearnModel(SKLearnModel):
    def __init__(self):

        param_grid = {
            'criterion': ['gini', 'entropy'],
            'splitter': ['best', 'random'],
            'max_depth': [None, 1, 2, 5, 10, 20, 30],  # Simplified range
            'min_samples_split': [2, 4, 6, 8],         # Fewer options
            'min_samples_leaf': [1, 3, 5, 7],          # Fewer options
            'max_features': [None, 'sqrt', 'log2']     # Valid options
        }
        super().__init__(DecisionTreeClassifier(), param_grid)



class GaussianNBSKLearnModel(SKLearnModel):
    def __init__(self):
        param_grid = {
            'var_smoothing': np.logspace(0, -9, num=100)
        }
        super().__init__(GaussianNB(), param_grid)

class MultinomialNBSKLearnModel(SKLearnModel):
    def __init__(self):
        # Suppress specific warnings
        warnings.filterwarnings("ignore", category=FutureWarning, module="sklearn.naive_bayes")
        warnings.filterwarnings("ignore", category=UserWarning, module="sklearn.naive_bayes")

        # Start alpha range from a slightly higher value
        param_grid = {
            'alpha': np.linspace(1e-2, 1.0, num=100)
        }
        # Add force_alpha if it's available in your scikit-learn version
        # Check sklearn's version and documentation for this
        super().__init__(MultinomialNB(), param_grid)


class BernoulliNBSKLearnModel(SKLearnModel):
    def __init__(self):
        # Adjusting the alpha parameter range and reducing the number of points
        param_grid = {
            'alpha': np.linspace(0.01, 1.0, num=100)  # Adjusted alpha range
        }

        # Suppress warnings if the force_alpha parameter is not available
        warnings.filterwarnings("ignore", category=FutureWarning, module="sklearn.naive_bayes")

        super().__init__(BernoulliNB(), param_grid)

class KNeighborsSKLearnModel(SKLearnModel):
    def __init__(self):
        param_grid = {
            'n_neighbors': range(1, 31),
            'weights': ['uniform', 'distance'],
            'algorithm': ['auto', 'ball_tree', 'kd_tree', 'brute'],
            'p': [1, 2]
        }
        super().__init__(KNeighborsClassifier(), param_grid)

# Note: This method might required scaled input data
class LDASKLearnModel(SKLearnModel):
    def __init__(self):
        param_grid = {
            'solver': ['lsqr', 'eigen'],
            'shrinkage': [None, 'auto'] + list(np.linspace(0.1, 1.0, num=10))
        }
        # Suppress specific warnings
        warnings.filterwarnings("ignore", category=UserWarning)

        super().__init__(LinearDiscriminantAnalysis(), param_grid)

class XGBSKLearnModel(SKLearnModel):
    def __init__(self):
        model = XGBClassifier(objective='binary:logistic')
        super().__init__(model)

class CatBoostSKLearnModel(SKLearnModel):
    def __init__(self):
        param_grid = {
            'iterations': [100, 200, 300],
            'learning_rate': [0.01, 0.05, 0.1],
            'depth': [4, 6, 8],
            'l2_leaf_reg': [1, 3, 5, 7]
        }
        super().__init__(CatBoostClassifier(verbose=0), param_grid)

class LGBMSKLearnModel(SKLearnModel):
    def __init__(self):
        param_grid = {
            'num_leaves': [2, 4, 6],  # Further reduce complexity
            'max_depth': [2, 3, 4],   # Shallower trees to avoid overfitting
            'learning_rate': [0.01, 0.02],  # Reduced learning rates for small dataset
            'n_estimators': [10, 20,],  # Adjust number of trees
            'min_child_samples': [5, 10],  # Adjust to handle small dataset
            'subsample': [0.6,  1.0],  # Subsampling rates
            'colsample_bytree': [0.6, 1.0],  # Feature subsampling rates
            'class_weight': ['balanced'],  # Address class imbalance
            'force_row_wise': [True],  # Based on LightGBM suggestion
            # 'force_col_wise': [True]   # Based on LightGBM suggestion
        }
        super().__init__(LGBMClassifier(), param_grid)

def instantiate_sk_learn_models():
    """
    Instantiate and return a list of SKLearn models.

    Returns:
        models (list): List of instantiated SKLearn models.
    """
    # Instantiate each model with its specific grid search parameters
    models = [
        WeightedMajorityVotingClassifierBalancedAccuracyBased(),
        WeightedMajorityVotingClassifier(),
        RandomForestSKLearnModel(),
        GradientBoostingSKLearnModel(),
        AdaBoostSKLearnModel(),
        DummySKLearnModel(),
        LogisticRegressionSKLearnModel(),
        SVCSKLearnModel(),
        DecisionTreeSKLearnModel(),
        GaussianNBSKLearnModel(),
        MultinomialNBSKLearnModel(),
        BernoulliNBSKLearnModel(),
        KNeighborsSKLearnModel(),
        LDASKLearnModel(),
        CatBoostSKLearnModel(),
        LGBMSKLearnModel(),
        XGBSKLearnModel(),
    ]

    return models
