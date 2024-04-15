from pygam import LogisticGAM
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
import numpy as np
import time
from abc import ABC, abstractmethod
# Assuming the base class is defined in another module
from src.models.ModelBaseClass import ModelBaseClass
from src.utils.Logger import setup_logger
from pygam import LogisticGAM, s, f
from sklearn.metrics import balanced_accuracy_score, accuracy_score, precision_score, recall_score, f1_score, roc_auc_score
import numpy as np

class LogisticGAMModel(ModelBaseClass):
    def __init__(self):
        self.model = LogisticGAM()
        self.logger = setup_logger()

    def _train(self, X_train, Y_train):
        """
        Train the Logistic GAM model.
        """
        self.model.fit(X_train, Y_train)

    def predict(self, X):
        """
        Make predictions using the trained model.
        """
        return self.model.predict(X)

    def report_trained_parameters(self):
        """
        Report the parameter weights of the trained model.
        """
        # Extracting and logging model parameters (coefficients)
        if self.model.terms is not None:
            coefficients = self.model.terms
            self.logger.info(f"Model Coefficients: {coefficients}")
        else:
            self.logger.info("Model is not trained yet.")

    def evaluate_model(self, X_test, y_test):
        """
        Evaluate the model on the test set and print accuracy.
        """
        y_pred = self.predict(X_test)
        accuracy = accuracy_score(y_test, y_pred)
        self.logger.info(f"Model Accuracy: {accuracy}")


class LogisticGAMModel2(ModelBaseClass):
    def __init__(self):
        self.model = LogisticGAM()
        self.logger = setup_logger()

    def _train(self, X_train, Y_train, search_lam=True, n_lam_points=5, n_random_samples=10):
        """
        Train the Logistic GAM model with optional hyperparameter tuning.
        """
        if search_lam:
            # Define the lambda range for each term
            lam = np.logspace(-3, 3, n_lam_points)

            # Create the grid for each term
            lam_grid = [lam] * X_train.shape[1]  # Assuming one lambda per feature

            # Perform grid search
            # Since pyGAM expects a dictionary, we use a dictionary comprehension
            self.model.gridsearch(X_train, Y_train, lam=dict(enumerate(lam_grid)))

        else:
            self.model.fit(X_train, Y_train)

        self.logger.info("Model training complete.")

    def predict(self, X):
        """
        Make predictions using the trained model.
        """
        return self.model.predict(X)

    def evaluate_model(self, X_test, y_test):
        """
        Evaluate the model on the test set with multiple metrics.
        """
        y_pred = self.predict(X_test)
        metrics = {
            "Balanced Accuracy": balanced_accuracy_score(y_test, y_pred),
            "Accuracy": accuracy_score(y_test, y_pred),
            "Precision": precision_score(y_test, y_pred),
            "Recall": recall_score(y_test, y_pred),
            "F1 Score": f1_score(y_test, y_pred),
            "ROC AUC": roc_auc_score(y_test, y_pred)
        }

        for metric, value in metrics.items():
            self.logger.info(f"{metric}: {value}")

        return metrics
    
    def report_trained_parameters(self):
        """
        Report the parameter weights of the trained model.
        """
        # Extracting and logging model parameters (coefficients)
        if self.model.terms is not None:
            coefficients = self.model.terms
            self.logger.info(f"Model Coefficients: {coefficients}")
        else:
            self.logger.info("Model is not trained yet.")

