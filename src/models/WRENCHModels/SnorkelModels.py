from snorkel.labeling.model import LabelModel, MajorityLabelVoter
# from abc import ABC, abstractmethod
import numpy as np
from src.models.ModelBaseClass import ModelBaseClass
from sklearn.exceptions import NotFittedError

class SnorkelLabelModel(ModelBaseClass):
    def __init__(self, cardinality=2, verbose=True, n_epochs=500, log_freq=100, seed=123, class_balance=None, lr=0.01, l2=0.0, optimizer='sgd'):
        self.model = LabelModel(cardinality=cardinality, verbose=verbose)
        # Store additional training parameters as class attributes
        self.n_epochs = n_epochs
        self.log_freq = log_freq
        self.seed = seed
        self.class_balance = class_balance
        self.lr = lr
        self.l2 = l2
        self.optimizer = optimizer

    def _train(self, L_train, Y_dev=None):
        # Use stored training parameters
        self.model.fit(L_train=L_train, Y_dev=Y_dev, n_epochs=self.n_epochs, log_freq=self.log_freq, seed=self.seed, 
                       class_balance=self.class_balance, lr=self.lr, l2=self.l2, optimizer=self.optimizer)

    def get_fitted_model(self):
        return self.model
    
    def predict(self, L):
        predictions =  self.model.predict(L)
    
        # Handling ties or unexpected values
        for i, pred in enumerate(predictions):
            if pred not in [0, 1]:
                # Replace with a default value or a random choice
                predictions[i] = np.random.choice([0, 1])
        
        return predictions

    def predict_proba(self, L):
        """
        Predict class probabilities for L using the Snorkel LabelModel.

        :param L: Array-like of shape (n_samples, n_features)
        :return: Array of shape (n_samples, n_classes) with class probabilities
        """
        try:
            return self.model.predict_proba(L)
        except AttributeError:
            raise NotImplementedError("predict_proba is not implemented for the Snorkel LabelModel.")
        except NotFittedError as e:
            raise NotFittedError(f"This SnorkelLabelModel instance is not fitted yet. Call 'fit' with appropriate arguments before using this estimator. Error: {e}")
    
    def report_trained_parameters(self):
        return {
            "weights": self.model.get_weights(),
            "n_epochs": self.n_epochs,
            "log_freq": self.log_freq,
            "seed": self.seed,
            "class_balance": self.class_balance,
            "lr": self.lr,
            "l2": self.l2,
            "optimizer": self.optimizer
        }

class SnorkelModelLoader:
    def __init__(self):
        self.baseline_params = {
            'cardinality': 2,
            'verbose': True,
            'n_epochs': 500,
            'log_freq': 100,
            'seed': 123,
            'class_balance': [0.5, 0.5],
            'lr': 0.01,
            'l2': 0.01,
            'optimizer': 'adam'
        }

        self.param_groups = {
            'n_epochs': [100, 1000],
            'log_freq': [10, 50],
            'class_balance': [None, [0.3, 0.7]],
            'lr': [0.001, 0.1],
            'l2': [0.0, 0.1],
            'optimizer': ['sgd', 'adam']
        }

    def load_snorkel_models(self):
        snorkel_models = [SnorkelLabelModel()]
        for param, values in self.param_groups.items():
            for value in values:
                new_params = self.baseline_params.copy()
                new_params[param] = value
                model = SnorkelLabelModel(**new_params)
                snorkel_models.append(model)

        return snorkel_models


class SnorkelMajorityLabelVoter(ModelBaseClass):
    def __init__(self, cardinality=2):
        self.model = MajorityLabelVoter(cardinality=cardinality)

    def _train(self, L_train, Y_train=None):
        # MajorityLabelVoter doesn't have a fit/train method as it's based on majority voting
        pass

    def predict(self, L):
        L_as_int = L.astype(int)
        predictions = self.model.predict(L_as_int)
        
        # Handling ties or unexpected values
        for i, pred in enumerate(predictions):
            if pred not in [0, 1]:
                # Replace with a default value or a random choice
                predictions[i] = np.random.choice([0, 1])
        
        return predictions


    def report_trained_parameters(self):
        # MajorityLabelVoter doesn't have parameters to learn
        return "MajorityLabelVoter has no trainable parameters."

