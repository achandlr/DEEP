from src.models.ModelBaseClass import ModelBaseClass
from sklearn.ensemble import RandomForestClassifier, AdaBoostClassifier
from sklearn.svm import SVC
from sklearn.linear_model import LogisticRegression
from mlens.ensemble import SuperLearner
from sklearn.ensemble import RandomForestClassifier

class BoostingEnsemble(ModelBaseClass):
    """
    A class representing a boosting ensemble model.

    Inherits from ModelBaseClass.
    """

    def _train(self, X_train, Y_train):
        """
        Trains the boosting ensemble model.

        Args:
            X_train (array-like): The input features for training.
            Y_train (array-like): The target labels for training.
        """
        self.model = AdaBoostClassifier()
        self.model.fit(X_train, Y_train)

    def predict(self, X):
        """
        Predicts the target labels for the given input features.

        Args:
            X (array-like): The input features for prediction.

        Returns:
            array-like: The predicted target labels.
        """
        return self.model.predict(X)

    def report_trained_parameters(self):
        """
        Reports the trained parameters of the boosting ensemble model.

        Returns:
            dict: The trained parameters of the model.
        """
        return self.model.get_params()


class StackingEnsemble(ModelBaseClass):
    """
    A class representing a stacking ensemble model.

    Inherits from ModelBaseClass.
    """

    def _train(self, X_train, Y_train):
        """
        Trains the stacking ensemble model.

        Args:
            X_train (array-like): The input features for training.
            Y_train (array-like): The target labels for training.
        """
        self.model = SuperLearner()
        self.model.add([RandomForestClassifier(), SVC(probability=True), LogisticRegression()])
        self.model.add_meta(LogisticRegression())
        self.model.fit(X_train, Y_train)

    def predict(self, X):
        """
        Predicts the target labels for the given input features.

        Args:
            X (array-like): The input features for prediction.

        Returns:
            array-like: The predicted target labels.
        """
        return self.model.predict(X)

    def report_trained_parameters(self):
        """
        Reports the trained parameters of the stacking ensemble model.

        Returns:
            dict: The trained parameters of the model.
        """
        # For simplicity, reporting meta-learner's parameters
        return self.model.layers[-1].get_params()

class BaggingEnsemble(ModelBaseClass):
    """
    A class representing a bagging ensemble model.

    Inherits from ModelBaseClass.
    """

    def _train(self, X_train, Y_train):
        """
        Trains the bagging ensemble model.

        Args:
            X_train (array-like): The input features for training.
            Y_train (array-like): The target labels for training.
        """
        self.model = RandomForestClassifier()
        self.model.fit(X_train, Y_train)

    def predict(self, X):
        """
        Predicts the target labels for the given input features.

        Args:
            X (array-like): The input features for prediction.

        Returns:
            array-like: The predicted target labels.
        """
        return self.model.predict(X)

    def report_trained_parameters(self):
        """
        Reports the trained parameters of the bagging ensemble model.

        Returns:
            dict: The trained parameters of the model.
        """
        return self.model.get_params()
