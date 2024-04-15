from abc import ABC, abstractmethod
from src.utils.Logger import setup_logger
import time

class ModelBaseClass(ABC):
    """
    This is the base class for all models in the project.
    """

    def train_with_timing_stats(self, X_train, Y_train):
        """
        This method should be used to train the model and also collect timing stats.
        
        Parameters:
        - X_train: The input training data.
        - Y_train: The target training data.
        """

        logger = setup_logger()
        start_time = time.time()
        self._train(X_train, Y_train)
        end_time = time.time()
        training_time = end_time - start_time
        logger.info(f"Training time: {training_time/60} minutes \n\n")


    @abstractmethod
    def _train(self, X_train, Y_train):
        """
        This method should be implemented by the derived classes to train the model.

        Parameters:
        - X_train: The input training data.
        - Y_train: The target training data.
        """
        pass

    @abstractmethod
    def predict(self, X):
        """
        This method should be implemented by the derived classes to make predictions.

        Parameters:
        - X: The input data for making predictions.

        Returns:
        - The predicted values.
        """
        pass

    @abstractmethod
    def report_trained_parameters(self, X):
        """
        This method should be implemented by the derived classes to report the trained parameters.

        Parameters:
        - X: The input data.

        Returns:
        - The trained parameters.
        """
        pass