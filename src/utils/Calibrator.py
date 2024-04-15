import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import brier_score_loss
from sklearn.calibration import CalibratedClassifierCV 
import numpy as np
from sklearn.base import BaseEstimator, RegressorMixin
import calibration as cal
from src.utils.Statistics import cluster_std, find_best_accuracy_subset # find_subset_with_accuracy
import datetime
import pickle
import argparse

class ProbabilisticBinningCalibrator(BaseEstimator, RegressorMixin):
    """
    A probabilistic binning calibrator that maps predicted probabilities to calibrated probabilities.

    Parameters:
    - model: The base model used for prediction.
    - min_bins: The minimum number of bins to use for calibration. Default is 2.
    - max_bins: The maximum number of bins to use for calibration. Default is 10.
    """
    def __init__(self, model, min_bins=2, max_bins=10):
        self.model = model
        self.min_bins = max(5, min_bins)
        self.max_bins = max(self.min_bins, max_bins)

    def _validate_input(self, y_true, y_pred_probs):
        """
        Validate the input arrays.

        Parameters:
        - y_true: The true labels.
        - y_pred_probs: The predicted probabilities.

        Raises:
        - ValueError: If the input arrays are not numpy arrays or have different lengths.
        - ValueError: If the predicted probabilities are not between 0 and 1.
        """
        if not isinstance(y_true, np.ndarray) or not isinstance(y_pred_probs, np.ndarray):
            raise ValueError("y_true and y_pred_probs must be numpy arrays")
        if len(y_true) != len(y_pred_probs):
            raise ValueError("y_true and y_pred_probs must be the same length")
        if not np.all((y_pred_probs >= 0) & (y_pred_probs <= 1)):
            raise ValueError("Predicted probabilities must be between 0 and 1")

    def _bin_accuracies(self, y_true, y_pred_probs, num_bins):
        """
        Bin the accuracies based on the predicted probabilities.

        Parameters:
        - y_true: The true labels.
        - y_pred_probs: The predicted probabilities.
        - num_bins: The number of bins to use.

        Returns:
        - bin_true_means: The mean of y_true for each bin.
        - bin_edges: The bin edges.
        """
        # Step 1: Sort y_pred_probs and y_true based on the sorted indices of y_pred_probs
        sorted_indices = np.argsort(y_pred_probs)
        sorted_probs = y_pred_probs[sorted_indices]
        sorted_true = y_true[sorted_indices]

        # Step 2: Split sorted_probs and sorted_true into num_bins bins
        bins_probs = np.array_split(sorted_probs, num_bins)
        bins_true = np.array_split(sorted_true, num_bins)
        assert all(len(bin_probs) == len(bin_true) for bin_probs, bin_true in zip(bins_probs, bins_true))

        # Step 3: Calculate the mean of y_true for each bin
        bin_true_means = [np.mean(bin) for bin in bins_true]

        # Step 4: Determine bin edges
        bin_edges = [0]
        for bin in bins_probs:
            if len(bin_edges) == 0 or bin_edges[-1] != bin[0]:
                bin_edges.append(bin[0])
        bin_edges[-1] = 1  # Ensure the last edge is 1
        bin_edges = np.array(bin_edges)

        return bin_true_means, bin_edges

    def fit(self, y_true, y_pred_probs):
        """
        Fit the calibration model using the true labels and predicted probabilities.

        Parameters:
        - y_true: The true labels.
        - y_pred_probs: The predicted probabilities.
        """
        self._validate_input(y_true, y_pred_probs)

        self.bin_accuracies_sets = []
        self.bin_edges_sets = []

        for num_bins in range(self.min_bins, self.max_bins + 1):
            bin_true_means, bin_edges = self._bin_accuracies(y_true, y_pred_probs, num_bins)
            self.bin_accuracies_sets.append(bin_true_means)
            self.bin_edges_sets.append(bin_edges)
        
        # Initialize lists for training data
        X_train = []
        y_train = []

        # Iterate over each predicted probability and bin size
        for predicted_probability, true_label in zip(y_pred_probs, y_true):
            accuracy_for_this_predicted_probability = []
            for bin_size_index, bin_size in enumerate(range(self.min_bins, self.max_bins + 1)):
                bin_index = np.digitize([predicted_probability], self.bin_edges_sets[bin_size_index])[0] - 1
                bin_index = min(bin_index, len(self.bin_accuracies_sets[bin_size_index]) - 1)  # Ensure index is in range
                bin_accuracy = self.bin_accuracies_sets[bin_size_index][bin_index]
                accuracy_for_this_predicted_probability.append(bin_accuracy)

            X_train.append(accuracy_for_this_predicted_probability)
            y_train.append(true_label)

        # Calculate equal weights for all bins
        num_bins = self.max_bins - self.min_bins + 1
        equal_weights = np.ones(num_bins) / num_bins
        self.importance_weights = equal_weights

    def predict(self, y_test):
        """
        Predict the labels for the test data.

        Parameters:
        - y_test: The test labels.

        Returns:
        - y_pred: The predicted labels.
        """
        y_pred_probs = self.predict_proba(y_test)
        # convert to binary
        y_pred = np.where(y_pred_probs > 0.5, 1, 0)
        return y_pred

    def predict_proba(self, y_test):
        """
        Predict the probabilities for the test data.

        Parameters:
        - y_test: The test labels.

        Returns:
        - calibrated_probs: The calibrated probabilities.
        """
        if not hasattr(self, "importance_weights"):
            raise Exception("Model not trained. Please call fit() before predict_proba().")

        # Generate predicted probabilities using the uncalibrated model
        y_pred_probs = self.model.predict_proba(y_test)[:, 1]  # Assuming binary classification

        # Initialize the array for calibrated probabilities
        calibrated_probs = []

        # Iterate over each predicted probability
        for predicted_probability in y_pred_probs:
            weighted_prob = 0

            # Calculate weighted sum of bin accuracies for each bin size
            for bin_size_index, bin_size in enumerate(range(self.min_bins, self.max_bins + 1)):
                bin_index = np.digitize([predicted_probability], self.bin_edges_sets[bin_size_index])[0] - 1
                bin_index = min(bin_index, len(self.bin_accuracies_sets[bin_size_index]) - 1)  # Ensure index is in range

                bin_accuracy = self.bin_accuracies_sets[bin_size_index][bin_index]
                weight = self.importance_weights[bin_size_index]
                weighted_prob += bin_accuracy * weight

            calibrated_probs.append(weighted_prob)

        return np.array(calibrated_probs)

class BBQCalibration:
    """
    BBQ Calibration class that maps predicted probabilities to calibrated probabilities.

    Parameters:
    - model: The base model used for prediction.
    - n_bins: The number of bins to use for calibration. Default is 10.
    - binning_type: The type of binning to use. Can be 'uniform' or 'dynamic'. Default is 'uniform'.
    """
    def __init__(self, model, n_bins=10, binning_type='uniform'):
        self.model = model
        self.n_bins = n_bins
        self.binning_type = binning_type
        self.bin_edges = None
        self.bin_true_probs = None

    def fit(self, y_true, y_pred_probs):
        """
        Fit the calibration model using the true labels and predicted probabilities.

        Parameters:
        - y_true: The true labels.
        - y_pred_probs: The predicted probabilities.
        """
        if len(y_true) != len(y_pred_probs):
            raise ValueError("y_true and y_pred_probs must be of the same length.")

        # Handling uniform binning
        if self.binning_type == 'uniform':
            raise ValueError("BBQ Calibration does not support uniform binning.")
            self.bin_edges = np.linspace(0, 1, self.n_bins + 1)
        elif self.binning_type == 'dynamic':
            # Step 1: Sort y_pred_probs and y_true based on the sorted indices of y_pred_probs
            sorted_indices = np.argsort(y_pred_probs)
            sorted_probs = y_pred_probs[sorted_indices]
            sorted_true = y_true[sorted_indices]

            # Step 2: Split sorted_probs and sorted_true into num_bins bins
            bins_probs = np.array_split(sorted_probs, self.n_bins)
            bins_true = np.array_split(sorted_true, self.n_bins)
            assert all(len(bin_probs) == len(bin_true) for bin_probs, bin_true in zip(bins_probs, bins_true))

            # Step 3: Calculate the mean of y_true for each bin
            bin_true_means = [np.mean(bin) for bin in bins_true]

            # Step 4: Determine bin edges
            bin_edges = [0]
            for bin in bins_probs:
                if len(bin_edges) == 0 or bin_edges[-1] != bin[0]:
                    bin_edges.append(bin[0])
            bin_edges[-1] = 1  # Ensure the last edge is 1
            bin_edges = np.array(bin_edges)

            self.bin_edges = bin_edges
            self.n_bins = len(bins_probs)
            self.bin_true_probs = bin_true_means

        return

    def predict_proba(self, y_test):
        """
        Predict the probabilities for the test data.

        Parameters:
        - y_test: The test labels.

        Returns:
        - calibrated_probs: The calibrated probabilities.
        """
        y_pred_probs = self.model.predict_proba(y_test)[:, 1]
        bin_indices = np.digitize(y_pred_probs, self.bin_edges) - 1
        # Clip bin_indices to ensure they are within the range [0, self.n_bins - 1]
        bin_indices = np.clip(bin_indices, 0, self.n_bins - 1)

        # Use bin_indices to index self.bin_true_probs
        calibrated_probs = np.array([self.bin_true_probs[index] for index in bin_indices])
        return calibrated_probs

    def predict(self, y_test):
        """
        Predict the labels for the test data.

        Parameters:
        - y_test: The test labels.

        Returns:
        - y_pred: The predicted labels.
        """
        y_pred_probs = self.predict_proba(y_test)
        # convert to binary
        y_pred = np.where(y_pred_probs > 0.5, 1, 0)
        return y_pred


class HistogramBinningCalibration:
    """
    Histogram Binning Calibration class that maps predicted probabilities to calibrated probabilities.

    Parameters:
    - model: The base model used for prediction.
    - n_bins: The number of bins to use for calibration. Default is 10.
    - binning_type: The type of binning to use. Can be 'uniform' or 'dynamic'. Default is 'uniform'.
    """
    def __init__(self, model, n_bins=10, binning_type='uniform'):
        self.model = model
        self.n_bins = n_bins
        self.binning_type = binning_type
        self.bin_edges = None
        self.bin_true_probs = None

    def fit(self, y_true, y_pred_probs):
        """
        Fit the calibration model using the true labels and predicted probabilities.

        Parameters:
        - y_true: The true labels.
        - y_pred_probs: The predicted probabilities.
        """
        if self.binning_type == 'uniform':
            # Create uniform bin edges
            self.bin_edges = np.linspace(0, 1, self.n_bins + 1)
        elif self.binning_type == 'dynamic':
            raise ValueError("Histogram Binning Calibration does not support dynamic binning.")
            # Use quantiles for dynamic bin edges
            percentiles = np.linspace(0, 100, self.n_bins + 1)
            self.bin_edges = np.percentile(y_pred_probs, percentiles)
        else:
            raise ValueError("Invalid binning type. Choose 'uniform' or 'dynamic'.")

        # Assign each prediction to a bin
        bin_indices = np.digitize(y_pred_probs, self.bin_edges) - 1
        self.bin_true_probs = np.zeros(self.n_bins)

        # Calculate the true probability for each bin
        for i in range(self.n_bins):
            mask = bin_indices == i
            if np.sum(mask) > 0:
                self.bin_true_probs[i] = np.mean(y_true[mask])
            else:
                self.bin_true_probs[i] = None # 0.5  # Handle empty bins

        # Fill in empty bins
        for i in range(self.n_bins):
            if self.bin_true_probs[i] is None or np.isnan(self.bin_true_probs[i]):
                self.bin_true_probs[i] = 0.5  # Handle empty bins
                left = self.bin_true_probs[i - 1] if i > 0 else np.nan
                right = self.bin_true_probs[i + 1] if i < self.n_bins - 1 else np.nan

                if not np.isnan(left) and not np.isnan(right):
                    self.bin_true_probs[i] = (left + right) / 2
                elif not np.isnan(left):
                    self.bin_true_probs[i] = left
                elif not np.isnan(right):
                    self.bin_true_probs[i] = right
                else:
                    raise ValueError()
        return


    def predict_proba(self, y_test):
        """
        Predict calibrated probabilities.
        """
        if self.bin_true_probs is None:
            raise ValueError("Fit method must be called before predict_proba.")
        y_pred_probs = self.model.predict_proba(y_test)[:, 1]
        bin_indices = np.digitize(y_pred_probs, self.bin_edges) - 1

        # Clip bin_indices to ensure they are within the range [0, self.n_bins - 1]
        bin_indices = np.clip(bin_indices, 0, self.n_bins - 1)

        return self.bin_true_probs[bin_indices]

    def predict(self, y_test):
        """
        Predict binary labels.

        Parameters:
        - y_test: The test data.

        Returns:
        - y_pred: The predicted binary labels.
        """
        y_pred_probs = self.predict_proba(y_test)
        # convert to binary
        y_pred = np.where(y_pred_probs > 0.5, 1, 0)
        return y_pred


class Calibrator:
    """
    A class that represents a calibrator for a classifier model.

    Attributes:
        classifier: The classifier model to be calibrated.
        X_val: The validation data used for calibration.
        y_val: The validation labels used for calibration.
        X_test: The test data used for evaluation.
        Y_test: The test labels used for evaluation.
        cv: The number of cross-validation folds for calibration. Default is None.
        visualization_path: The path to save visualizations. Default is "data/visualizations".
        platt_scaling_model: The calibrated classifier using Platt scaling method.
        isotonic_model: The calibrated classifier using isotonic regression method.
        platt_binner_marginal_calibratior: The PlattBinnerCalibrator for marginal calibration.
        platt_binner_top_calibratior: The PlattBinnerCalibrator for top calibration.
        histogram_top_calibrator: The HistogramCalibrator for top calibration.
        platt_calibrator: The PlattCalibrator for calibration.
        histogram_calibrator: The HistogramCalibrator for calibration.
        bbq_scaling_model_dynamic_20: The BBQCalibration model with dynamic binning and 20 bins.
        bbq_scaling_model_dynamic_10: The BBQCalibration model with dynamic binning and 10 bins.
        hist_binning_model_uniform_20: The HistogramBinningCalibration model with uniform binning and 20 bins.
        hist_binning_model_uniform_10: The HistogramBinningCalibration model with uniform binning and 10 bins.
        probabilistic_binning_model: The ProbabilisticBinningCalibrator model.
        calibrated_models: A dictionary containing all the calibrated models.

    Methods:
        initialize_calibrators: Initializes the calibration models.
        fit_calibrators: Fits the calibration models.
        calculate_calibration_errors: Calculates the Expected Calibration Error (ECE) and Maximum Calibration Error (MCE).
        calculate_ece_method_2: Calculates the ECE using the uniform binning approach.
        evaluate_calibrators: Evaluates the calibrated models.
    """

    def __init__(self, classifier, X_val, y_val, X_test, Y_test, cv=None):
        """
        Initializes a Calibrator object.

        Args:
            classifier: The classifier model to be calibrated.
            X_val: The validation data used for calibration.
            y_val: The validation labels used for calibration.
            X_test: The test data used for evaluation.
            Y_test: The test labels used for evaluation.
            cv: The number of cross-validation folds for calibration. Default is None.
        """
        self.classifier = classifier
        self.X_val = X_val
        self.y_val = y_val
        self.X_test = X_test
        self.Y_test = Y_test
        self.visualization_path = "data/visualizations"
        self.cv = cv
        self.initialize_calibrators()

    def initialize_calibrators(self):
        """
        Initializes the calibrators for the classifier.

        This method initializes various calibrators for the classifier, including Platt scaling, isotonic, BBQ scaling,
        histogram binning, and probabilistic binning calibrators. It also creates a dictionary of calibrated models.

        Returns:
            None
        """
        if hasattr(self.classifier, "get_fitted_model"):
            fitted_model = self.classifier.get_fitted_model()
        else:
            fitted_model = self.classifier
        if self.cv != None:
            self.platt_scaling_model = CalibratedClassifierCV(fitted_model, method='sigmoid', cv=self.cv)
            self.isotonic_model = CalibratedClassifierCV(fitted_model, method='isotonic', cv=self.cv)
        else:
            self.platt_scaling_model = CalibratedClassifierCV(fitted_model, method='sigmoid', cv='prefit')
            self.isotonic_model = CalibratedClassifierCV(fitted_model, method='isotonic', cv='prefit')

        self.platt_binner_marginal_calibratior = cal.PlattBinnerCalibrator(num_calibration = len(self.y_val), num_bins=10)

        self.platt_binner_top_calibratior = cal.PlattBinnerCalibrator(num_calibration = len(self.y_val), num_bins=10)
        self.histogram_top_calibrator = cal.HistogramCalibrator(num_calibration = len(self.y_val), num_bins=10)
        self.platt_calibrator = cal.PlattCalibrator(num_calibration = len(self.y_val), num_bins=10)
        self.histogram_calibrator = cal.HistogramCalibrator(num_calibration = len(self.y_val), num_bins=10)

        self.bbq_scaling_model_dynamic_20 = BBQCalibration(fitted_model, n_bins=20, binning_type='dynamic')
        self.bbq_scaling_model_dynamic_10 = BBQCalibration(fitted_model, n_bins=10, binning_type='dynamic') 
        self.hist_binning_model_uniform_20 = HistogramBinningCalibration(fitted_model, n_bins=20, binning_type='uniform')
        self.hist_binning_model_uniform_10 = HistogramBinningCalibration(fitted_model, n_bins=10, binning_type='uniform')

        try:
            predict_proba = fitted_model.predict_proba(self.X_val)[:, 1]
            num_unique_vals = len(np.unique(predict_proba))
            num_desired_bins = max(num_unique_vals//10, 4)
            min_bins = min(num_unique_vals, 3)
            self.probabilistic_binning_model = ProbabilisticBinningCalibrator(fitted_model, min_bins=min_bins, max_bins=num_desired_bins)
        except Exception as e:
            self.probabilistic_binning_model = None
        self.calibrated_models = {"original" : fitted_model, "platt_scaling": self.platt_scaling_model, "isotonic": self.isotonic_model, "bbq_dynamic_20": self.bbq_scaling_model_dynamic_20, "bbq_dynamic_10": self.bbq_scaling_model_dynamic_10, "hist_uniform_20": self.hist_binning_model_uniform_20, "hist_uniform_10": self.hist_binning_model_uniform_10, "probabilistic_binning": self.probabilistic_binning_model}

        return

    def fit_calibrators(self):
        """
        Fits various calibrators to the validation data.

        This method fits different calibrators to the validation data, including platt scaling models,
        isotonic models, binner marginal calibrator, binner top calibrator, histogram top calibrator,
        platt calibrator, histogram calibrator, BBQ dynamic 20 model, BBQ dynamic 10 model, hist uniform 20 model,
        hist uniform 10 model, and probabilistic binning model.

        Returns:
            None
        """
        try:
            self.platt_scaling_model.fit(self.X_val, self.y_val)
            self.isotonic_model.fit(self.X_val, self.y_val)
        except Exception as e:
            print(e)
            print("Unable to fit platt scaling and isotonic models")

        predicted_probs = self.classifier.predict_proba(self.X_val)[:, 1]

        try:
            self.platt_binner_marginal_calibratior.train_calibration(predicted_probs,  self.y_val)
        except Exception as e:
            print(e)
            print("Unable to fit binner marginal calibrator")

        try:
            self.platt_binner_top_calibratior.train_calibration(predicted_probs,  self.y_val) 
        except Exception as e:
            print(e)
            print("Unable to fit binner top calibrator")
        try:
            self.histogram_top_calibrator.train_calibration(predicted_probs,  self.y_val) 
        except Exception as e:
            print(e)
            print("Unable to fit histogram top calibrator")
        try:
            self.platt_calibrator.train_calibration(predicted_probs,  self.y_val)
        except Exception as e:
            print(e)
            print("Unable to fit platt calibrator")
        try:
            self.histogram_calibrator.train_calibration(predicted_probs,  self.y_val)
        except Exception as e:
            print(e)
            print("Unable to fit histogram calibrator")
        try:
            self.bbq_scaling_model_dynamic_20.fit(self.y_val, predicted_probs)
        except Exception as e:
            pass
        try:
            self.bbq_scaling_model_dynamic_10.fit(self.y_val, predicted_probs)
        except Exception as e:
            pass
        try:
            self.hist_binning_model_uniform_20.fit(self.y_val, predicted_probs)
        except Exception as e:
            pass
        try:
            self.hist_binning_model_uniform_10.fit(self.y_val, predicted_probs)
        except Exception as e:
            pass
        try:   
            self.probabilistic_binning_model.fit(self.y_val, predicted_probs)
        except Exception as e:
            pass
        return
    
    @staticmethod
    def calculate_calibration_errors(true_labels, predicted_probs, n_bins=10, binning_strategy='uniform', min_probability = 0):
        """
        Calculate Expected Calibration Error (ECE) and Maximum Calibration Error (MCE).

        Args:
            true_labels (np.array): Array of true labels.
            predicted_probs (np.array): Array of predicted probabilities.
            n_bins (int): Number of bins to use for calibration error calculation.
            binning_strategy (str): Strategy for binning ('uniform' or 'adaptive').
            min_probability (float): Minimum probability value. Default is 0.

        Returns:
            tuple: A tuple containing ECE and MCE.
        """

        if binning_strategy == 'uniform':
            bin_boundaries = np.linspace(min_probability, 1, n_bins + 1)
        elif binning_strategy == 'adaptive':
            # Adaptive binning based on quantiles
            bin_boundaries = np.quantile(predicted_probs, np.linspace(min_probability, 1, n_bins + 1))
        else:
            raise ValueError("Invalid binning strategy. Choose 'uniform' or 'adaptive'.")

        bin_lowers = bin_boundaries[:-1]
        bin_uppers = bin_boundaries[1:]

        ece = 0.0
        mce = 0.0

        for bin_lower, bin_upper in zip(bin_lowers, bin_uppers):
            in_bin = (predicted_probs > bin_lower) & (predicted_probs <= bin_upper)
            bin_size = np.sum(in_bin)

            if bin_size > 0:
                accuracy_in_bin = np.mean(true_labels[in_bin] == 1)
                confidence_in_bin = np.mean(predicted_probs[in_bin])
                ece += np.abs(accuracy_in_bin - confidence_in_bin) * (bin_size / len(predicted_probs))
                mce = max(mce, np.abs(accuracy_in_bin - confidence_in_bin))

        return ece, mce

    @staticmethod
    def calculate_ece_method_2(true_labels, pred_probas, M=5):
        """
        Calculate Expected Calibration Error (ECE) using the uniform binning approach.

        Parameters:
        true_labels (array-like): True labels for the samples.
        pred_probas (array-like): Predicted probabilities for class 1.
        M (int, optional): Number of bins. Defaults to 5.

        Returns:
        float: Expected Calibration Error (ECE) value.
        """

        # Uniform binning approach with M number of bins
        bin_boundaries = np.linspace(0, 1, M + 1)
        bin_lowers = bin_boundaries[:-1]
        bin_uppers = bin_boundaries[1:]

        # Confidences are the predicted probabilities for class 1
        confidences = pred_probas

        # Calculate accuracies: true_labels == 1 for correct class 1 predictions, and true_labels == 0 otherwise
        # For binary classification, this direct comparison is sufficient
        accuracies = (confidences > 0.5) == true_labels

        ece = 0
        for bin_lower, bin_upper in zip(bin_lowers, bin_uppers):
            # Determine if sample falls within the current bin
            in_bin = np.logical_and(confidences >= bin_lower, confidences < bin_upper)
            prob_in_bin = np.mean(in_bin)

            if prob_in_bin > 0:
                # Calculate accuracy for predictions within the bin
                accuracy_in_bin = np.mean(accuracies[in_bin])
                # Calculate average confidence for predictions within the bin
                avg_confidence_in_bin = np.mean(confidences[in_bin])
                # Update ECE with the contribution from this bin
                ece += np.abs(avg_confidence_in_bin - accuracy_in_bin) * prob_in_bin

        return ece

    def evaluate_calibrators(self):
        """
        Evaluate different calibrators and calculate various calibration metrics.

        This method evaluates different calibrators by calculating various calibration metrics such as Brier score,
        calibration error, maximum calibration error, and cluster standard deviation. It also finds the best accuracy
        subset for each calibrator.

        Returns:
            None
        """
        results = {}
        y_pred_proba = self.classifier.predict_proba(self.X_test)[:, 1]
        original_brier = brier_score_loss(self.Y_test, y_pred_proba)
        original_ece, original_mce = Calibrator.calculate_calibration_errors(self.Y_test, y_pred_proba)
        original_cluster_std = cluster_std(y_pred_proba)
        uncalibrated_final_size, uncalibrated_pred_proba_best_possible_acc = find_best_accuracy_subset(pred_probs = y_pred_proba, ground_truth = self.Y_test, min_needed_samples=15)

        try:
            platt_proba = self.platt_scaling_model.predict_proba(self.X_test)[:, 1]
            platt_ece, platt_mce = Calibrator.calculate_calibration_errors(self.Y_test, platt_proba)
            ece_3_bin = Calibrator.calculate_ece_method_2(self.Y_test, platt_proba, M=3)
            ece_5_bin = Calibrator.calculate_ece_method_2(self.Y_test, platt_proba, M=5)
            ece_10_bin = Calibrator.calculate_ece_method_2(self.Y_test, platt_proba, M=10)
            platt_brier = brier_score_loss(self.Y_test, platt_proba)
            platt_cluster_std = cluster_std(platt_proba)
            platt_max_proba = max(platt_proba)
            platt_final_size, platt_pred_proba_best_possible_acc = find_best_accuracy_subset(pred_probs = platt_proba, ground_truth = self.Y_test, min_needed_samples=15)
        except Exception as e:
            platt_ece = None
            platt_mce = None
            platt_brier = None
            platt_cluster_std = None
            platt_pred_proba_best_possible_acc = None
            platt_proba = None


        try:
            calibrated_probs = self.platt_binner_marginal_calibratior.calibrate(y_pred_proba)
            platt_binner_marginal_calibratior_ece, platt_binner_marginal_calibratior_mce = Calibrator.calculate_calibration_errors(self.Y_test, calibrated_probs)
            platt_binner_marginal_calibratior_ece_3_bin = Calibrator.calculate_ece_method_2(self.Y_test, calibrated_probs, M=3)
            platt_binner_marginal_calibratior_ece_5_bin = Calibrator.calculate_ece_method_2(self.Y_test, calibrated_probs, M=5)
            platt_binner_marginal_calibratior_ece_10_bin = Calibrator.calculate_ece_method_2(self.Y_test, calibrated_probs, M=10)
            platt_binner_marginal_calibratior_brier = brier_score_loss(self.Y_test, platt_proba)
            platt_binner_marginal_calibratior_cluster_std = cluster_std(calibrated_probs)

        except Exception as e:
            platt_binner_marginal_calibratior_ece = None
            platt_binner_marginal_calibratior_mce = None
            platt_binner_marginal_calibratior_brier = None
            platt_binner_marginal_calibratior_cluster_std = None
        try:
            calibrated_probs = self.platt_binner_top_calibratior.calibrate(y_pred_proba)
            platt_binner_top_calibratior_ece, platt_binner_top_calibratior_mce = Calibrator.calculate_calibration_errors(self.Y_test, calibrated_probs)
            platt_binner_top_calibratior_ece_3_bin = Calibrator.calculate_ece_method_2(self.Y_test, calibrated_probs, M=3)
            platt_binner_top_calibratior_ece_5_bin = Calibrator.calculate_ece_method_2(self.Y_test, calibrated_probs, M=5)
            platt_binner_top_calibratior_ece_10_bin = Calibrator.calculate_ece_method_2(self.Y_test, calibrated_probs, M=10)
            platt_binner_top_calibratior_brier = brier_score_loss(self.Y_test, platt_proba)
            platt_binner_top_calibratior_cluster_std = cluster_std(calibrated_probs)
        except Exception as e:
            platt_binner_top_calibratior_ece = None
            platt_binner_top_calibratior_mce = None
            platt_binner_top_calibratior_brier = None
            platt_binner_top_calibratior_cluster_std = None

        try:
            calibrated_probs = self.histogram_top_calibrator.calibrate(y_pred_proba)
            histogram_top_calibrator_ece, histogram_top_calibrator_mce = Calibrator.calculate_calibration_errors(self.Y_test, calibrated_probs)
            histogram_top_calibrator_ece_3_bin = Calibrator.calculate_ece_method_2(self.Y_test, calibrated_probs, M=3)
            histogram_top_calibrator_ece_5_bin = Calibrator.calculate_ece_method_2(self.Y_test, calibrated_probs, M=5)
            histogram_top_calibrator_ece_10_bin = Calibrator.calculate_ece_method_2(self.Y_test, calibrated_probs, M=10)
            histogram_top_calibrator_brier = brier_score_loss(self.Y_test, platt_proba)
            histogram_top_calibrator_cluster_std = cluster_std(calibrated_probs)
        except Exception as e:
            histogram_top_calibrator_ece = None
            histogram_top_calibrator_mce = None
            histogram_top_calibrator_brier = None
            histogram_top_calibrator_cluster_std = None

        try:
            calibrated_probs = self.platt_calibrator.calibrate(y_pred_proba)
            platt_calibrator_ece, platt_calibrator_mce = Calibrator.calculate_calibration_errors(self.Y_test, calibrated_probs)
            platt_calibrator_ece_3_bin = Calibrator.calculate_ece_method_2(self.Y_test, calibrated_probs, M=3)
            platt_calibrator_ece_5_bin = Calibrator.calculate_ece_method_2(self.Y_test, calibrated_probs, M=5)
            platt_calibrator_ece_10_bin = Calibrator.calculate_ece_method_2(self.Y_test, calibrated_probs, M=10)
            platt_calibrator_brier = brier_score_loss(self.Y_test, platt_proba)
            platt_calibrator_cluster_std = cluster_std(calibrated_probs)
        except Exception as e:
            platt_calibrator_ece = None
            platt_calibrator_mce = None
            platt_calibrator_brier = None
            platt_calibrator_cluster_std = None

        try:
            calibrated_probs = self.histogram_calibrator.calibrate(y_pred_proba)
            histogram_calibrator_ece, histogram_calibratormce = Calibrator.calculate_calibration_errors(self.Y_test, calibrated_probs)
            histogram_calibrator_ece_3_bin = Calibrator.calculate_ece_method_2(self.Y_test, calibrated_probs, M=3)
            histogram_calibrator_ece_5_bin = Calibrator.calculate_ece_method_2(self.Y_test, calibrated_probs, M=5)
            histogram_calibrator_ece_10_bin = Calibrator.calculate_ece_method_2(self.Y_test, calibrated_probs, M=10)
            histogram_calibrator_brier = brier_score_loss(self.Y_test, platt_proba)
            histogram_calibrator_cluster_std = cluster_std(calibrated_probs)
        except Exception as e:
            histogram_calibrator_ece = None
            histogram_calibratormce = None
            histogram_calibrator_brier = None
            histogram_calibrator_cluster_std = None


        try:
            isotonic_proba = self.isotonic_model.predict_proba(self.X_test)[:, 1]
            isotonic_ece, isotonic_mce = Calibrator.calculate_calibration_errors(self.Y_test, isotonic_proba)
            isotonic_brier = brier_score_loss(self.Y_test, isotonic_proba)
            isotonic_cluster_std = cluster_std(isotonic_proba)
        except Exception as e:
            isotonic_ece = None
            isotonic_mce = None
            isotonic_brier = None
            isotonic_cluster_std = None

        try:
            bbq_scaling_model_dynamic_10_proba = self.bbq_scaling_model_dynamic_10.predict_proba(self.X_test)
            bbq_dynamic_10_ece, bbq_dynamic_10_mce = Calibrator.calculate_calibration_errors(self.Y_test, bbq_scaling_model_dynamic_10_proba)
            bbq_dynamic_10_brier = brier_score_loss(self.Y_test, bbq_scaling_model_dynamic_10_proba)
            bbq_dynamic_10_cluster_std = cluster_std(bbq_scaling_model_dynamic_10_proba)
        except Exception as e:
            bbq_dynamic_10_ece = None
            bbq_dynamic_10_mce = None
            bbq_dynamic_10_brier = None
            bbq_dynamic_10_cluster_std = None

        try:
            bbq_scaling_model_dynamic_20_proba = self.bbq_scaling_model_dynamic_20.predict_proba(self.X_test)
            bbq_dynamic_20_ece, bbq_dynamic_20_mce = Calibrator.calculate_calibration_errors(self.Y_test, bbq_scaling_model_dynamic_20_proba)
            bbq_dynamic_20_brier = brier_score_loss(self.Y_test, bbq_scaling_model_dynamic_20_proba)
            bbq_dynamic_20_cluster_std = cluster_std(bbq_scaling_model_dynamic_20_proba)
        except Exception as e:
            bbq_dynamic_20_ece = None
            bbq_dynamic_20_mce = None
            bbq_dynamic_20_brier = None
            bbq_dynamic_20_cluster_std = None

        try:
            hist_uniform_10_ece_proba = self.hist_binning_model_uniform_10.predict_proba(self.X_test)
            hist_uniform_10_ece, hist_uniform_10_mce = Calibrator.calculate_calibration_errors(self.Y_test, hist_uniform_10_ece_proba)
            hist_uniform_10_brier = brier_score_loss(self.Y_test, hist_uniform_10_ece_proba)
            hist_uniform_10_cluster_std = cluster_std(hist_uniform_10_ece_proba)
        except Exception as e:
            hist_uniform_10_ece = None
            hist_uniform_10_mce = None
            hist_uniform_10_brier = None
            hist_uniform_10_cluster_std = None

        try:
            hist_uniform_20_ece_proba = self.hist_binning_model_uniform_20.predict_proba(self.X_test)
            hist_uniform_20_ece, hist_uniform_20_mce = Calibrator.calculate_calibration_errors(self.Y_test, hist_uniform_20_ece_proba)
            hist_uniform_20_brier = brier_score_loss(self.Y_test, hist_uniform_20_ece_proba)
            hist_uniform_20_cluster_std = cluster_std(hist_uniform_20_ece_proba)
        except Exception as e:
            hist_uniform_20_ece = None
            hist_uniform_20_mce = None
            hist_uniform_20_brier = None
            hist_uniform_20_cluster_std = None

        try:
            probabilist_binning_proba = self.probabilistic_binning_model.predict_proba(self.X_test)
            probabilist_binning_ece, probabilist_binning_mce = Calibrator.calculate_calibration_errors(self.Y_test, probabilist_binning_proba)
            probalistic_brier = brier_score_loss(self.Y_test, probabilist_binning_proba)
            probalistic_cluster_std = cluster_std(probabilist_binning_proba)
        except Exception as e:
            probabilist_binning_ece = None
            probabilist_binning_mce = None
            probalistic_brier = None
            probalistic_cluster_std = None

        results['original'] = {
            'ECE': original_ece,
            'MCE': original_mce,
            'Brier': original_brier,
            "cluster_std": original_cluster_std,
            "pred_proba_best_possible_acc": uncalibrated_pred_proba_best_possible_acc
        }
        results['platt_scaling'] = {
            'ECE': platt_ece,
            'MCE': platt_mce,
            'Brier': platt_brier,
            "cluster_std": platt_cluster_std,
            "pred_proba_best_possible_acc": platt_pred_proba_best_possible_acc,
            "pred_proba": platt_proba
        }
        results['isotonic'] = {
            'ECE': isotonic_ece,
            'MCE': isotonic_mce,
            'Brier': isotonic_brier,
            "cluster_std": isotonic_cluster_std
        }

        results['bbq_dynamic_20'] = {
            'ECE': bbq_dynamic_20_ece,
            'MCE': bbq_dynamic_20_mce,
            'Brier': bbq_dynamic_20_brier,
            "cluster_std": bbq_dynamic_20_cluster_std
        }
        results['bbq_dynamic_10'] = {
            'ECE': bbq_dynamic_10_ece,
            'MCE': bbq_dynamic_10_mce,
            'Brier': bbq_dynamic_10_brier,
            "cluster_std": bbq_dynamic_10_cluster_std
        }

        results['hist_uniform_20'] = {
            'ECE': hist_uniform_20_ece,
            'MCE': hist_uniform_20_mce,
            'Brier': hist_uniform_20_brier,
        }
        results['hist_uniform_10'] = {
            'ECE': hist_uniform_10_ece,
            'MCE': hist_uniform_10_mce,
            'Brier': hist_uniform_10_brier,
            "cluster_std": hist_uniform_10_cluster_std
        }
        results['probabilistic_binning'] = {
            'ECE': probabilist_binning_ece,
            'MCE': probabilist_binning_mce,
            'Brier': probalistic_brier,
            "cluster_std": probalistic_cluster_std
        }

        results['platt_binner_marginal_calibratior'] = {
            'ECE': platt_binner_marginal_calibratior_ece,
            'MCE': platt_binner_marginal_calibratior_mce,
            'Brier': platt_binner_marginal_calibratior_brier,
            "cluster_std": platt_binner_marginal_calibratior_cluster_std
        }
        results['platt_binner_top_calibratior'] = {
            'ECE': platt_binner_top_calibratior_ece,
            'MCE': platt_binner_top_calibratior_mce,
            'Brier': platt_binner_top_calibratior_brier,
            "cluster_std": platt_binner_top_calibratior_cluster_std
        }
        results['histogram_top_calibrator'] = {
            'ECE': histogram_top_calibrator_ece,
            'MCE': histogram_top_calibrator_mce,
            'Brier': histogram_top_calibrator_brier,
            "cluster_std": histogram_top_calibrator_cluster_std
        }
        results['platt_calibrator'] = {
            'ECE': platt_calibrator_ece,
            'MCE': platt_calibrator_mce,
            'Brier': platt_calibrator_brier,
            "cluster_std": platt_calibrator_cluster_std
        }
        results['histogram_calibrator'] = {
            'ECE': histogram_calibrator_ece,
            'MCE': histogram_calibratormce,
            'Brier': histogram_calibrator_brier,
            "cluster_std": histogram_calibrator_cluster_std
        }

        lowest_ece_value = float('inf')
        lowest_ece_method = None

        for method, metrics in results.items():
            ece = metrics['ECE']
            if ece != None and ece < lowest_ece_value:
                lowest_ece_value = ece
                lowest_ece_method = method

        reduction_in_ece = original_ece - lowest_ece_value

        results['reduction_in_ece'] = reduction_in_ece
        results['lowest_ece_method'] = lowest_ece_method
        results['lowest_ece_value'] = lowest_ece_value
        platt_scaling_reduces_ece =  platt_ece!=None and original_ece!= None and platt_ece < original_ece 
        results['platt_scaling_reduces_ece'] = platt_scaling_reduces_ece

        return results

    def plot_reliability_diagrams(self):
            """
            Plots reliability diagrams for calibrated models.

            This method plots reliability diagrams for each calibrated model in the `calibrated_models` dictionary.
            It calculates the predicted probabilities and confidences for each model, and then saves the statistics
            for the reliability diagrams in a pickled file.

            Returns:
                None
            """
            
            true_labels = self.Y_test
            for model_name, model in self.calibrated_models.items():
                if model_name == "original":
                    title = "Uncalibrated Reliability Diagram"
                else:
                    title = f"Reliability Diagram Calibrating with {model_name}"
                # predicted_probs = self.classifier.predict_proba(self.X_test)[:, 1]
                y_pred_proba = model.predict_proba(self.X_test)
                if len(y_pred_proba.shape) == 2:
                    y_pred_proba = y_pred_proba[:, 1]
                else:
                    pass
                # y_pred_proba = model.predict_proba(self.X_test)[:, 1]
                confidences = np.where(y_pred_proba < 0.5, 1 - y_pred_proba, y_pred_proba)
                y_pred = np.where(y_pred_proba < 0.5, 0, 1)


                if model_name == "original":
                    y_pred_uncalibrated = y_pred
                    y_pred_proba_uncalibrated = y_pred_proba
                    y_pred_confidences_uncalibrated = confidences
                if model_name == "platt_scaling":
                    y_pred_platt_scaling = y_pred
                    y_pred_proba_platt_scaling = y_pred_proba
                    y_pred_confidences_platt_scaling = confidences

            stats_for_reliabilitiy_diagrams = {"model_name": str(self.classifier), "original": {"y_pred": y_pred_uncalibrated, "y_pred_proba": y_pred_proba_uncalibrated, "y_pred_confidences": y_pred_confidences_uncalibrated, "true_labels": true_labels}, "platt_scaling": {"y_pred": y_pred_platt_scaling, "y_pred_proba": y_pred_proba_platt_scaling, "y_pred_confidences": y_pred_confidences_platt_scaling, "true_labels": true_labels}}

            current_datetime = datetime.datetime.now()
            date_time_string = current_datetime.strftime("%Y-%m-%d-%H-%M-%S")
            with open(f"data\generated\stats_for_reliability_diagrams\stats_for_reliabilitiy_diagrams_{date_time_string}.pkl", "wb") as f:
                pickle.dump(stats_for_reliabilitiy_diagrams, f)
            
            return
    

    def plot_reliability_diagram(self, true_labels, pred_labels, confidences,  title = None, model = None):
        """
        Plots the reliability diagram.
        """
        if title == None:
            title="Reliability Diagram"
        try:
            diagram = reliability_diagram(true_labels, pred_labels, confidences, return_fig=True, title=title, model=model)
            # reliability_diagram = reliability_diagrams(n_bins=n_bins, equal_intervals=equal_intervals, return_fig=True)
            diagram.show()
            plt.close()
        except Exception as e:
            print(e)
            print("Unable to plot reliability diagram")
            diagram = None
        return diagram


'''
The following code in this fial is a modified version from https://raw.githubusercontent.com/hollance/reliability-diagrams/master/reliability_diagrams.py
It has not been thoroughly tested and may need to be modified for specific use cases.
'''
def compute_calibration(true_labels, pred_labels, confidences, model=None):
    """Collects predictions into bins used to draw a reliability diagram.

    Arguments:
        true_labels: the true labels for the test examples
        pred_labels: the predicted labels for the test examples
        confidences: the predicted confidences for the test examples
        num_bins: number of bins

    The true_labels, pred_labels, confidences arguments must be NumPy arrays;
    pred_labels and true_labels may contain numeric or string labels.

    For a multi-class model, the predicted label and confidence should be those
    of the highest scoring class.

    Returns a dictionary containing the following NumPy arrays:
        accuracies: the average accuracy for each bin
        confidences: the average confidence for each bin
        counts: the number of examples in each bin
        bins: the confidence thresholds for each bin
        avg_accuracy: the accuracy over the entire test set
        avg_confidence: the average confidence over the entire test set
        expected_calibration_error: a weighted average of all calibration gaps
        max_calibration_error: the largest calibration gap across all bins
    """
    assert(len(confidences) == len(pred_labels))
    assert(len(confidences) == len(true_labels))

    if model != None and hasattr(model, 'bin_edges'):
        bins = model.bin_edges 
    else:
        bins = np.linspace(0.5, 1.0, 11) 
    num_bins = len(bins) - 1 
    indices = np.digitize(confidences, bins, right=True)
    first_bin_index = min(indices) -1
    last_bin_index = max(indices)
    bins = bins[first_bin_index:last_bin_index+1]
    num_bins = len(bins) - 1
    indices = np.digitize(confidences, bins, right=True)
    bin_accuracies = np.zeros(num_bins, dtype=float)
    bin_confidences = np.zeros(num_bins, dtype=float)
    bin_counts = np.zeros(num_bins, dtype=np.int32)

    for b in range(num_bins):
        selected = np.where(indices == b + 1)[0]
        if len(selected) > 0:
            bin_accuracies[b] = np.mean(true_labels[selected] == pred_labels[selected])
            if bin_accuracies[b] < .2:
                pass
            bin_confidences[b] = np.mean(confidences[selected])
            bin_counts[b] = len(selected)

    avg_acc = np.sum(bin_accuracies * bin_counts) / np.sum(bin_counts)
     
    avg_conf = np.sum(bin_confidences * bin_counts) / np.sum(bin_counts)

    gaps = np.abs(bin_accuracies - bin_confidences)
    ece = np.sum(gaps * bin_counts) / np.sum(bin_counts)
    mce = np.max(gaps)

    return { "accuracies": bin_accuracies, 
             "confidences": bin_confidences, 
             "counts": bin_counts, 
             "bins": bins,
             "avg_accuracy": avg_acc,
             "avg_confidence": avg_conf,
             "expected_calibration_error": ece,
             "max_calibration_error": mce }


def _reliability_diagram_subplot(ax, bin_data, 
                                 draw_ece=True, 
                                 draw_bin_importance=False,
                                 title="Reliability Diagram", 
                                 xlabel="Confidence", 
                                 ylabel="Expected Accuracy"):
    """Draws a reliability diagram into a subplot."""
    accuracies = bin_data["accuracies"]
    confidences = bin_data["confidences"]
    counts = bin_data["counts"]
    bins = bin_data["bins"]

    bin_size = .5 / len(counts)
    positions = bins[:-1] + bin_size/2.0
    # widths = bin_size
    widths = np.mean(np.diff(positions))
    alphas = 0.3
    min_count = np.min(counts)
    max_count = np.max(counts)
    normalized_counts = (counts - min_count) / (max_count - min_count)

    if draw_bin_importance == "alpha":
        alphas = 0.2 + 0.8*normalized_counts
    elif draw_bin_importance == "width":
        widths = 0.1*np.diff(positions) + 0.9*np.diff(positions)*normalized_counts
    
    colors = np.zeros((len(counts), 4))
    colors[:, 0] = 240 / 255.
    colors[:, 1] = 60 / 255.
    colors[:, 2] = 60 / 255.
    colors[:, 3] = alphas

    gap_plt = ax.bar(positions, np.abs(accuracies - confidences), 
                     bottom=np.minimum(accuracies, confidences), width=widths,
                     edgecolor=colors, color=colors, linewidth=1, label="Gap")

    acc_plt = ax.bar(positions, 0, bottom=accuracies, width=widths,
                     edgecolor="black", color="black", alpha=1.0, linewidth=3,
                     label="Accuracy")

    confidence_plot = ax.bar(positions, 0, bottom=confidences, width=widths,
                     edgecolor="blue", color="blue", alpha=1.0, linewidth=3,
                     label="Confidence")
    
    ax.set_aspect("equal")
    ax.plot([0,1], [0,1], linestyle = "--", color="gray")
    
    if draw_ece:
        ece = (bin_data["expected_calibration_error"])
        ax.text(0.98, 0.02, "ECE=%.2f" % ece, color="black", 
                ha="right", va="bottom", transform=ax.transAxes, fontsize=14)

    ax.set_xlim(.5, 1)
    ax.set_ylim(0, 1)
    ax.set_xticks(bins)

    ax.set_title(title)
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)

    ax.legend(handles=[gap_plt, acc_plt, confidence_plot])
    return


def _confidence_histogram_subplot(ax, bin_data, 
                                  draw_averages=True,
                                  title="Examples per bin", 
                                  xlabel="Confidence",
                                  ylabel="Count"):
    """Draws a confidence histogram into a subplot."""
    counts = bin_data["counts"]
    bins = bin_data["bins"]

    bin_size = .5 / len(counts)
    positions = bins[:-1] + bin_size/2.0

    ax.bar(positions, counts, width=bin_size * 0.9)
   
    ax.set_xlim(.5, 1)
    ax.set_title(title)
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)

    if draw_averages:
        acc_plt = ax.axvline(x=bin_data["avg_accuracy"], ls="solid", lw=3, 
                             c="black", label="Accuracy")
        conf_plt = ax.axvline(x=bin_data["avg_confidence"], ls="dotted", lw=3, 
                              c="#444", label="Avg. confidence")
        ax.legend(handles=[acc_plt, conf_plt])
    return


def _reliability_diagram_combined(bin_data, 
                                  draw_ece, draw_bin_importance, draw_averages, 
                                  title, figsize, dpi, return_fig):
    """Draws a reliability diagram and confidence histogram using the output
    from compute_calibration()."""
    figsize = (figsize[0], figsize[0] * 1.4)

    fig, ax = plt.subplots(nrows=2, ncols=1, sharex=True, figsize=figsize, dpi=dpi, 
                           gridspec_kw={"height_ratios": [4, 1]})

    plt.tight_layout()
    plt.subplots_adjust(hspace=-0.1)

    _reliability_diagram_subplot(ax[0], bin_data, draw_ece, draw_bin_importance, 
                                 title=title, xlabel="")

    # Draw the confidence histogram upside down.
    orig_counts = bin_data["counts"]
    bin_data["counts"] = -bin_data["counts"]
    _confidence_histogram_subplot(ax[1], bin_data, draw_averages, title="")
    bin_data["counts"] = orig_counts

    # Also negate the ticks for the upside-down histogram.
    new_ticks = np.abs(ax[1].get_yticks()).astype(np.int32)  # Use np.int32
    ax[1].set_yticklabels(new_ticks)

    if return_fig: return fig



def reliability_diagram(true_labels, pred_labels, confidences, 
                        draw_ece=True, draw_bin_importance=False, 
                        draw_averages=True, title="Reliability Diagram", 
                        figsize=(6, 6), dpi=72, return_fig=False, model = None):
    """Draws a reliability diagram and confidence histogram in a single plot.
    
    First, the model's predictions are divided up into bins based on their
    confidence scores.

    The reliability diagram shows the gap between average accuracy and average 
    confidence in each bin. These are the red bars.

    The black line is the accuracy, the other end of the bar is the confidence.

    Ideally, there is no gap and the black line is on the dotted diagonal.
    In that case, the model is properly calibrated and we can interpret the
    confidence scores as probabilities.

    The confidence histogram visualizes how many examples are in each bin. 
    This is useful for judging how much each bin contributes to the calibration
    error.

    The confidence histogram also shows the overall accuracy and confidence. 
    The closer these two lines are together, the better the calibration.
    
    The ECE or Expected Calibration Error is a summary statistic that gives the
    difference in expectation between confidence and accuracy. In other words,
    it's a weighted average of the gaps across all bins. A lower ECE is better.

    Arguments:
        true_labels: the true labels for the test examples
        pred_labels: the predicted labels for the test examples
        confidences: the predicted confidences for the test examples
        num_bins: number of bins
        draw_ece: whether to include the Expected Calibration Error
        draw_bin_importance: whether to represent how much each bin contributes
            to the total accuracy: False, "alpha", "widths"
        draw_averages: whether to draw the overall accuracy and confidence in
            the confidence histogram
        title: optional title for the plot
        figsize: setting for matplotlib; height is ignored
        dpi: setting for matplotlib
        return_fig: if True, returns the matplotlib Figure object
    """
    bin_data = compute_calibration(true_labels, pred_labels, confidences, model=model)
    return _reliability_diagram_combined(bin_data, draw_ece, draw_bin_importance,
                                         draw_averages, title, figsize=figsize, 
                                         dpi=dpi, return_fig=return_fig)


def reliability_diagrams(results, num_bins=10,
                         draw_ece=True, draw_bin_importance=False, 
                         num_cols=4, dpi=72, return_fig=False):
    """Draws reliability diagrams for one or more models.
    
    Arguments:
        results: dictionary where the key is the model name and the value is
            a dictionary containing the true labels, predicated labels, and
            confidences for this model
        num_bins: number of bins
        draw_ece: whether to include the Expected Calibration Error
        draw_bin_importance: whether to represent how much each bin contributes
            to the total accuracy: False, "alpha", "widths"
        num_cols: how wide to make the plot
        dpi: setting for matplotlib
        return_fig: if True, returns the matplotlib Figure object
    """
    ncols = num_cols
    nrows = (len(results) + ncols - 1) // ncols
    figsize = (ncols * 4, nrows * 4)

    fig, ax = plt.subplots(nrows=nrows, ncols=ncols, sharex=True, sharey=True, 
                           figsize=figsize, dpi=dpi, constrained_layout=True)

    for i, (plot_name, data) in enumerate(results.items()):
        y_true = data["true_labels"]
        y_pred = data["pred_labels"]
        y_conf = data["confidences"]
        
        bin_data = compute_calibration(y_true, y_pred, y_conf, num_bins)
        
        row = i // ncols
        col = i % ncols
        _reliability_diagram_subplot(ax[row, col], bin_data, draw_ece, 
                                     draw_bin_importance, 
                                     title="\n".join(plot_name.split()),
                                     xlabel="Confidence" if row == nrows - 1 else "",
                                     ylabel="Expected Accuracy" if col == 0 else "")

    for i in range(i + 1, nrows * ncols):
        row = i // ncols
        col = i % ncols        
        ax[row, col].axis("off")
        
    if return_fig: return fig


def calculate_accuracies_and_confidences(y_true, y_pred_proba, M=10, bin_digitization ="uniform"):
    """
    Calculate accuracies and average confidences for M bins.

    Parameters:
    - y_true: array-like, true labels
    - y_pred_proba: array-like, predicted probabilities
    - M: int, number of bins (default: 10)
    - bin_digitization: str, method for digitizing bins (default: "uniform")

    Returns:
    - accs: array, accuracies for each bin
    - confs: array, average confidences for each bin
    - bin_average_confidence: array, average confidence for each bin

    Raises:
    - ValueError: if bin_digitization is not "uniform" or "quantile"
    """
    assert bin_digitization in ["uniform", "quantile"]
    """Calculate accuracies and average confidences for M bins."""
    # Calculate the bin edges and indices for each prediction
    if bin_digitization == "uniform":
        bins = np.linspace(.5, 1, M + 1)
    elif bin_digitization == "quantile":
        bins = np.quantile(y_pred_proba, np.linspace(0, 1, M + 1))
    else:
        raise ValueError("Invalid bin digitization method. Please use 'uniform' or 'quantile'.")
    bin_average_confidence = bins[0:len(bins)-1]+np.diff(bins)/2
    # bin_indices = np.digitize(y_pred_proba, bins) - 1
    bin_indices = np.digitize(y_pred_proba, bins) - 1
    assert min(y_pred_proba) >=.5
    accs = np.zeros(M)
    confs = np.zeros(M)

    for i in range(M):
        # Select data for the current bin
        in_bin = bin_indices == i
        if np.sum(in_bin) > 0:
            # Calculate accuracy as the fraction of correct predictions
            accs[i] = np.mean(y_true[in_bin])
            # Calculate the average confidence (predicted probability)
            confs[i] = np.mean(y_pred_proba[in_bin])
        else:
            accs[i] = None
            confs[i] = None

    return accs, confs, bin_average_confidence

def rel_diagram_sub(accs, confs, bin_average_confidence, ax, ECE, name = "Reliability Diagram", xname = "", yname=""):
    """
    Plot a reliability diagram.

    Parameters:
    - accs (list): List of empirical accuracies for each bin.
    - confs (list): List of output confidences for each bin.
    - bin_average_confidence (list): List of average confidences for each bin.
    - ax (matplotlib.axes.Axes): The axes on which to plot the diagram.
    - ECE (float): Expected Calibration Error.
    - name (str, optional): Title of the diagram. Defaults to "Reliability Diagram".
    - xname (str, optional): Label for the x-axis. Defaults to an empty string.
    - yname (str, optional): Label for the y-axis. Defaults to an empty string.

    Returns:
    None
    """    
    bin_size = bin_average_confidence[1] - bin_average_confidence[0]
    positions = bin_average_confidence
    assert len(accs) == len(confs)
    gap_plt = ax.bar(positions, accs, width = bin_size, edgecolor = "black", color = "blue", label="Empirical Accuracy", linewidth=2, zorder=4)
    output_plt = ax.bar(positions, confs, width = bin_size, edgecolor = "black", color = "red", label="Output Confidence", zorder = 3)
    ax.plot([.5, 1], [0.5, 1], color='red', linestyle=':', linewidth=2, zorder=5)
    ece_text = f"ECE: {ECE:.2%}"  # Format the ECE as a percentage with two decimal places
    ax.text(0.95, 0.05, ece_text, fontsize=12, verticalalignment='bottom', horizontalalignment='right', transform=ax.transAxes, zorder=6, bbox=dict(facecolor='white', edgecolor='black', boxstyle='round,pad=0.5'))

    ax.set_aspect('equal', adjustable='box')  # Uncomment this if you want to keep the aspect ratio
    ax.legend(handles = [gap_plt, output_plt])
    ax.set_xlim(.5,1)  # Set x-axis limits
    ax.set_ylim(0,1)   # Set y-axis limits
    ax.set_title(name, fontsize=12, pad=10)
    if xname != "":
        ax.set_xlabel(xname, fontsize=14, color = "black")
    if yname != "":
        ax.set_ylabel(yname, fontsize=14, color = "black")
    
    ax.set_yticks(np.arange(0, 1.1, 0.1))
    xtick_vals = [0.6, 0.8, 1.0]
    ax.set_xticks(xtick_vals)
    # ax.set_xticklabels([f"{tick:.1f}" for tick in np.arange(.5, 1.1, 0.1)], fontsize=10)
    formatted = ["0.6", "0.8", "1.0"]
    ax.set_xticklabels(formatted, fontsize=10)
    # ax.axvline(xmin=0, x_max =1, ymin=0, ymax=1, color='red', linestyle=':', linewidth=2, zorder=5)
    return

def split_data_pred_pos_pred_neg(true_labels, pred_probs, threshold=0.5):
    # Adjust probabilities for negative predictions
    adjusted_probs = np.where(pred_probs < threshold, 1-pred_probs, pred_probs)
    positive_predictions = pred_probs >= threshold
    negative_predictions = ~positive_predictions

    # Split data based on prediction type
    pos_true_labels = true_labels[positive_predictions]
    neg_true_labels = true_labels[negative_predictions]
    pos_pred_probs = adjusted_probs[positive_predictions]
    neg_pred_probs = adjusted_probs[negative_predictions]

    return pos_true_labels, neg_true_labels, pos_pred_probs, neg_pred_probs
