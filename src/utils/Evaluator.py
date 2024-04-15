from sklearn.metrics import accuracy_score, balanced_accuracy_score, f1_score, confusion_matrix, recall_score, precision_score
from src.utils.Calibrator import Calibrator
from sklearn.metrics import accuracy_score, balanced_accuracy_score, f1_score, recall_score, precision_score, confusion_matrix
from scipy.stats import sem, t
from sklearn.utils import resample
import numpy as np
from statsmodels.stats.proportion import proportion_confint

class Evaluator:
    @staticmethod
    def calculate_confidence_interval(y_true, y_pred, confidence=0.95, n_iterations=1000, seed=None):
        """
        Calculate the confidence interval for the accuracy of predictions and return a dictionary.

        Parameters:
        - y_true: array-like, true labels.
        - y_pred: array-like, predicted labels.
        - confidence: float, the confidence level for the interval.
        - n_iterations: int, number of bootstrap samples to use.
        - seed: int, seed for random number generator for reproducibility.

        Returns:
        - results: dict, containing the mean accuracy, confidence interval, margin of error (as a percentage),
                lower bound, upper bound, confidence level, and number of iterations.
        """

        # Check if input arrays are of the same length
        if len(y_true) != len(y_pred):
            raise ValueError("The length of y_true and y_pred must be the same.")

        # Initialize the random number generator with the provided seed for reproducibility
        np.random.seed(seed)

        # Calculate initial accuracy
        initial_accuracy = accuracy_score(y_true, y_pred)
        
        # Number of successess
        successes = int(initial_accuracy * len(y_true))

        # Calculate confidence interval using the statsmodels library
        lower_bound, upper_bound = proportion_confint(successes, len(y_true), alpha=1-confidence, method='binom_test')

        # Margin of error as a percentage
        margin_error = (upper_bound - lower_bound) / 2 * 100

        # Mean accuracy as a percentage
        mean_accuracy = initial_accuracy * 100
        
        # Lower and upper bounds as a percentage
        lower_bound *= 100
        upper_bound *= 100

        # Return the results as a dictionary
        results = {
            'mean_accuracy': mean_accuracy,
            'confidence_interval': (lower_bound, upper_bound),
            'margin_of_error_percent': margin_error,
            'lower_bound_percent': lower_bound,
            'upper_bound_percent': upper_bound,
            'confidence': confidence,
            'n_iterations': n_iterations
        }

        return results

    @staticmethod
    def find_highest_balanced_accuracy(calibration_results):
        """
        Finds the combination of main key and sub key that has the highest balanced accuracy.

        Args:
            calibration_results (dict): A dictionary containing calibration results.

        Returns:
            tuple: A tuple containing the best combination of main key and sub key, and the corresponding highest balanced accuracy.

        """
        max_balanced_accuracy = float('-inf')
        best_combination = None
        
        # Loop through the main keys (e.g., 'using_val', 'using_val_with_cv', etc.)
        for main_key in calibration_results.keys():
            # Loop through the sub-keys (e.g., 'original', 'platt_scaling', etc.)
            for sub_key in calibration_results[main_key].keys():
                # Check if 'balanced_accuracy' is in the dictionary for this combination
                if isinstance(calibration_results[main_key][sub_key], dict) and 'balanced_accuracy' in calibration_results[main_key][sub_key]:
                    # Get the balanced accuracy for this combination
                    balanced_accuracy = calibration_results[main_key][sub_key]['balanced_accuracy']
                    # Update the max_balanced_accuracy and best_combination if this is the highest so far
                    if balanced_accuracy and balanced_accuracy > max_balanced_accuracy:
                        max_balanced_accuracy = balanced_accuracy
                        best_combination = (main_key, sub_key)
        
        return best_combination, max_balanced_accuracy

    @staticmethod
    def find_lowest_ece(calibration_results):
        lowest_ece = float('inf')
        comparable_original_ece = float('inf')
        best_combination = None
        
        # Loop through the main keys (e.g., 'using_val', 'using_val_with_cv', etc.)
        for main_key in calibration_results.keys():
            # Loop through the sub-keys (e.g., 'original', 'platt_scaling', etc.)
            for sub_key in calibration_results[main_key].keys():
                # Check if 'balanced_accuracy' is in the dictionary for this combination
                if isinstance(calibration_results[main_key][sub_key], dict) and 'ECE' in calibration_results[main_key][sub_key]:
                    # Get the balanced accuracy for this combination
                    ece = calibration_results[main_key][sub_key]['ECE']
                    # Update the max_balanced_accuracy and best_combination if this is the highest so far
                    if ece and ece < lowest_ece:
                        lowest_ece = ece
                        comparable_original_ece = calibration_results[main_key]['original']['ECE']
                        best_combination = (main_key, sub_key)
        
        return best_combination, lowest_ece, comparable_original_ece
    
    @staticmethod
    def get_stats(model, X_train, Y_train, X_test, Y_test, X_val, y_val, plot_calibration_curves, perform_calibration = True):
        """
        Calculate various evaluation metrics for a given model.

        Args:
            model: The trained model.
            X_train: The input features of the training set.
            Y_train: The target labels of the training set.
            X_test: The input features of the test set.
            Y_test: The target labels of the test set.
            X_val: The input features of the validation set.
            y_val: The target labels of the validation set.

        Returns:
            A dictionary containing the evaluation metrics:
            - accuracy: The accuracy score.
            - balanced_accuracy: The balanced accuracy score.
            - f1_score: The F1 score.
            - sensitivity: The sensitivity (recall) score.
            - specificity: The specificity score.
            - precision: The precision score.
            - TP_count: The count of true positives.
            - TN_count: The count of true negatives.
            - FP_count: The count of false positives.
            - FN_count: The count of false negatives.
            - y_pred: The predicted labels.
            - y_true: The true labels.
            - brier_score: The Brier score (if model supports predict_proba), otherwise None.
        """
        Y_pred = model.predict(X_test)
        if hasattr(model, 'predict_proba'):
            try:
                y_pred_proba = model.predict_proba(X_test)
            except Exception as e:
                y_pred_proba = None
        else:
            y_pred_proba = None
        
        accuracy, balanced_accuracy, f1, sensitivity, precision, tn, fp, fn, tp, specificity = [None] * 10

        # Calculate basic metrics with exception handling
        try:
            accuracy = accuracy_score(Y_test, Y_pred)
        except Exception as e:
            accuracy = None

        try:
            balanced_accuracy = balanced_accuracy_score(Y_test, Y_pred)
        except Exception as e:
            balanced_accuracy = None

        try:
            f1 = f1_score(Y_test, Y_pred)
        except Exception as e:
            f1 = None

        try:
            sensitivity = recall_score(Y_test, Y_pred)
        except Exception as e:
            sensitivity = None

        try:
            precision = precision_score(Y_test, Y_pred, zero_division=0)
        except Exception as e:
            precision = None

        try:
            tn, fp, fn, tp = confusion_matrix(Y_test, Y_pred).ravel()
            specificity = tn / (tn + fp) if (tn + fp) != 0 else 0
        except Exception as e:
            specificity = None
            tn, fp, fn, tp = None, None, None, None
        try:
            confidence_interval_data = Evaluator.calculate_confidence_interval(Y_test, Y_pred, confidence=0.95,)
        except Exception as e:
            confidence_interval_data = None
        # Initialize the results dictionary
        results = {
            'accuracy': accuracy,
            'balanced_accuracy': balanced_accuracy,
            'f1_score': f1,
            'sensitivity': sensitivity,
            'specificity': specificity,
            'precision': precision,
            'TP_count': tp,
            'TN_count': tn,
            'FP_count': fp,
            'FN_count': fn,
            "y_pred": Y_pred,
            "y_true": Y_test,
            "y_pred_proba": y_pred_proba,
            "calibration_results": {},
            "best_calibration_balanced_accuracy_combination": None,
            "best_calibration_balanced_accuracy": None,
            "lowest_ece_calibration_combination": None,
            "lowest_ece": None,
            "comparable_original_ece": None,
            "confidence_interval_data": confidence_interval_data

        }

        if perform_calibration and hasattr(model, 'predict_proba'):
            calibration_results = Evaluator.perform_calibration(model, X_train, Y_train, X_test, Y_test, X_val, y_val, plot_calibration_curves)
            results["calibration_results"] = calibration_results
            best_calibration_combination, best_calibration_balanced_accuracy = Evaluator.find_highest_balanced_accuracy(calibration_results)
            results["best_calibration_balanced_accuracy_combination"] = best_calibration_combination
            results["best_calibration_balanced_accuracy"] = best_calibration_balanced_accuracy

            lowest_ece_calibration_combination, lowest_ece, comparable_original_ece= Evaluator.find_lowest_ece(calibration_results)
            results["lowest_ece_calibration_combination"] = lowest_ece_calibration_combination
            results["lowest_ece"] = lowest_ece
            results["comparable_original_ece"] = comparable_original_ece
            pass
        else:
            pass
        return results
    
    @staticmethod
    def meets_conditions_for_reliability_diagram(calibration_results_in_use):
        MIN_DESIRED_ECE = .05 
        # DESIRED_LOWEST_ECE_METHOD = "platt_scaling"
        MIN_NEEDED_CALIBRATION_DIFFERENCE = .10
        minimum_cluster_std = .05
        # minimum_max_calibrated_platt_acc = .84
        minimum_needed_best_platt_subset_acc = .81 
        # Add Brier score and calibration if model supports predict_proba
        MINIMUM_ALLOWED_MAX_PROBABILITY = .83
        MAXIMUM_ALLOWED_MIN_CONFIDENCE= .6

        if calibration_results_in_use['platt_scaling']['ECE'] == None:
            return False
        if calibration_results_in_use['original']['ECE'] == None:
            return False
        found_calibration_difference = calibration_results_in_use['original']['ECE'] - calibration_results_in_use['platt_scaling']['ECE']
        if found_calibration_difference < MIN_NEEDED_CALIBRATION_DIFFERENCE:
            return False
        if calibration_results_in_use['platt_scaling']['ECE'] > MIN_DESIRED_ECE:
            return False
        if calibration_results_in_use["platt_scaling"]["cluster_std"] < minimum_cluster_std:
            return False
        if calibration_results_in_use["platt_scaling"]["pred_proba_best_possible_acc"] < minimum_needed_best_platt_subset_acc:
            return False
        platt_probas = calibration_results_in_use["platt_scaling"]["pred_proba"]
        if platt_probas is None:
            return False
        platt_confidences = [1 - p if p < 0.5 else p for p in platt_probas]
        if max(platt_probas) < MINIMUM_ALLOWED_MAX_PROBABILITY:
            return False
        if min(platt_confidences) > MAXIMUM_ALLOWED_MIN_CONFIDENCE:
            return False

        return True
        

    @staticmethod
    def perform_calibration(model, X_train, Y_train, X_test, Y_test, X_val, y_val, plot_calibration_curves):

        calibration_results = {}
        if X_val is not None and y_val is not None:
            calibrator = Calibrator(model, X_val, y_val, X_test, Y_test)
            calibrator.fit_calibrators()
            calibration_results_using_val = calibrator.evaluate_calibrators()
            if Evaluator.meets_conditions_for_reliability_diagram(calibration_results_using_val):
                calibrator.plot_reliability_diagrams()
            for calibrator_model_name, calibrated_model in calibrator.calibrated_models.items():
                try:
                    calibrated_model_stats = Evaluator.get_stats(calibrated_model, X_train, Y_train, X_test, Y_test, X_val, y_val, plot_calibration_curves = False, perform_calibration = False)
                    calibration_results_using_val[calibrator_model_name].update(calibrated_model_stats)
                except Exception as e:
                    pass
            calibration_results["using_val"] = calibration_results_using_val
            calibrator_using_val_with_cv = Calibrator(model, X_train, Y_train, X_test, Y_test, cv = 3)
            calibrator_using_val_with_cv.fit_calibrators()
            calibration_results_using_val_with_cv = calibrator_using_val_with_cv.evaluate_calibrators()
            if Evaluator.meets_conditions_for_reliability_diagram(calibration_results_using_val_with_cv):
                calibrator.plot_reliability_diagrams()
            
            for calibrator_model_name, calibrated_model in calibrator.calibrated_models.items():
                try:
                    calibrated_model_stats = Evaluator.get_stats(calibrated_model, X_train, Y_train, X_test, Y_test, X_val, y_val, plot_calibration_curves = False, perform_calibration = False)
                    calibration_results_using_val_with_cv[calibrator_model_name].update(calibrated_model_stats)
                except Exception as e:
                    pass
            calibration_results["using_val_with_cv"] = calibration_results_using_val_with_cv

        calibrator_using_train = Calibrator(model, X_train, Y_train, X_test, Y_test)
        calibrator_using_train.fit_calibrators()
        calibration_results_using_train = calibrator_using_train.evaluate_calibrators()
        calibration_results["using_train"] = calibration_results_using_train

        if Evaluator.meets_conditions_for_reliability_diagram(calibration_results_using_train):
                calibrator.plot_reliability_diagrams()
                
        for calibrator_model_name, calibrated_model in calibrator.calibrated_models.items():
            try:
                calibrated_model_stats = Evaluator.get_stats(calibrated_model, X_train, Y_train, X_test, Y_test, X_val, y_val, plot_calibration_curves = False, perform_calibration = False)
                calibration_results_using_train[calibrator_model_name].update(calibrated_model_stats)
            except Exception as e:
                pass
        calibration_results["using_train"] = calibration_results_using_train

        calibrator_using_train_with_cv = Calibrator(model, X_train, Y_train, X_test, Y_test, cv = 3)
        calibrator_using_train_with_cv.fit_calibrators()
        calibration_results_using_train_with_cv = calibrator_using_train_with_cv.evaluate_calibrators()

        calibration_results["using_train_with_cv"] = calibration_results_using_train_with_cv



        if Evaluator.meets_conditions_for_reliability_diagram(calibration_results_using_train_with_cv):
            calibrator.plot_reliability_diagrams()

        for calibrator_model_name, calibrated_model in calibrator.calibrated_models.items():
            try:
                calibrated_model_stats = Evaluator.get_stats(calibrated_model, X_train, Y_train, X_test, Y_test, X_val, y_val, plot_calibration_curves = False, perform_calibration = False)
                calibration_results_using_train_with_cv[calibrator_model_name].update(calibrated_model_stats)
            except Exception as e:
                pass
        calibration_results["using_train_with_cv"] = calibration_results_using_train_with_cv

        return calibration_results
    
    @staticmethod
    def get_desired_val(calibration_results, metric_path, des_val = 'ECE'):
        # Navigate through the nested dictionary safely and return the value or np.inf if not found
        ece = []
        for key in metric_path:
            try:
                ece_val = calibration_results[key][des_val] 
                ece.append(ece_val)
            except:
                pass
        return ece


    @staticmethod
    def extract_calibration_results(row, calibration_key = "min_train_val", des_val = 'ECE'):
        assert calibration_key in ['min_train_val', 'using_train', 'using_val'], f"Invalid calibration_key: {calibration_key}"
        calibration_results = row.get('calibration_results', {})
        using_train = calibration_results.get('using_train', {})
        using_val = calibration_results.get('using_val', {})

        # Define paths to the metrics
        paths = [
            (['original'], 'Uncalibrated'),
            (['platt_scaling'], 'Platt Scaling'),
            (['isotonic_regression'], 'Isotonic Regression'),
            (['bbq_dynamic_20', 'bbq_dynamic_10'], 'BBQ'),
            (['hist_unform_20', 'hist_unform_10'], 'Hist Binning')
        ]

        metrics = {}
        for path, name in paths:
            if name == "Uncalibrated":
                pass
            if row["TestOrigin_Tuple"] == ('HALU_EVAL_SUMMARIZATION',):
                pass # This line is for debugging purposes
            if len(path) == 1:
                train_values = Evaluator.get_desired_val(using_train, path, des_val)
                val_values = Evaluator.get_desired_val(using_val, path, des_val)
            else:
                train_values = [Evaluator.get_desired_val(using_train, path_element, des_val) for path_element in path]
                val_values = [Evaluator.get_desired_val(using_val, path_element, des_val) for path_element in path]
            if name in ['BBQ', 'Hist Binning']:
                new_train_values = []
                new_val_values = []
                for train_val in train_values:
                    if len(train_val) > 0:
                        new_train_values.append(min(train_val))
                for val_value in val_values:
                    if len(val_value) > 0:
                        new_val_values.append(min(val_value))
                train_values = new_train_values
                val_values = new_val_values


            if calibration_key == 'min_train_val':
                if len(train_values) >= 1 and len(val_values) >=1:
                    min_train = min(train_values)
                    min_val = min(val_values)
                    if min_train == None and min_val == None:
                        metrics[name] = None
                    elif min_train == None:
                        metrics[name] = min_val
                    elif min_val == None:
                        metrics[name] = min_train
                    else:
                        metrics[name] = min(min_train, min_val)
                elif len(train_values) >= 1:
                    metrics[name] =  min(train_values)
                elif len(val_values) >= 1:
                    metrics[name] =  min(val_values)
                else:
                    metrics[name] = None

            elif calibration_key == 'using_train':
                if len(train_values) >= 1:
                    metrics[name] =  min(train_values)
                else:
                    metrics[name] = None
            elif calibration_key == 'using_val':
                if len(val_values) >= 1:
                    metrics[name] =  min(val_values)
                else:
                    metrics[name] = None
            else:
                raise ValueError(f"Invalid calibration_key: {calibration_key}")
 
        return metrics
