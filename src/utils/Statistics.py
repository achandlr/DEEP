from matplotlib import pyplot as plt
import numpy as np
import pandas as pd
from os.path import join
# from sklearn.metrics import calibration_curve
# from sklearn.calibration import calibration_curve
# import pickle
from scipy import stats
from sklearn.cluster import KMeans
# from sklearn.metrics import accuracy_score

from scipy.stats import bootstrap
import pickle

from sklearn.metrics import f1_score, accuracy_score, roc_auc_score, balanced_accuracy_score
from sklearn.utils import resample
from scipy.stats import ttest_ind
from scipy.stats import permutation_test

def permutation_test_p_value(obs1, obs2, n_permutations=10000):
    """
    Perform a permutation test to calculate the p-value for the difference in means.
    
    Parameters:
    - obs1, obs2 (np.array): Observations from the two groups.
    - n_permutations (int): Number of permutations.
    
    Returns:
    - float: p-value from the permutation test.
    """
    # Calculate the observed difference in means
    observed_diff = np.mean(obs1) - np.mean(obs2)
    
    # Combine all observations
    combined_obs = np.concatenate([obs1, obs2])
    
    # Count how many times the permuted differences are greater than or equal to the observed difference
    count = 0
    for _ in range(n_permutations):
        np.random.shuffle(combined_obs)
        permuted_diff = np.mean(combined_obs[:len(obs1)]) - np.mean(combined_obs[len(obs1):])
        if abs(permuted_diff) >= abs(observed_diff):
            count += 1
    
    # Calculate the p-value
    p_value = count / n_permutations
    return p_value
# Define a function to perform bootstrap resampling and calculate the difference in performance
def bootstrap_resampling_and_difference(obs1, obs2, n_bootstrap=10000):
    """
    Perform bootstrap resampling on two sets of observations and calculate differences.
    
    Parameters:
    - obs1 (np.array): Predictions from the first model.
    - obs2 (np.array): Predictions from the second model.
    - n_bootstrap (int): Number of bootstrap samples to generate.
    
    Returns:
    - np.array: Differences in performance for each bootstrap sample.
    """
    bootstrapped_differences = []
    for _ in range(n_bootstrap):
        # Generate bootstrap samples for each set of observations
        boot_obs1 = resample(obs1)
        boot_obs2 = resample(obs2)
        
        # Calculate performance metric for each bootstrap sample
        # Here, we assume it's accuracy, but it could be replaced with any other metric
        performance_obs1 = np.mean(boot_obs1)
        performance_obs2 = np.mean(boot_obs2)
        
        # Store the difference in performance
        bootstrapped_differences.append(performance_obs1 - performance_obs2)
        
    return np.array(bootstrapped_differences)

# Define a function to calculate p-values and confidence intervals
def statistical_significance(bootstrapped_differences, alpha=0.05):
    """
    Calculate p-values and confidence intervals from bootstrapped differences.
    
    Parameters:
    - bootstrapped_differences (np.array): Differences in performance from bootstrap resampling.
    - alpha (float): Significance level for calculating confidence intervals.
    
    Returns:
    - float: p-value from t-test.
    - tuple: Lower and upper bounds of the confidence interval.
    """
    # Perform two-sample t-test
    t_stat, p_value = ttest_ind(bootstrapped_differences, np.zeros_like(bootstrapped_differences))
    
    # Calculate confidence intervals
    lower_bound = np.percentile(bootstrapped_differences, (alpha/2)*100)
    upper_bound = np.percentile(bootstrapped_differences, (1-alpha/2)*100)
    
    return p_value, (lower_bound, upper_bound)

def find_best_threshold(scores, binary_labels, metric='balanced_accuracy'):
    best_threshold = 0.0
    best_score = 0.0
    
    # Iterate over potential threshold values
    for threshold in np.linspace(0, 1, 1000):
        # Convert scores to binary predictions based on current threshold
        binary_predictions = [1 if score >= threshold else 0 for score in scores]
        
        # Calculate metric at current threshold
        if metric == 'f1':
            current_score = f1_score(binary_labels, binary_predictions)
        elif metric == 'accuracy':
            current_score = accuracy_score(binary_labels, binary_predictions)
        elif metric == 'roc_auc':
            raise "Implementation Error"
            current_score = roc_auc_score(binary_labels, scores)
        elif metric == 'balanced_accuracy':
            current_score = balanced_accuracy_score(binary_labels, binary_predictions)
        else:
            raise ValueError("Invalid metric. Choose either 'f1', 'accuracy', 'roc_auc', or 'balanced_accuracy'.")
        
        # Update best threshold and score if this threshold is better
        if current_score > best_score:
            best_score = current_score
            best_threshold = threshold
            
    return best_threshold, best_score


def find_best_accuracy_subset(pred_probs, ground_truth, min_needed_samples):
    # Combine probabilities with ground truth and sort by probability descending
    combined = sorted(zip(pred_probs, ground_truth), key=lambda x: x[0], reverse=True)
    
    # Function to calculate accuracy
    def calculate_accuracy(subset):
        correct = sum(1 for prob, truth in subset if (prob > 0.5) == truth)
        return correct / len(subset)
    
    # Initialize best accuracy and its corresponding subset size
    best_acc = 0.0
    for i in range(min_needed_samples, len(combined) + 1):
        # Consider current subset from the start to the ith element
        current_subset = combined[:i]
        current_acc = calculate_accuracy(current_subset)
        if current_acc > best_acc:
            best_acc = current_acc
            final_size = i

    # Return the final subset size and the best accuracy found
    return final_size, best_acc

def cluster_std(probabilities, num_clusters=2, visualize=False):
    # Reshape probabilities to a 2D array for scikit-learn
    probs_reshaped = probabilities.reshape(-1, 1)
    
    # Perform k-means clustering
    kmeans = KMeans(n_clusters=num_clusters, random_state=0).fit(probs_reshaped)
    labels = kmeans.labels_
    
    # Calculate variance within each cluster
    stds = []
    cluster_sizes = []
    for i in range(num_clusters):
        cluster_probs = probabilities[labels == i]
        # print(f"cluster_probs: {cluster_probs}")
        cluster_sizes.append(len(cluster_probs))
        stds.append(np.std(cluster_probs))
    
    # Calculate the weighted average variance
    weighted_avg_std = np.average(stds, weights=cluster_sizes)
    
    if visualize:
        plt.scatter(probabilities, np.zeros_like(probabilities), c=labels, cmap='viridis', marker='o')
        plt.scatter(kmeans.cluster_centers_, np.zeros_like(kmeans.cluster_centers_), c='red', marker='x')
        plt.title('Cluster Visualization')
        plt.xlabel('Predicted Probabilities')
        plt.yticks([])
        plt.show()
    
    return weighted_avg_std

def check_continuous_subset(full_labels, subset_labels):
    """
    Check if there is a continuous set of values within full_labels that matches subset_labels.
    
    Parameters:
    - full_labels (np.array): Full set of labels, potentially containing the subset.
    - subset_labels (np.array): The subset of labels to be checked for continuity in full_labels.
    
    Returns:
    - bool: True if there is a continuous subset, False otherwise.
    - int: Starting index of the continuous subset if found, -1 otherwise.
    """
    # Get the length of the subset
    subset_length = len(subset_labels)
    
    # Iterate over the full_labels to find a matching continuous subset
    for start_idx in range(len(full_labels) - subset_length + 1):
        # Slice the full_labels to get a subset of the same length as subset_labels
        full_subset = full_labels[start_idx:start_idx + subset_length]
        
        # Check if the sliced subset from full_labels matches the subset_labels
        if np.array_equal(full_subset, subset_labels):
            return True, start_idx  # Return True with the starting index
    
    return False, -1  # Return False with -1 if no match is found
