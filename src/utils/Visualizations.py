from src.utils.Calibrator import Calibrator, split_data_pred_pos_pred_neg, calculate_accuracies_and_confidences, rel_diagram_sub
import matplotlib.pyplot as plt
import pickle
import argparse

def plot_reliability_diagram(stats_for_reliability_diagrams):
    """
    Plots reliability diagrams for the given statistics.

    Parameters:
    - stats_for_reliability_diagrams (dict): A dictionary containing statistics for reliability diagrams.

    Returns:
    - None
    """

    # Create 4 subplots
    fig, axs = plt.subplots(2, 2, figsize=(10, 10))   # Adjusted for 4 plots
    model_name = stats_for_reliability_diagrams.pop('model_name', None)
    model_name = model_name.split(" object")[0].split(".")[-1]
    model_name = model_name.rstrip("SKLearnModel")

    for i, (key, data) in enumerate(stats_for_reliability_diagrams.items()):
        # Split data based on positive or negative prediction
        pos_true_labels, neg_true_labels, pos_pred_probs, neg_pred_probs = split_data_pred_pos_pred_neg(data['true_labels'], data['y_pred_proba'])
        M = 8
        BIN_DIGITIZATION = "uniform" # "uniform" # "uniform" or "quantile"
        # print(neg_pred_probs)
        accs_pos, confs_pos, bin_average_confidence_positive = calculate_accuracies_and_confidences(pos_true_labels, pos_pred_probs, M=M, bin_digitization=BIN_DIGITIZATION)
        accs_neg, confs_neg, bin_average_confidence_negative = calculate_accuracies_and_confidences(1-neg_true_labels, neg_pred_probs, M=M, bin_digitization=BIN_DIGITIZATION)

        ECE_pos = Calibrator.calculate_calibration_errors(pos_true_labels, pos_pred_probs, M, binning_strategy='uniform', min_probability=0.5)[0]
        ECE_neg = Calibrator.calculate_calibration_errors(1-neg_true_labels, neg_pred_probs, M, binning_strategy='uniform', min_probability=0.5)[0]

        # For positive predictions
        calibration_model_status = 'Uncalibrated\n' if key == 'original' else 'Calibrated\n with Platt Scaling'
        calibration_model_name_pos = f"{model_name} {calibration_model_status}Positive Predictions"

        rel_diagram_sub(accs_pos, confs_pos, bin_average_confidence_positive, axs[i, 0], ECE = ECE_pos, name=calibration_model_name_pos, xname="Confidence", yname="Accuracy")

        # Plot for negative predictions
        calibration_model_name_neg = f"{model_name} {calibration_model_status}Negative Predictions"
        rel_diagram_sub(accs_neg, confs_neg, bin_average_confidence_negative, axs[i, 1],  ECE = ECE_neg, name=calibration_model_name_neg, xname="Confidence", yname="Accuracy")

        pass

    plt.tight_layout()
    plt.show()
    return


'''
Example Reliability Diagram:\nApplying Platt Scaling to BernouliNB Ensemble Model
'''
def plot_reliability_diagram_one_line(stats_for_reliability_diagrams):
    """
    Plots a reliability diagram with multiple subplots.

    Args:
        stats_for_reliability_diagrams (dict): A dictionary containing statistics for reliability diagrams.

    Returns:
        None
    """

    # Create 4 subplots in one row
    fig, axs = plt.subplots(1, 4, figsize=(12, 5))

    model_name = stats_for_reliability_diagrams.pop('model_name', None)
    model_name = model_name.split(" object")[0].split(".")[-1]
    model_name = model_name.rstrip("SKLearnModel")

    for i, (key, data) in enumerate(stats_for_reliability_diagrams.items()):
        pos_true_labels, neg_true_labels, pos_pred_probs, neg_pred_probs = split_data_pred_pos_pred_neg(data['true_labels'], data['y_pred_proba'])
        M = 8
        BIN_DIGITIZATION = "uniform"  # "uniform" or "quantile"

        accs_pos, confs_pos, bin_average_confidence_positive = calculate_accuracies_and_confidences(pos_true_labels, pos_pred_probs, M=M, bin_digitization=BIN_DIGITIZATION)
        accs_neg, confs_neg, bin_average_confidence_negative = calculate_accuracies_and_confidences(1-neg_true_labels, neg_pred_probs, M=M, bin_digitization=BIN_DIGITIZATION)

        ECE_pos = Calibrator.calculate_calibration_errors(pos_true_labels, pos_pred_probs, M, binning_strategy='uniform', min_probability=0.5)[0]
        ECE_neg = Calibrator.calculate_calibration_errors(1-neg_true_labels, neg_pred_probs, M, binning_strategy='uniform', min_probability=0.5)[0]

        # For positive predictions
        calibration_model_status = 'Uncalibrated\n' if key == 'original' else 'Calibrated\n'
        calibration_model_name_pos = f"{calibration_model_status}Positive Predictions"
        rel_diagram_sub(accs_pos, confs_pos, bin_average_confidence_positive, axs[i*2], ECE=ECE_pos, name=calibration_model_name_pos, xname="Confidence", yname="")

        # Plot for negative predictions
        calibration_model_name_neg = f"{calibration_model_status}Negative Predictions"
        rel_diagram_sub(accs_neg, confs_neg, bin_average_confidence_negative, axs[i*2+1], ECE=ECE_neg, name=calibration_model_name_neg, xname="Confidence", yname="")


    fig.text(0.02, 0.5, 'Accuracy     ', va='center', rotation='vertical', fontsize=15)
    fig.suptitle('Example Reliability Diagram:\nApplying Platt Scaling to BernouliNB Ensemble Model', fontsize=18)
    for ax in axs:
        ax.legend_.remove()

    # Set common legend
    handles, labels = axs[0].get_legend_handles_labels()
    unique_labels = ['Empirical Accuracy', 'Output Confidence']
    unique_handles = [handles[labels.index(label)] for label in unique_labels]
    fig.legend(unique_handles, unique_labels, loc='lower center', ncol=2, fontsize=15,  bbox_to_anchor=(.5, 0))
    fig.subplots_adjust(hspace=0.3, wspace=0.009)
    fig.subplots_adjust(bottom=0.72)
    plt.tight_layout(rect=[0, .1, 0, 0.2])
    plt.show()
    return

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--input_file", help="Path to the input file", default = "data\generated\stats_for_reliability_diagrams\stats_for_reliabilitiy_diagrams_2024-02-12-18-19-15.pk", required=True)
    args = parser.parse_args()

    with open(args.input_file, "rb") as input_file:
        stats_for_reliability_diagrams = pickle.load(input_file)
    
    plot_reliability_diagram_one_line(stats_for_reliability_diagrams)
    plot_reliability_diagram(stats_for_reliability_diagrams)
