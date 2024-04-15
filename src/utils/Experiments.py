from enum import Enum, auto
from typing import List, Union, Optional, Set
import yaml

class DatasetOrigin(Enum):
    AGGREFACCT_SOTA_XSUM_TEST = "AGGREFACCT_SOTA_XSUM_TEST" 
    AGGREFACCT_SOTA_XSUM_VAL = "AGGREFACCT_SOTA_XSUM_VAL"
    AGGREFACCT_SOTA_CNN_DM_TEST = "AGGREFACCT_SOTA_CNN_DM_TEST"
    AGGREFACCT_SOTA_CNN_DM_VAL = "AGGREFACCT_SOTA_CNN_DM_VAL"
    HALU_EVAL_SUMMARIZATION = "HALU_EVAL_SUMMARIZATION"
    TOFU_EVAL_MEETING_BANK_SUMMARIZATION = "TOFU_EVAL_MEETING_BANK_SUMMARIZATION"
    TOFU_EVAL_MEDIA_SUM_SUMMARIZATION = "TOFU_EVAL_MEDIA_SUM_SUMMARIZATION"

class DataTypeOrigin(Enum):
    Confident_Train     = "Confident_Train"  
    Unsure_Train        = "Unsure_Train"
    Confident_Test      = "Confident_Test"
    Unsure_Test         = "Unsure_Test"
    HALU_EVAL_SUMMARIZATION = "HALU_EVAL_SUMMARIZATION"

class CalibrationSettings(Enum):
    PERFORM_CALIBRATION = "perform_calibration"
    SKIP_CALIBRATION = "skip_calibration"

class TrainTestSplitSettings(Enum):
    VAL_SIZE_50 = 0.5
    VAL_SIZE_0 = 0.0

class Experiment:
    def __init__(self, 
                 train_origin: Union[DatasetOrigin, List[DatasetOrigin]], 
                 test_origin: Union[DatasetOrigin, List[DatasetOrigin]], 
                 train_test_split_setting: TrainTestSplitSettings,
                 calibration_setting: CalibrationSettings,
                 skip_rows_with_null_values: bool = True, 
                 prompt_columns_in_use: Optional[List[str]] = None):
        self.train_origin = self._ensure_dataset_origin_list(train_origin)
        self.test_origin = self._ensure_dataset_origin_list(test_origin)
        self.skip_rows_with_null_values = skip_rows_with_null_values
        self.prompt_columns_in_use = prompt_columns_in_use or ["prompt_1", "prompt_2"]
        self.calibration_setting = calibration_setting
        self.train_test_split_setting = train_test_split_setting
        self.verify_configuration()
        self.adjust_calibration_setting()
        """
        Initializes an Experiment instance.
        :param train_origin: DatasetOrigin or list of DatasetOrigins for training data.
        :param test_origin: DatasetOrigin or list of DatasetOrigins for testing data.
        :param skip_rows_with_null_values: Boolean indicating whether to skip rows with null values.
        :param prompt_columns_in_use: List of columns to use as prompts. Defaults to ["prompt_1", "prompt_2"].
        """
        self.train_origin = self._ensure_dataset_origin_list(train_origin)
        self.test_origin = self._ensure_dataset_origin_list(test_origin)
        self.skip_rows_with_null_values = skip_rows_with_null_values
        self.prompt_columns_in_use = prompt_columns_in_use or ["prompt_1", "prompt_2"]
        self.calibration_setting = calibration_setting
        self.train_test_split_setting = train_test_split_setting
        self.verify_configuration()
        self.adjust_calibration_setting()

    def adjust_calibration_setting(self):
        """
        Adjusts calibration setting based on validation size.
        """
        if self.train_test_split_setting == TrainTestSplitSettings.VAL_SIZE_0:
            self.calibration_setting = CalibrationSettings.SKIP_CALIBRATION

    def _ensure_dataset_origin_list(self, origins: Union[DatasetOrigin, List[DatasetOrigin]]) -> List[DatasetOrigin]:
        return list(origins) if isinstance(origins, list) else [origins]

    def verify_configuration(self):
        """
        Verifies that the training and testing datasets do not overlap.
        Raises ValueError if there is an intersection.
        """
        train_set = set(self.train_origin)
        test_set = set(self.test_origin)

        if train_set.intersection(test_set):
            raise ValueError("Intersection of train_origin and test_origin should be empty.")
        
    def adjust_calibration_setting(self):
        if self.train_test_split_setting == TrainTestSplitSettings.VAL_SIZE_0:
            self.calibration_setting = CalibrationSettings.SKIP_CALIBRATION

def load_experiment_configs(experiment_config_path) -> List[Experiment]:
    """
    Load experiment configurations from the 'experiment_config.yaml' file and generate a list of Experiment objects.

    Returns:
        List[Experiment]: A list of Experiment objects representing different experiment configurations.
    """
    with open(experiment_config_path, 'r') as file:
        experiment_config = yaml.safe_load(file)
    train_test_combinations = experiment_config["train_test_combinations"]
    desired_prompt_sets = experiment_config["desired_prompt_sets"]
    configs = []
    for train_test_set in train_test_combinations:
        train_set = train_test_set["train_origin"]
        test_set = train_test_set["test_origin"]
        for prompt_set in desired_prompt_sets:
            want_to_calibrate = experiment_config['calibration_configs']["perform_calibration"]
            calibration_setting = CalibrationSettings.PERFORM_CALIBRATION if want_to_calibrate else CalibrationSettings.SKIP_CALIBRATION
            train_test_split_setting = TrainTestSplitSettings.VAL_SIZE_50 if want_to_calibrate else TrainTestSplitSettings.VAL_SIZE_0
            configs.append(Experiment(train_origin=[DatasetOrigin(origin).name for origin in train_set],
                                      test_origin=[DatasetOrigin(origin).name for origin in test_set],
                                      calibration_setting=calibration_setting,
                                      train_test_split_setting=train_test_split_setting,
                                      skip_rows_with_null_values=True,
                                      prompt_columns_in_use=prompt_set))
    return configs
