import numpy as np
import pickle
import re
import torch
from torch.utils.data import TensorDataset, DataLoader
import numpy as np
import pandas as pd
import random
from collections import Counter
from src.utils.Logger import setup_logger


def load_placeholder_data():
    """
    Generates a placeholder DataFrame with random data.

    Returns:
    pd.DataFrame: The generated placeholder DataFrame.
    """
    def generate_manual_eval():
        return random.choices([1, 0, None], weights=[45, 45, 10], k=1)[0]

    def generate_origin():
        origins = ["AGGREFACCT_SOTA_CNN_DM_DEV", "AGGREFACCT_SOTA_CNN_DM_TEST", "AGGREFACCT_SOTA_CNN_DM_VAL", "AGGREFACCT_SOTA_XSUM_TEST", "AGGREFACCT_SOTA_XSUM_DEV"]
        # Define your likelihood for each origin value
        return random.choices(origins, weights=[20, 20, 20, 20, 20], k=1)[0]

    num_total_data_points = 2000
    data = {
        "Context": [f"Context {i}" for i in range(0, num_total_data_points)],
        "Summary": [f"Summary {i}" for i in range(0, num_total_data_points)],
        "Manual_Eval": [generate_manual_eval() for _ in range(num_total_data_points)],
    }

    for col in ["col1", "col2", "col3", "col4", "col5"]:
        data[col] = [random.choice([data["Manual_Eval"][i], None]) if random.random() < 0.1 else data["Manual_Eval"][i] for i in range(num_total_data_points)]
    
    data["origin"] = [generate_origin() for _ in range(num_total_data_points)]

    placeholder_df = pd.DataFrame(data)
    return placeholder_df

def convert_csv_files_to_df(csv_files):
    """
    Converts two CSV files into a single DataFrame.

    Args:
    file_path_1 (str): The path to the first CSV file.
    file_path_2 (str): The path to the second CSV file.

    Returns:
    pd.DataFrame: The combined DataFrame.
    """
    if isinstance(csv_files, str):
        csv_files = [csv_files]
    dfs = [pd.read_csv(file) for file in csv_files]
    combined_df = pd.concat(dfs)
    return combined_df

def filter_df_by_non_null_prompt(df, needed_non_null_columns):
    """
    Filters the DataFrame to include only rows where all specified columns are non-null.

    Args:
    df (pd.DataFrame): The pandas DataFrame to filter.
    needed_non_null_columns (list): List of column names that must be non-null.

    Returns:
    pd.DataFrame: Filtered DataFrame.
    """
    # Create a boolean mask where each row is True if all needed columns are non-null
    non_null_mask = df[needed_non_null_columns].notnull().all(axis=1)

    # Apply the mask to the DataFrame
    filtered_df = df[non_null_mask]

    return filtered_df


class BinaryDataLoader:
    def __init__(self):
        self.string_error_counter = Counter()
        self.logger = setup_logger()

    def load_data(self, data_format = "dictionary_to_df", skip_rows_with_null = True):
        """
        Loads data into the BinaryDataLoader object.

        Args:
        data_format (str, optional): The format of the data to load. Defaults to "dictionary_to_df".
        skip_rows_with_null (bool, optional): Whether to skip rows with null values. Defaults to True.
        """
        if data_format == "dictionary_to_df":
            self.X_train, self.Y_train = self.convert_devesh_df_to_x_y_array(self.train_file, skip_rows_with_null)
            self.X_test, self.Y_test = self.convert_devesh_df_to_x_y_array(self.test_file, skip_rows_with_null)
        else:
            raise NotImplementedError()

    @staticmethod
    def convert_devesh_df_to_x_y_array(file_name, skip_rows_with_null):
        """
        Converts a Devesh DataFrame to X and Y arrays.

        Args:
        file_name (str): The file name of the Devesh DataFrame.
        skip_rows_with_null (bool): Whether to skip rows with null values.

        Returns:
        dict: A dictionary containing the X and Y arrays, as well as summaries and contexts.
        """
        with open(file_name, "rb") as f:
            devesh_df_data = pickle.load(f) 
        X = []
        contexts = devesh_df_data[list(devesh_df_data.keys())[0]]["Context"].tolist()
        summaries = devesh_df_data[list(devesh_df_data.keys())[0]]["Summary"].tolist()
        bad_value_indices_set = set()

        for key, df in devesh_df_data.items():

            unparsed_data = list(df["Unparsed"])
            predictions = [BinaryDataLoader.advanced_parse_llm_output(output) for output in unparsed_data]
            bad_value_indices = [i for i, x in enumerate(predictions) if x is None]

            bad_value_indices_set.update(set(bad_value_indices))
            X.append(predictions)
        X = np.array(X)
        X = np.transpose(X)

        ground_truth = devesh_df_data['GPT_improved_nli_style_prompt']["Manual_Eval"]
        y = [1 if z == "Correct" else 0 for z in ground_truth]
    
        if skip_rows_with_null :
            X_row_count_with_none = 0
            X_without_none = []
            y_without_none = []
            summaries_without_none = []
            context_without_none = []

            for x, y_val, summary, context  in zip(X, y, summaries, contexts):
                if None in x:
                    X_row_count_with_none +=1
                    continue
                else:
                    X_without_none.append(x)
                    y_without_none.append(y_val)
                    summaries_without_none.append(summary)
                    context_without_none.append(context)

            X_without_none = np.array(X_without_none)
            y_without_none = np.array(y_without_none)

            X_without_none = X_without_none.astype(int)
            y_without_none = y_without_none.astype(int)
            return {"x_array" : X_without_none, "y_array": y_without_none, "summaries" : summaries_without_none, "contexts" : context_without_none}
        
        else:
            X = X.astype(int) 
            y = y.astype(int)
            return {"x_array" : X, "y_array": y, "summaries" : summaries, "contexts" : contexts}

    def report_llm_answer_errors(self):
        """
        Reports the most common errors encountered during LLM answer parsing.
        """
        for string, freq in self.string_error_counter.most_common():
            if freq > 1:
                self.logger.debug(f"ERROR_COUNT: {freq} \t STRING: {string} \n\n")
                print(f"ERROR_COUNT: {freq} \t STRING: {string} \n\n")

    def advanced_parse_llm_output(self, input_string):
        """
        Parses the LLM output for a variety of responses including detailed explanations.
        Prioritizes detection of negated phrases.

        Args:
        input_string (str): The string output from the LLM.

        Returns:
        int or None: True, False, or None based on the analysis of the input string.
        """
        input_string = input_string.lower()
        if input_string == None:
            return None
        elif isinstance(input_string, int) or isinstance(input_string, float):
            return int(input_string)

        negated_pattern = r'\b(not supported|inconsistent|unsupported)\b'
        affirmative_pattern = r'\b(supported|consistent)\b'
        
        negated_match = re.findall(negated_pattern, input_string)
        affirmative_match = re.findall(affirmative_pattern, input_string)

        if negated_match:
            return 0
        elif affirmative_match:
            return 1
        else:
            self.string_error_counter.update([input_string])
            return None
    
    @staticmethod
    def convert_ground_truth_to_binary(input_string):
        """
        Converts a ground truth string to binary.

        Args:
        input_string (str): The ground truth string.

        Returns:
        int: The binary representation of the ground truth.
        """
        if input_string == "Correct":
            return 1
        elif input_string == "Wrong":
            return 0
        else:
            raise NotImplementedError()
        
    def convert_llm_answers_to_binary(self, df, columns, ground_truth_column_name, llm_parsing_method = "advanced_parse_llm_output"):
        """
        Converts LLN answers in a DataFrame to binary representation.

        Args:
        df (pd.DataFrame): The DataFrame containing the LLN answers.
        columns (list): List of column names to convert.
        ground_truth_column_name (str): The column name of the ground truth.
        llm_parsing_method (str, optional): The method to use for LLN answer parsing. Defaults to "advanced_parse_llm_output".

        Returns:
        pd.DataFrame: The DataFrame with converted LLN answers.
        """
        for col in columns:
            if col in df.columns:
                if llm_parsing_method == "advanced_parse_llm_output":
                    df[col] = df[col].apply(lambda x: self.advanced_parse_llm_output(x) if isinstance(x, str) else x)
                elif llm_parsing_method == "convert_correct_wrong_to_binary":
                    df[col] = df[col].apply(lambda x: BinaryDataLoader.convert_ground_truth_to_binary(x) if isinstance(x, str) else x)
                else:
                    raise ValueError()
        df[ground_truth_column_name] = df[ground_truth_column_name].apply(lambda x: BinaryDataLoader.convert_ground_truth_to_binary(x) if isinstance(x, str) else x)
        
        return df

    def convert_to_torch_train_test_data_loader(self, batch_size=64, set_class_data_loaders = True):
        """
        Converts the data to PyTorch DataLoader objects.

        Args:
        batch_size (int, optional): The batch size for the DataLoader. Defaults to 64.
        set_class_data_loaders (bool, optional): Whether to set the class data loaders. Defaults to True.

        Returns:
        DataLoader, DataLoader: The training and testing DataLoader objects.
        """
        X_train_tensor = torch.tensor(self.X_train.astype(np.float32))
        Y_train_tensor = torch.tensor(self.Y_train.astype(np.float32))
        X_test_tensor = torch.tensor(self.X_test.astype(np.float32))
        Y_test_tensor = torch.tensor(self.Y_test.astype(np.float32))

        train_dataset = TensorDataset(X_train_tensor, Y_train_tensor)
        test_dataset = TensorDataset(X_test_tensor, Y_test_tensor)

        train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
        test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

        if set_class_data_loaders:
            self.train_loader = train_loader
            self.test_loader = test_loader
        
        return train_loader, test_loader
    
def train_test_split_by_index(df, test_size=0.2, random_state=42):
    """
    Splits a DataFrame into training and validation sets based on indices.

    Args:
    df (pd.DataFrame): The DataFrame to split.
    test_size (float): The proportion of the dataset to include in the validation split.
    random_state (int, optional): Controls the shuffling applied to the data before applying the split. Pass an int for reproducible output across multiple function calls.

    Returns:
    pd.DataFrame, pd.DataFrame: The training and validation DataFrames.
    """
    if random_state is not None:
        np.random.seed(random_state)

    indices = np.random.permutation(len(df)) if random_state is not None else range(len(df))

    split_index = int((1 - test_size) * len(df))

    train_indices = indices[:split_index]
    val_indices = indices[split_index:]

    train_df = df.iloc[train_indices]
    val_df = df.iloc[val_indices]

    return train_df, val_df

def load_llm_outputs(file_path):
    dtype = file_path.split(".")[-1]
    assert dtype in ["pkl", "csv"], "dtype must be either 'pkl' or 'csv'"
    if dtype == "pkl":
        with open(file_path, "rb") as f:
            llm_outputs = pickle.load(f)
    else:
        llm_outputs = pd.read_csv(file_path)
    return llm_outputs