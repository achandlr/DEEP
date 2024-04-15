import pandas as pd
from src.utils.Evaluator import Evaluator
import pickle
from src.utils.DataLoader import BinaryDataLoader, filter_df_by_non_null_prompt
from src.utils.ModelLoader import load_models
from src.utils.Experiments import load_experiment_configs  
from src.utils.Logger import setup_logger
from src.utils.DataLoader import train_test_split_by_index
import time
from datetime import datetime
from src.utils.Experiments import CalibrationSettings
from tqdm import tqdm
import argparse
from src.utils.ConfigLoader import load_config, get_config_path
from src.utils.DataLoader import load_llm_outputs   

class Benchmark:
    def __init__(self, models, df, experiment_configs):
        """
        Initializes the Benchmark class.

        :param models: List of models (instances of ModelBaseClass or its subclasses).
        :param df: DataFrame containing the data for benchmarking.
        """
        self.models = models
        self.df = df
        self.results = pd.DataFrame()
        self.logger = setup_logger()
        self.experiment_configs = experiment_configs
        self.plot_calibration_curves = False
        self.ground_truth_column_name = "Manual_Eval"
        self.data_train_test_filter_column = 'dataset_name'
        self.llm_results_stored_as_binary = True

    def run_benchmark(self):
        """
        Runs the benchmarking process: training models, making predictions, and evaluating performance.
        """
        df = self.df
        ground_truth_column_name = self.ground_truth_column_name
        experiment_configs = self.experiment_configs

        self.logger.info(f"Training a total of {len(self.models)} models on a total of {len(experiment_configs)} different configurations")        
        for idx, experiment_config in tqdm(enumerate(experiment_configs), desc="Running experiments"):
            self.logger.info(f"START of experiment {idx}: {experiment_config}")
            # Access the experiment configuration
            train_origin = experiment_config.train_origin
            test_origin = experiment_config.test_origin
            skip_nulls = experiment_config.skip_rows_with_null_values
            prompt_columns_in_use = experiment_config.prompt_columns_in_use
   
            if not self.llm_results_stored_as_binary:
                data_loader = BinaryDataLoader()
                df = data_loader.convert_llm_answers_to_binary(df, columns = prompt_columns_in_use, ground_truth_column_name =  ground_truth_column_name)
            if skip_nulls:
                df_no_null = filter_df_by_non_null_prompt(df, needed_non_null_columns = prompt_columns_in_use + [ground_truth_column_name])
                train_df = df_no_null[df_no_null[self.data_train_test_filter_column].isin(train_origin)]
                test_df = df_no_null[df_no_null[self.data_train_test_filter_column].isin(test_origin)]
            else:
                train_df = df[df[self.data_train_test_filter_column].isin(train_origin)]
                test_df = df[df[self.data_train_test_filter_column].isin(test_origin)]

            # Access calibration and train/test split settings
            perform_calibration = experiment_config.calibration_setting == CalibrationSettings.PERFORM_CALIBRATION
            val_size = experiment_config.train_test_split_setting.value
            train_df, val_df = train_test_split_by_index(train_df, test_size=val_size, random_state=42)

            X_train= train_df[prompt_columns_in_use].to_numpy().astype(int)
            Y_train= train_df[ground_truth_column_name].to_numpy().astype(int)

            X_val= val_df[prompt_columns_in_use].to_numpy().astype(int)
            y_val= val_df[ground_truth_column_name].to_numpy().astype(int)

            X_test= test_df[prompt_columns_in_use].to_numpy().astype(int)
            Y_test= test_df[ground_truth_column_name].to_numpy().astype(int)

            assert len(Y_train) > 0, "No training data available. Please check the configuration of the experiment."
            assert len(X_train) > 0, "No training data available. Please check the configuration of the experiment."
            assert len(Y_test) > 0, "No test data available. Please check the configuration of the experiment."
            assert len(X_test) > 0, "No test data available. Please check the configuration of the experiment."

            for model_idx, model in enumerate(tqdm(self.models, desc="Processing models")):
                self.logger.info(f"Running experiment: {experiment_config} on model {self.models}")
                model.train_with_timing_stats(X_train, Y_train) 
                stats = Evaluator.get_stats(model, X_train, Y_train, X_test, Y_test, X_val, y_val, self.plot_calibration_curves, perform_calibration=perform_calibration)
                # Compile results
                self.compile_results(model, stats, experiment_config)

            self.logger.info(f"END of experiment: {experiment_config}")
            
    def compile_results(self, model, stats, experiment_config):
        """
        Compiles the results of the benchmarking process.

        :param model: The model used in the experiment.
        :param stats: The statistics of the model's performance.
        :param experiment_config: The configuration of the experiment.
        """
        model_name = model.__class__.__name__
        model_parameters = model.report_trained_parameters()
        result = {
            'Model': model_name,
            'TrainOrigin': experiment_config.train_origin,
            'TestOrigin': experiment_config.test_origin,
            'SkipNulls': experiment_config.skip_rows_with_null_values,
            'PromptColumnsInUse': experiment_config.prompt_columns_in_use,
            'ModelParameters': model_parameters
        }
        result.update(stats)

        result_df = pd.DataFrame([result])  
        self.results = pd.concat([self.results, result_df], ignore_index=True)

    def get_results(self):
        """
        Returns the results DataFrame.

        :return: The DataFrame containing the benchmarking results.
        """
        return self.results

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Benchmarking script")
    parser.add_argument("--output_file", type=str, default="data/generated/statistics/benchmarking_stats_df.pkl", help="Output file path")
    parser.add_argument("--model_configs_path", type=str, default=None, help="Path to model configurations")
    parser.add_argument("--experiment_configs_path", type=str, default=r"configs/experiment_config.yaml", help="Path to experiment configurations")
    parser.add_argument("--llm_outputs", type=str, default="data/imported/llm_prompt_results.pkl", help="Path to LLM outputs")
    args = parser.parse_args()
    logger = setup_logger()
    logger.info("START OF BENCHMARKING")
    bench_start_time = time.time()
    
    model_configs = load_config(args.model_configs_path)
    experiment_configs = load_experiment_configs(args.experiment_configs_path)  #
    models = load_models(model_configs)

    llm_outputs = load_llm_outputs(args.llm_outputs)
    benchmark = Benchmark(models=models, df=llm_outputs, experiment_configs=experiment_configs)
    benchmark.run_benchmark()

    current_date = datetime.now().strftime("%m-%d-%H-%S-%Y").replace("-", "_")
    with open(f"data/generated/statistics/benchmarking_stats_df_{current_date}.pkl", "wb") as f:
        pickle.dump(benchmark.results, f)

    bench_end_time = time.time()

    logger.info(f"END OF BENCHMARKING: Benchmarking took {round((bench_end_time - bench_start_time)/60, 1)} minutes")