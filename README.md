# Detecting Errors through Ensembling Prompts (DEEP): An End-to-End LLM Framework for Detecting Factual Errors

### Disclaimer
This repository contains the majority of the source code for our publication to ACL (Association for Computational Linguistics). 

## Description
This repository contains an extensive set of ensembling methods for the binary classification a text factual consistency. These methods utilize binary outputs from Large Language Models (LLMs) primed with prompts, treating these outputs as individual features for the ensemble process. By effectively aggregating the binary classifications from LLMs, these ensembling methods aim to significantly improve the accuracy and reliability of factual consistency assessments in summarization tasks. The primary focus of this repository is on the ensembling techniques, rather than aspects like API handling or prompt creation. 

## End-To-End Pipeline Visualization
<div style="border: 1px solid black; padding: 10px;"> <div style="display: flex; justify-content: center;"> <img src="data\\visualizations\\End to End Pipeline.png" alt="End-To-End Pipeline Visualization" style="border: 1px solid black; width: 90%; margin: 5px;" /> </div> </div>

Our framework achieves **state-of-the-art** (SOTA) balanced accuracy on the **AggreFact-XSUM FTSOTA**, **TofuEval Summary-Level**, and **HaluEval Summarization** benchmarks in detecting factual errors within transformer-generated text summaries. It does so without any fine-tuning of the language model or reliance on thresholding techniques not available in practical settings.

It features over 16 different ensembling methods, categorized as follows:

- **Linear Models**: LogisticRegression, LDA

- **Tree-Based Methods**: RandomForest, GradientBoosting, AdaBoost, DecisionTree, CatBoost, XGB, LGBM

- **Ensemble Voting**: MajorityLabelVoter, WeightedMajorityLabelVoter

- **Label Aggregation Models**: LabelModel, Dawid-Skene. The LabelModel, as delineated in Ratner et al. (2017), is particularly effective in learning the conditional probabilities of labeling functions, adeptly reweighting their outputs in semi-supervised contexts. The LabelModel represents an evolution in semi-supervised learning, encompassing techniques such as those in FlyingSquid (Fu et al., 2020), Dawid-Skene (Dawid and Skene, 2018), Data Programming (Ratner et al.,
2016), and MeTaL (Ratner et al., 2019), all of which were originally included as part of the Wrench benchmark (Zhang et al., 2021).

- **Other Methods**: Support Vector Machines, Nearest Neighbors, Naive Bayes (BernboulliNB) 

## Datasets in Use
### Releasing Dataset
To enhance reproducibility, we have made our dataset accessible through HuggingFace, available at [this link](https://huggingface.co/datasets/achandlr/FactualConsistencyScoresTextSummarization). This dataset integrates AggreFact SOTA Benchmark, Halu-Eval Summarization, and TofuEval datasets, designed for assessing factual consistency and detecting hallucinations in text summarization. Additionally, we provide encoder scores for the summaries within this dataset. We are releasing these scores because many GitHub repositories for earlier encoder models depend on outdated libraries that are no longer maintained.

The AggreFact SOTA benchmark [Tang 2023 Understanding](https://arxiv.org/pdf/2205.12854.pdf) is specialized for evaluating the factual consistency of generated text summaries. It consists of three main elements: a summary, a context, and a label indicating whether the summary contains any factual inconsistencies that are not supported by the context. The dataset is derived from an aggregation of nine existing annotated factuality datasets, where existing scores for each dataset are initially binary or converted from different annotation schemes to binary.

The AggreFact benchmark is categorized based on the development timeline of the underlying summarization models into FTSOTA, EXFORMER, and OLD categories. The AggreFact dataset, comprising summaries from CNN/DM  [Herman 2015 Teaching](https://arxiv.org/abs/1506.03340) and XSum [Narayan 2018](https://arxiv.org/abs/1808.08745) articles, is divided into two subsets: AggreFact CNN/DM and AggreFact XSUM. AggreFact XSUM, with its more abstractive summarizations, has a higher error count, while CNN/DM's more extractive nature results in minor deviations that often lead to factual inconsistency annotations. We use the existing AggreFact SOTA validation and test splits from [Tang 2023 Understanding](https://arxiv.org/pdf/2205.12854.pdf) in our work, and use the test subset for all AggreFact benchmarking.

HaluEval [Li 2023 Halu-Eval](https://arxiv.org/abs/2305.11747) is a comprehensive benchmark tailored for assessing hallucination in LLMs. It utilizes a ChatGPT-based two-step framework, specifically a sampling-then-filtering method, to generate its samples. Our focus is on detecting factual inconsistencies in summarization, so we utilize the summarization subset of Halu-Eval. In the Halu-Eval summarization subset, each of the 10,000 sampled document contexts is paired with two summaries: one with hallucination and one without. 

TofuEval [Tang 2024 TofuEval](https://arxiv.org/pdf/2402.13249.pdf) is a topic-focused dialogue summarization benchmark containing 1.5K LLM-generated summaries from the MediaSum and MeetingBank  datasets. TofuEval's binary labels are generated through a two-stage annotation process, where two expert linguists independently assess the binary relevance, completeness, and factual consistency of each sentence in the summaries. We focus solely on the factual consistency annotations of main topic summaries, merging summarization sentences into one paragraph deemed consistent if all summarization sentences are individually marked as consistent.

## Calibration Support
The Calibrator class located in src\utils\Calibrator.py is designed to enhance the accuracy of a classifier's probabilistic predictions. It achieves this by aligning the model's predicted probabilities with the actual frequencies observed in a validation dataset. This process is known as calibration, and it's crucial for understanding a model's confidence in its predictions. The Calibrator takes a classifier and validation data as input, using these to map predicted probabilities to actual outcomes. This mapping is realized through two key functions: actual_from_predicted, which translates a model's predicted probabilities to the actual observed probabilities, and predicted_from_actual, which does the reverse. These functions allow for a more realistic view of the model's predictions.

In addition to creating a meaningful correlation between predicted and actual probabilities, the Calibrator class offers several visualization tools. These include plotting calibration curves and histograms to visually assess the model's calibration. It also computes the Brier score loss, a measure of the accuracy of probabilistic predictions. A low Brier score indicates that the model's predictions are well-calibrated. By using these features, the Calibrator allows for a finer understanding of the likelihood associated with each feature in the model. This is particularly useful in tasks like classifying factual consistency annotations, where it's essential to gauge the confidence level of each prediction accurately.

## Calibration Visualization
<div style="border: 1px solid black; padding: 10px;">
    <div style="display: flex; justify-content: flex-start;">
        <img src="data\\visualizations\\ReliabilityDiagram.png" alt="Calibrated Model" style="border: 1px solid black; width: 100%; margin: 5px;">
    </div>
</div>


## Novel Benchmarking

Encoder-based models for binary factual consistency classification heavily rely on dataset-specific thresholds for converting numerical scores to binary annotations. Optimal thresholds vary widely across datasets (Tang 2023), and small changes can drastically reduce accuracy. Thresholds are often calibrated on test data, leading to overestimated performance on "unseen" data and hindering practical use across various text types without dataset-specific fine-tuning.

Our novel benchmarking of state-of-the-art factual consistency models highlights this limitation, demonstrating substantial performance degradation when thresholds are learned exclusively from training data. 

<div style="border: 1px solid black; padding: 0; display: flex;">
    <div style="margin-right: 5px; border-right: 1px solid black;">
        <img src="data\\visualizations\\EncoderModelsPerformance.png" alt="Encoder Models Performance" style="width: 100%; height: auto;">
    </div>
    <div>
        <img src="data\\visualizations\\OptimalThresholdsPerDataset.png" alt="Optimal Thresholds Per Dataset" style="width: 100%; height: auto;">
    </div>
</div>


# Creating a Virtual Environment
Creating a virtual environment is a crucial step in ensuring that the dependencies for this project do not interfere with those of other Python projects you may have. Here's how to set it up on different operating systems:

## Windows

python -m venv FactualConsistencyBenchmarking

Then activate it doing the following
FactualConsistencyBenchmarking\Scripts\activate

When creating a virtual environment, click yes in the bottom right if given the option to add the virtual enviornment to your workspace
## macOS/Linux
python3 -m venv FactualConsistencyBenchmarking

Then activate it doing the following
source FactualConsistencyBenchmarking/bin/activate

Ensure that your virtual environment name is now visible in your terminal surrounded by parenthesis.

### Prerequisites
- This code was developed using Python 3.11. All other needed libraries are in requirements.txt and can be properly installed by following the steps outlined in Setup. Backwards compatibility with earlier Python Versions is not guarenteed. 
- This code requires one to pass in the a dataframe or CSV file containing the LLM outputs that one wishes to ensemble.

# Setup

#### Clone the repository
Clone the repository:

git clone git@github.com:achandlr/Factual-Consistency-Benchmarking.git

####  Installing Dependencies
Before installing the package, it's important to install the required dependencies listed in requirements.txt. This ensures that all necessary libraries are available to the project. In the root directory of the project, run:
pip install -r requirements.txt

##### Installing the Package
After installing the dependencies, you can install the package itself. For a standard installation, run:

pip install .

For a development installation, which allows you to make changes to the code and see them reflected without reinstallation, run:

pip install -e .

#### Note: 
The setup.py script should handle the installation of the dependencies listed in requirements.txt if they are properly specified. However, manually installing them with pip install -r requirements.txt is a good practice to ensure all dependencies are correctly installed.

#### Benchmark Execution
To execute the main script, Benchmarking.py, ensure that your virtual environment is activated and all dependencies are installed. Then run:

python src\utils\Benchmarking.py

(Replace python with python3 if you are on macOS/Linux and if python does not point to Python 3.x)

### Benchmarking Results
<div style="border: 1px solid black; padding: 10px; text-align: left;">
    <img src="data\\visualizations\\BenchmarkingResults.png" alt="Benchmarking Results" style="border: 1px solid black; width: 80%; height: auto;"/>
</div>

<div style="border: 1px solid black; padding: 10px; text-align: left;">
    <img src="data\\visualizations\\ComparingSOTA.png" alt="Comparing SOTA" style="border: 1px solid black; width: 80%; height: auto;"/>
</div>

## Contributing
This library is written and managed by Alex Chandler.

## Contact
For questions around the use of this library, contact alex.chandler@utexas.edu
