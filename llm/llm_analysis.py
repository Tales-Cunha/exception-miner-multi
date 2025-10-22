# import seaborn as sns
# import matplotlib.pyplot as plt
import ast
import csv
import logging
import os
import re
from difflib import SequenceMatcher

import pandas as pd

# import numpy as np
from sklearn.metrics import (
    accuracy_score,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
)
from sklearn.preprocessing import MultiLabelBinarizer

logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)


def extract_model_from_filename(filename):
    """
    Extract model name from filename pattern:
    - combined_{model}_results_all.csv
    - combined_{model}_task{N}_style-{style}_results.csv
    Example: combined_phi4:latest_task1_style-few-shot_results.csv -> phi4:latest
    Example: combined_codellama-latest_results_all.csv -> codellama-latest
    """
    # Try pattern for aggregate results file first
    match = re.match(r"combined_(.+?)_results_all\.csv", filename)
    if match:
        return match.group(1)

    # Try pattern for individual task files
    match = re.match(r"combined_(.+?)_task\d+", filename)
    if match:
        return match.group(1)
    return None


def load_dataframe_with_model():
    """
    Load all CSV files from output directory and add model column based on filename.
    Returns a pandas DataFrame with an additional 'model' column.
    """
    output_dir = "llm/output"
    all_files = [
        f
        for f in os.listdir(output_dir)
        if f.endswith(".csv") and f.startswith("combined_")
    ]
    logger.info(f"Collecting results from...: {all_files}")

    df_list = []
    for file in all_files:
        model_name = extract_model_from_filename(file)
        if model_name:
            df_temp = pd.read_csv(os.path.join(output_dir, file))
            df_temp["model"] = model_name
            df_list.append(df_temp)

    if df_list:
        return pd.concat(df_list, ignore_index=True)
    else:
        return pd.DataFrame()


def extract_statements(code):
    try:
        tree = ast.parse(code)
        return [node for node in ast.walk(tree) if isinstance(node, ast.stmt)]
    except:
        return []


""" Function to create vector representation of try block. 
For example, if the code has 3 statements inside the try block that contains 6 statements, the vector will be [1, 1, 1, 0, 0, 0]
"""


def create_try_vector(code):
    statements = extract_statements(code)
    vector = [0] * len(statements)

    for i, stmt in enumerate(statements):
        if isinstance(stmt, ast.Try):
            for body_stmt in stmt.body:
                start = body_stmt.lineno
                end = body_stmt.end_lineno
                vector[start - 1 : end] = [1] * (end - start + 1)

    return vector


# Function to parse LLM response for task2
def parse_task_response2(response):
    # Handle NaN or non-string values
    if not isinstance(response, str) or pd.isnull(response):
        return ""

    try:
        # Try to extract code block with <code> tags
        code_block = re.search(r"<code>(.*?)</code>", response, re.DOTALL).group(1)
        # Keep newlines for proper Python parsing
        return code_block.strip()
    except:
        # If no <code> tags found, try to extract Python code directly from response
        # Look for function definitions or other code patterns
        if "def " in response:
            # Try to extract the main function definition
            match = re.search(r"(def\s+\w+.*?)(?=\ndef\s|\Z)", response, re.DOTALL)
            if match:
                return match.group(1).strip()
        return ""


# Function to parse LLM response for task3
def parse_task3_response(response):
    # Handle NaN or non-string values
    if not isinstance(response, str) or pd.isnull(response):
        return []

    # Split the response by commas and strip whitespace
    exceptions = [exc.strip() for exc in response.split(",")]
    exceptions = [exc for exc in exceptions if exc]
    return exceptions


# Function to parse str_except_identifiers
def parse_str_except_identifiers(identifiers):
    try:
        # Convert string representation of list to actual list
        return ast.literal_eval(identifiers)
    except:
        return []


# Function to extract except block from LLM response for task4
def extract_except_block(response):
    try:
        # Extract code block from the response
        code_block = re.search(r"<code>(.*?)</code>", response, re.DOTALL).group(1)
        code_block = code_block.replace("\n", "")  # Remove newline characters

        return code_block.strip()
    except:
        return ""


def evaluate_exception_handling(true_code, pred_code):
    """Evaluate exception handling code more accurately"""

    # Clean and normalize the code snippets
    def normalize_code(code):
        # Remove whitespace variations and comments
        code = re.sub(r"#.*$", "", code, flags=re.MULTILINE)
        code = re.sub(r"\s+", " ", code).strip()
        return code

    # Extract exception types
    def extract_exception_types(code):
        exception_pattern = r"except\s+(\w+(?:\.\w+)*)(?:\s+as\s+\w+)?:"
        return set(re.findall(exception_pattern, code))

    # Normalize both code snippets
    norm_true = normalize_code(true_code)
    norm_pred = normalize_code(pred_code)

    # Calculate basic similarity
    similarity = SequenceMatcher(None, norm_true, norm_pred).ratio()

    # Compare exception types
    true_exceptions = extract_exception_types(true_code)
    pred_exceptions = extract_exception_types(pred_code)

    # Calculate exception type precision and recall
    exception_precision = (
        len(true_exceptions.intersection(pred_exceptions)) / len(pred_exceptions)
        if pred_exceptions
        else 0
    )
    exception_recall = (
        len(true_exceptions.intersection(pred_exceptions)) / len(true_exceptions)
        if true_exceptions
        else 0
    )
    exception_f1 = (
        2
        * (exception_precision * exception_recall)
        / (exception_precision + exception_recall)
        if (exception_precision + exception_recall) > 0
        else 0
    )

    return {
        "text_similarity": similarity,
        "exception_precision": exception_precision,
        "exception_recall": exception_recall,
        "exception_f1": exception_f1,
    }


# Function to calculate similarity between two code blocks
def code_similarity(code1, code2):
    return SequenceMatcher(None, code1, code2).ratio()


# Load the dataset with model information
df = load_dataframe_with_model()

tasks = ["task1", "task2", "task3", "task4"]
models = df["model"].unique().tolist() if "model" in df.columns and not df.empty else []
metrics_data = []

print(f"Found {len(models)} models: {models}")
print(f"Found {len(df)} total rows")

for model in models:
    for task in tasks:
        print(f"\nEvaluating {model} - {task}:")

        df_task = df[(df["task"] == task) & (df["model"] == model)]

        for prompt_type in df_task["prompt_type"].unique():
            results = df_task[df_task["prompt_type"] == prompt_type]

            if task == "task1":
                y_true = results["n_try_except"]
                y_pred = results["llm_response"].apply(
                    lambda x: 1 if "yes" in x.lower() else 0
                )

                # Calculate metrics
                accuracy = accuracy_score(y_true, y_pred)
                precision = precision_score(y_true, y_pred)
                recall = recall_score(y_true, y_pred)
                f_measure = f1_score(y_true, y_pred)

                print(f"\nMetrics for {prompt_type} prompt in task1:")
                print(f"Accuracy: {accuracy:.2f}")
                print(f"Precision: {precision:.2f}")
                print(f"Recall: {recall:.2f}")
                print(f"F-measure: {f_measure:.2f}")

                # Confusion Matrix
                cm = confusion_matrix(y_true, y_pred)
                print("\nConfusion Matrix:")
                print(cm)

                metrics_data.append(
                    {
                        "task": task,
                        "style-prompt": prompt_type,
                        "model": model,
                        "metric": "Accuracy",
                        "value": accuracy,
                    }
                )
                metrics_data.append(
                    {
                        "task": task,
                        "style-prompt": prompt_type,
                        "model": model,
                        "metric": "Precision",
                        "value": precision,
                    }
                )
                metrics_data.append(
                    {
                        "task": task,
                        "style-prompt": prompt_type,
                        "model": model,
                        "metric": "Recall",
                        "value": recall,
                    }
                )
                metrics_data.append(
                    {
                        "task": task,
                        "style-prompt": prompt_type,
                        "model": model,
                        "metric": "F-measure",
                        "value": f_measure,
                    }
                )

            elif task == "task2":  # task2
                y_true = []
                y_pred = []
                for _, row in results.iterrows():
                    code1 = row["func_body"]
                    code2 = row["llm_response"]
                    original_vector = create_try_vector(row["func_body"])
                    llm_vector = create_try_vector(
                        parse_task_response2(row["llm_response"])
                    )

                    max_len = max(len(original_vector), len(llm_vector))
                    original_vector += [0] * (max_len - len(original_vector))
                    llm_vector += [0] * (max_len - len(llm_vector))

                    y_true.extend(original_vector)
                    y_pred.extend(llm_vector)

                # Calculate metrics
                accuracy = accuracy_score(y_true, y_pred)
                precision = precision_score(y_true, y_pred)
                recall = recall_score(y_true, y_pred)
                f_measure = f1_score(y_true, y_pred)

                print(f"\nMetrics for {prompt_type} prompt in task2:")
                print(f"Accuracy: {accuracy:.2f}")
                print(f"Precision: {precision:.2f}")
                print(f"Recall: {recall:.2f}")
                print(f"F-measure: {f_measure:.2f}")

                # Confusion Matrix
                cm = confusion_matrix(y_true, y_pred)
                print("\nConfusion Matrix:")
                print(cm)

                metrics_data.append(
                    {
                        "task": task,
                        "style-prompt": prompt_type,
                        "model": model,
                        "metric": "Accuracy",
                        "value": accuracy,
                    }
                )
                metrics_data.append(
                    {
                        "task": task,
                        "style-prompt": prompt_type,
                        "model": model,
                        "metric": "Precision",
                        "value": precision,
                    }
                )
                metrics_data.append(
                    {
                        "task": task,
                        "style-prompt": prompt_type,
                        "model": model,
                        "metric": "Recall",
                        "value": recall,
                    }
                )
                metrics_data.append(
                    {
                        "task": task,
                        "style-prompt": prompt_type,
                        "model": model,
                        "metric": "F-measure",
                        "value": f_measure,
                    }
                )

            elif task == "task3":
                y_true = []
                y_pred = []
                accuracies = []  # List to store individual Accuracy@k values
                k = 3  # Set the value of k
                for _, row in results.iterrows():
                    true_exceptions = row[
                        "str_except_identifiers"
                    ]  # Parse true exceptions
                    predicted_exceptions = parse_task3_response(
                        row["llm_response"]
                    )  # Parse predicted exceptions

                    # Handle potential NaN values and ensure the values are lists
                    if pd.isnull(true_exceptions):
                        true_exceptions = []
                    elif not isinstance(true_exceptions, list):
                        true_exceptions = [true_exceptions]

                    if predicted_exceptions is None or (
                        isinstance(predicted_exceptions, float)
                        and pd.isnull(predicted_exceptions)
                    ):
                        predicted_exceptions = []
                    elif not isinstance(predicted_exceptions, list):
                        predicted_exceptions = [predicted_exceptions]

                    # Extend the lists for multi-label classification
                    y_true.append(true_exceptions)
                    y_pred.append(predicted_exceptions)

                    # Calculate Accuracy@k for the current response
                    accuracy_at_k = (
                        1
                        if any(
                            item in true_exceptions for item in predicted_exceptions[:k]
                        )
                        else 0
                    )
                    accuracies.append(accuracy_at_k)  # Store the individual accuracy

                # Calculate the mean Accuracy@k
                mean_accuracy_at_k = (
                    sum(accuracies) / len(accuracies) if accuracies else 0
                )
                print(
                    f"\nMean Accuracy@{k} for {prompt_type} prompt in task 3: {mean_accuracy_at_k:.2f}"
                )

                # Multi-label classification metrics
                mlb = MultiLabelBinarizer()
                y_true_bin = mlb.fit_transform(y_true)
                y_pred_bin = mlb.transform(y_pred)

                # Check if there are any classes; if not, set metrics to 0 to avoid errors
                if len(mlb.classes_) == 0:
                    macro_f1 = 0
                    weighted_f1 = 0
                    weighted_accuracy = 0
                else:
                    macro_f1 = f1_score(
                        y_true_bin, y_pred_bin, average="macro", zero_division=0
                    )
                    weighted_f1 = f1_score(
                        y_true_bin, y_pred_bin, average="weighted", zero_division=0
                    )
                    weighted_accuracy = accuracy_score(y_true_bin, y_pred_bin)

                # Print detailed metrics
                print(f"\nDetailed Metrics for {prompt_type} prompt in task 3:")
                print(f"Macro F1-score:   {macro_f1:.2f}")
                print(f"Weighted F1-score:{weighted_f1:.2f}")
                print(f"Weighted Accuracy:{weighted_accuracy:.2f}")

                # Save metrics
                metrics_data.append(
                    {
                        "task": task,
                        "style-prompt": prompt_type,
                        "model": model,
                        "metric": "Mean Accuracy@k",
                        "value": mean_accuracy_at_k,
                    }
                )
                metrics_data.append(
                    {
                        "task": task,
                        "style-prompt": prompt_type,
                        "model": model,
                        "metric": "Macro F1-score",
                        "value": macro_f1,
                    }
                )
                metrics_data.append(
                    {
                        "task": task,
                        "style-prompt": prompt_type,
                        "model": model,
                        "metric": "Weighted F1-score",
                        "value": weighted_f1,
                    }
                )
                metrics_data.append(
                    {
                        "task": task,
                        "style-prompt": prompt_type,
                        "model": model,
                        "metric": "Weighted Accuracy",
                        "value": weighted_accuracy,
                    }
                )

            elif task == "task4":
                y_true = results["str_captures_except"]
                y_pred = results["llm_response"].apply(extract_except_block)

                # Advanced metrics
                metrics = []
                for true, pred in zip(y_true, y_pred):
                    if true and pred:  # Only evaluate if both exist
                        evaluation = evaluate_exception_handling(true, pred)
                        metrics.append(evaluation)

                # Calculate average metrics
                if metrics:
                    avg_text_similarity = sum(
                        m["text_similarity"] for m in metrics
                    ) / len(metrics)
                    avg_exception_precision = sum(
                        m["exception_precision"] for m in metrics
                    ) / len(metrics)
                    avg_exception_recall = sum(
                        m["exception_recall"] for m in metrics
                    ) / len(metrics)
                    avg_exception_f1 = sum(m["exception_f1"] for m in metrics) / len(
                        metrics
                    )
                else:
                    avg_text_similarity = avg_exception_precision = (
                        avg_exception_recall
                    ) = avg_exception_f1 = 0

                print(f"\nAdvanced Metrics for {prompt_type} prompt in task 4:")
                print(f"Average Text Similarity:       {avg_text_similarity:.2f}")
                print(f"Exception Type Precision:      {avg_exception_precision:.2f}")
                print(f"Exception Type Recall:         {avg_exception_recall:.2f}")
                print(f"Exception Type F1:             {avg_exception_f1:.2f}")

                metrics_data.append(
                    {
                        "task": task,
                        "style-prompt": prompt_type,
                        "model": model,
                        "metric": "Text Similarity",
                        "value": avg_text_similarity,
                    }
                )
                metrics_data.append(
                    {
                        "task": task,
                        "style-prompt": prompt_type,
                        "model": model,
                        "metric": "Exception Precision",
                        "value": avg_exception_precision,
                    }
                )
                metrics_data.append(
                    {
                        "task": task,
                        "style-prompt": prompt_type,
                        "model": model,
                        "metric": "Exception Recall",
                        "value": avg_exception_recall,
                    }
                )
                metrics_data.append(
                    {
                        "task": task,
                        "style-prompt": prompt_type,
                        "model": model,
                        "metric": "Exception F1",
                        "value": avg_exception_f1,
                    }
                )

output_csv_path = f"{os.getcwd()}/llm/output/metrics.csv"
with open(output_csv_path, mode="w", newline="") as csv_file:
    fieldnames = ["task", "style-prompt", "model", "metric", "value"]
    writer = csv.DictWriter(csv_file, fieldnames=fieldnames)

    writer.writeheader()
    for metric in metrics_data:
        writer.writerow(metric)
