import ast
import csv
import logging
import os
import re
from difflib import SequenceMatcher

import pandas as pd

from sklearn.metrics import (
    accuracy_score,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
)
from sklearn.preprocessing import MultiLabelBinarizer
from nltk.translate.bleu_score import sentence_bleu
import os
import csv

# Load the dataset
output_dir = "/home/tales/Documents/results/"  # Specify the output directory
# Only load per-task files (skip _results_all.csv and metrics.csv)
all_files = [
    f
    for f in os.listdir(output_dir)
    if f.endswith(".csv") and f != "metrics.csv" and "_results_all" not in f
]

# Read and concatenate all CSV files, extracting model name from filename
df_list = []
for file in all_files:
    tmp = pd.read_csv(os.path.join(output_dir, file))
    # Extract model name: combined_{model}_task{n}_... -> group before '_task'
    match = re.match(r"combined_(.+)_task\d+", file)
    if match:
        tmp["model"] = match.group(1)
    df_list.append(tmp)
df = pd.concat(df_list, ignore_index=True)  # Concatenate all DataFrames


def extract_statements(code):
    """
    Extract top-level statements from code (not nested statements).
    Handles both module-level code and function bodies.
    """
    try:
        tree = ast.parse(code)
        if tree.body and isinstance(tree.body[0], ast.FunctionDef):
            return tree.body[0].body
        return tree.body
    except:
        return []


def create_try_vector(code):
    """
    Create a binary vector indicating which top-level statements are try blocks.
    Example: [0, 1, 1, 0] means the 2nd and 3rd top-level statements are try blocks.

    This correctly maps statements to vector indices using top-level statement order,
    not AST line numbers.
    """
    statements = extract_statements(code)
    vector = [0] * len(statements)

    for i, stmt in enumerate(statements):
        if isinstance(stmt, ast.Try):
            # Mark this statement as a try block
            vector[i] = 1

    return vector


# Function to parse LLM response for task2
def parse_task_response2(response):
    # Handle NaN or non-string values
    if not isinstance(response, str) or pd.isnull(response):
        return ""

    markdown_match = re.search(r"```(?:python)?\n(.*?)\n```", response, re.DOTALL)
    if markdown_match:
        return markdown_match.group(1).strip()

    try:
        code_block = re.search(r"<code>(.*?)</code>", response, re.DOTALL).group(1)
        return code_block.strip()
    except:
        # If no <code> tags found, try to extract Python code directly from response
        # Look for function definitions or other code patterns
        if "def " in response:
            match = re.search(r"(def\s+\w+.*?)(?=\ndef\s|\Z)", response, re.DOTALL)
            if match:
                return match.group(1).strip()
        return ""


def parse_task3_response(response):
    if not isinstance(response, str):
        return []
    # Check if the response has more than 20 lines
    if response.count("\n") > 20:
        return ""  # Return an empty string if there are more than 20 lines

    # Split the response by commas and strip whitespace
    exceptions = [exc.strip() for exc in response.split(",")]
    exceptions = [exc for exc in exceptions if exc]
    return exceptions


def parse_str_except_identifiers(identifiers):
    """
    Parse exception identifiers from string format.

    Handles multiple formats:
    - Single exception: "KeyError"
    - Space-separated: "IOError ValueError"
    - Comma-separated: "KeyError, ValueError"
    """
    if not isinstance(identifiers, str) or pd.isnull(identifiers):
        return []

    identifiers = identifiers.strip()
    if not identifiers:
        return []

    if "," in identifiers:
        exceptions = [exc.strip() for exc in identifiers.split(",")]
    else:
        exceptions = identifiers.split()

    return [exc for exc in exceptions if exc]


def extract_except_block(response):
    """
    Extract code block from LLM response.
    Handles both markdown (```code```) and XML (<code>code</code>) formats.
    """
    if not isinstance(response, str):
        return ""

    markdown_match = re.search(r"```(?:python)?\n(.*?)\n```", response, re.DOTALL)
    if markdown_match:
        return markdown_match.group(1).strip()

    try:
        xml_match = re.search(r"<code>(.*?)</code>", response, re.DOTALL)
        if xml_match:
            return xml_match.group(1).strip()
    except:
        pass

    return ""


def evaluate_exception_handling(true_code, pred_code):
    """Evaluate exception handling code more accurately"""

    def normalize_code(code):
        code = re.sub(r"#.*$", "", code, flags=re.MULTILINE)
        code = re.sub(r"\s+", " ", code).strip()
        return code

    def extract_exception_types(code):
        exception_pattern = r"except\s+(\w+(?:\.\w+)*)(?:\s+as\s+\w+)?:"
        return set(re.findall(exception_pattern, code))

    norm_true = normalize_code(true_code)
    norm_pred = normalize_code(pred_code)

    similarity = SequenceMatcher(None, norm_true, norm_pred).ratio()

    true_exceptions = extract_exception_types(true_code)
    pred_exceptions = extract_exception_types(pred_code)

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


def calculate_bleu_score(reference, hypothesis, weights=(0.25, 0.25, 0.25, 0.25)):
    """
    Calculate BLEU score for code snippets.

    Args:
        reference: Reference (ground truth) code as string
        hypothesis: Generated (predicted) code as string
        weights: Weights for n-grams (default: equal weights for 1-4 grams)

    Returns:
        BLEU score (0-1)
    """

    def tokenize_code(code):
        tokens = re.findall(r"\w+|[()[\]{},:.;=<>!+-/*&|]", code)
        return tokens

    ref_tokens = tokenize_code(reference)
    hyp_tokens = tokenize_code(hypothesis)

    if not hyp_tokens:
        return 0.0

    reference_list = [ref_tokens]

    try:
        # Use smoothing to handle cases with few matches
        smoothing_function = SmoothingFunction().method1
        bleu_score = sentence_bleu(
            reference_list,
            hyp_tokens,
            weights=weights,
            smoothing_function=smoothing_function,
        )
        return bleu_score
    except:
        return 0.0


# Function to calculate similarity between two code blocks
def code_similarity(code1, code2):
    return SequenceMatcher(None, code1, code2).ratio()


tasks = ["task1", "task2", "task3", "task4"]
metrics_data = []

for task in tasks:
    print(f"\nEvaluating {task}:")

    df_task = df[df["task"] == task]

    for model_name in df_task["model"].unique():
        df_model = df_task[df_task["model"] == model_name]

        for prompt_type in df_model["prompt_type"].unique():
            results = df_model[df_model["prompt_type"] == prompt_type]

            if task == "task1":
                y_true = results["n_try_except"].apply(
                    lambda x: 1 if pd.notnull(x) and int(x) > 0 else 0
                )
                y_pred = results["llm_response"].apply(
                    lambda x: 1 if isinstance(x, str) and "yes" in x.lower() else 0
                )

                # Calculate metrics
                accuracy = accuracy_score(y_true, y_pred)
                precision = precision_score(y_true, y_pred)
                recall = recall_score(y_true, y_pred)
                f_measure = f1_score(y_true, y_pred)

                print(f"\nMetrics for {model_name} / {prompt_type} prompt in task1:")
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
                        "model": model_name,
                        "metric": "Accuracy",
                        "value": accuracy,
                    }
                )
                metrics_data.append(
                    {
                        "task": task,
                        "style-prompt": prompt_type,
                        "model": model_name,
                        "metric": "Precision",
                        "value": precision,
                    }
                )
                metrics_data.append(
                    {
                        "task": task,
                        "style-prompt": prompt_type,
                        "model": model_name,
                        "metric": "Recall",
                        "value": recall,
                    }
                )
                metrics_data.append(
                    {
                        "task": task,
                        "style-prompt": prompt_type,
                        "model": model_name,
                        "metric": "F-measure",
                        "value": f_measure,
                    }
                )

            elif task == "task2":
                y_true = []
                y_pred = []
                for _, row in results.iterrows():
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

                print(f"\nMetrics for {model_name} / {prompt_type} prompt in task2:")
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
                        "model": model_name,
                        "metric": "Accuracy",
                        "value": accuracy,
                    }
                )
                metrics_data.append(
                    {
                        "task": task,
                        "style-prompt": prompt_type,
                        "model": model_name,
                        "metric": "Precision",
                        "value": precision,
                    }
                )
                metrics_data.append(
                    {
                        "task": task,
                        "style-prompt": prompt_type,
                        "model": model_name,
                        "metric": "Recall",
                        "value": recall,
                    }
                )
                metrics_data.append(
                    {
                        "task": task,
                        "style-prompt": prompt_type,
                        "model": model_name,
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
                    true_exceptions = row["str_except_identifiers"]
                    predicted_exceptions = parse_task3_response(row["llm_response"])

                    if pd.isnull(true_exceptions):
                        true_exceptions = []
                    elif not isinstance(true_exceptions, list):
                        true_exceptions = parse_str_except_identifiers(true_exceptions)

                    if predicted_exceptions is None or (
                        isinstance(predicted_exceptions, float)
                        and pd.isnull(predicted_exceptions)
                    ):
                        predicted_exceptions = []
                    elif not isinstance(predicted_exceptions, list):
                        predicted_exceptions = [predicted_exceptions]

                    y_true.append(true_exceptions)
                    y_pred.append(predicted_exceptions)

                    accuracy_at_k = (
                        1
                        if any(
                            item in true_exceptions for item in predicted_exceptions[:k]
                        )
                        else 0
                    )
                    accuracies.append(accuracy_at_k)

                mean_accuracy_at_k = (
                    sum(accuracies) / len(accuracies) if accuracies else 0
                )
                print(
                    f"\nMean Accuracy@{k} for {model_name} / {prompt_type} prompt in task 3: {mean_accuracy_at_k:.2f}"
                )

                mlb = MultiLabelBinarizer()
                y_true_bin = mlb.fit_transform(y_true)
                y_pred_bin = mlb.transform(y_pred)

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

                print(
                    f"\nDetailed Metrics for {model_name} / {prompt_type} prompt in task 3:"
                )
                print(f"Macro F1-score:   {macro_f1:.2f}")
                print(f"Weighted F1-score:{weighted_f1:.2f}")
                print(f"Weighted Accuracy:{weighted_accuracy:.2f}")

                metrics_data.append(
                    {
                        "task": task,
                        "style-prompt": prompt_type,
                        "model": model_name,
                        "metric": "Mean Accuracy@k",
                        "value": mean_accuracy_at_k,
                    }
                )
                metrics_data.append(
                    {
                        "task": task,
                        "style-prompt": prompt_type,
                        "model": model_name,
                        "metric": "Macro F1-score",
                        "value": macro_f1,
                    }
                )
                metrics_data.append(
                    {
                        "task": task,
                        "style-prompt": prompt_type,
                        "model": model_name,
                        "metric": "Weighted F1-score",
                        "value": weighted_f1,
                    }
                )
                metrics_data.append(
                    {
                        "task": task,
                        "style-prompt": prompt_type,
                        "model": model_name,
                        "metric": "Weighted Accuracy",
                        "value": weighted_accuracy,
                    }
                )

            elif task == "task4":
                y_true = results["str_captures_except"]
                y_pred = results["llm_response"].apply(extract_except_block)

                bleu_scores = []
                exact_matches = []
                jaccard_scores = []

                from nltk.translate.bleu_score import sentence_bleu, SmoothingFunction

                smooth_fn = SmoothingFunction().method1

                for true, pred in zip(y_true, y_pred):
                    if pd.isnull(true) or not isinstance(true, str):
                        true = ""
                    if pd.isnull(pred) or not isinstance(pred, str):
                        pred = ""

                    true_tokens = true.split()
                    pred_tokens = pred.split()

                    bleu_score = sentence_bleu(
                        [true_tokens], pred_tokens, smoothing_function=smooth_fn
                    )
                    bleu_scores.append(bleu_score)

                    exact_match = 1 if true == pred else 0
                    exact_matches.append(exact_match)

                    set_true = set(true_tokens)
                    set_pred = set(pred_tokens)
                    union = set_true.union(set_pred)
                    intersection = set_true.intersection(set_pred)
                    jaccard = len(intersection) / len(union) if union else 0
                    jaccard_scores.append(jaccard)

                average_bleu = sum(bleu_scores) / len(bleu_scores) if bleu_scores else 0
                exact_match_rate = (
                    sum(exact_matches) / len(exact_matches) if exact_matches else 0
                )
                average_jaccard = (
                    sum(jaccard_scores) / len(jaccard_scores) if jaccard_scores else 0
                )

                print(f"\nMetrics for {model_name} / {prompt_type} prompt in task 4:")
                print(f"Average BLEU Score:         {average_bleu:.2f}")
                print(f"Exact Match Rate:           {exact_match_rate:.2f}")
                print(f"Average Jaccard Similarity: {average_jaccard:.2f}")

                metrics_data.append(
                    {
                        "task": task,
                        "style-prompt": prompt_type,
                        "model": model_name,
                        "metric": "Average BLEU Score",
                        "value": average_bleu,
                    }
                )
                metrics_data.append(
                    {
                        "task": task,
                        "style-prompt": prompt_type,
                        "model": model_name,
                        "metric": "Exact Match Rate",
                        "value": exact_match_rate,
                    }
                )
                metrics_data.append(
                    {
                        "task": task,
                        "style-prompt": prompt_type,
                        "model": model_name,
                        "metric": "Average Jaccard Similarity",
                        "value": average_jaccard,
                    }
                )

output_csv_path = f"{os.getcwd()}/llm/output/metrics.csv"
with open(output_csv_path, mode="w", newline="") as csv_file:
    fieldnames = ["task", "style-prompt", "model", "metric", "value"]
    writer = csv.DictWriter(csv_file, fieldnames=fieldnames)

    writer.writeheader()
    for metric in metrics_data:
        writer.writerow(metric)
