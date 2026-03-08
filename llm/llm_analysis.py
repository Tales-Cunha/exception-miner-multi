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
from nltk.translate.bleu_score import sentence_bleu, SmoothingFunction
from codebleu import calc_codebleu

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
    # Handle NaN or non-string values
    if not isinstance(response, str) or pd.isnull(response):
        return []

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

    markdown_match = re.search(r'```(?:python)?\n(.*?)\n```', response, re.DOTALL)
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
        tokens = re.findall(r'\w+|[()[\]{},:.;=<>!+-/*&|]', code)
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
            smoothing_function=smoothing_function
        )
        return bleu_score
    except:
        return 0.0


# Function to calculate similarity between two code blocks
def code_similarity(code1, code2):
    return SequenceMatcher(None, code1, code2).ratio()


# Load the dataset with model information
df = load_dataframe_with_model()

tasks = ["task1", "task2", "task3", "task4"]
models = df["model"].unique().tolist() if "model" in df.columns and not df.empty else []
metrics_data = {task: [] for task in tasks}

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
                    lambda x: 1 if (isinstance(x, str) and "yes" in x.lower()) else 0
                )

                accuracy = accuracy_score(y_true, y_pred)
                precision = precision_score(y_true, y_pred)
                recall = recall_score(y_true, y_pred)
                f_measure = f1_score(y_true, y_pred)

                print(f"\nMetrics for {prompt_type} prompt in task1:")
                print(f"Accuracy: {accuracy:.2f}")
                print(f"Precision: {precision:.2f}")
                print(f"Recall: {recall:.2f}")
                print(f"F-measure: {f_measure:.2f}")

                cm = confusion_matrix(y_true, y_pred)
                print("\nConfusion Matrix:")
                print(cm)

                metrics_data[task].append(
                    {
                        "task": task,
                        "style-prompt": prompt_type,
                        "model": model,
                        "metric": "Accuracy",
                        "value": accuracy,
                    }
                )
                metrics_data[task].append(
                    {
                        "task": task,
                        "style-prompt": prompt_type,
                        "model": model,
                        "metric": "Precision",
                        "value": precision,
                    }
                )
                metrics_data[task].append(
                    {
                        "task": task,
                        "style-prompt": prompt_type,
                        "model": model,
                        "metric": "Recall",
                        "value": recall,
                    }
                )
                metrics_data[task].append(
                    {
                        "task": task,
                        "style-prompt": prompt_type,
                        "model": model,
                        "metric": "F-measure",
                        "value": f_measure,
                    }
                )

            elif task == "task2":  
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

                accuracy = accuracy_score(y_true, y_pred)
                precision = precision_score(y_true, y_pred)
                recall = recall_score(y_true, y_pred)
                f_measure = f1_score(y_true, y_pred)

                print(f"\nMetrics for {prompt_type} prompt in task2:")
                print(f"Accuracy: {accuracy:.2f}")
                print(f"Precision: {precision:.2f}")
                print(f"Recall: {recall:.2f}")
                print(f"F-measure: {f_measure:.2f}")

                cm = confusion_matrix(y_true, y_pred)
                print("\nConfusion Matrix:")
                print(cm)

                metrics_data[task].append(
                    {
                        "task": task,
                        "style-prompt": prompt_type,
                        "model": model,
                        "metric": "Accuracy",
                        "value": accuracy,
                    }
                )
                metrics_data[task].append(
                    {
                        "task": task,
                        "style-prompt": prompt_type,
                        "model": model,
                        "metric": "Precision",
                        "value": precision,
                    }
                )
                metrics_data[task].append(
                    {
                        "task": task,
                        "style-prompt": prompt_type,
                        "model": model,
                        "metric": "Recall",
                        "value": recall,
                    }
                )
                metrics_data[task].append(
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
                accuracies = [] 
                k = 3  # Set the value of k
                for _, row in results.iterrows():
                    true_exceptions = parse_str_except_identifiers(
                        row["str_except_identifiers"]
                    )
                    predicted_exceptions = parse_task3_response(
                        row["llm_response"]
                    )

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
                    f"\nMean Accuracy@{k} for {prompt_type} prompt in task 3: {mean_accuracy_at_k:.2f}"
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

                print(f"\nDetailed Metrics for {prompt_type} prompt in task 3:")
                print(f"Macro F1-score:   {macro_f1:.2f}")
                print(f"Weighted F1-score:{weighted_f1:.2f}")
                print(f"Weighted Accuracy:{weighted_accuracy:.2f}")

                metrics_data[task].append(
                    {
                        "task": task,
                        "style-prompt": prompt_type,
                        "model": model,
                        "metric": "Mean Accuracy@k",
                        "value": mean_accuracy_at_k,
                    }
                )
                metrics_data[task].append(
                    {
                        "task": task,
                        "style-prompt": prompt_type,
                        "model": model,
                        "metric": "Macro F1-score",
                        "value": macro_f1,
                    }
                )
                metrics_data[task].append(
                    {
                        "task": task,
                        "style-prompt": prompt_type,
                        "model": model,
                        "metric": "Weighted F1-score",
                        "value": weighted_f1,
                    }
                )
                metrics_data[task].append(
                    {
                        "task": task,
                        "style-prompt": prompt_type,
                        "model": model,
                        "metric": "Weighted Accuracy",
                        "value": weighted_accuracy,
                    }
                )

            elif task == "task4":
                y_true_raw = results["str_captures_except"]
                y_pred = results["llm_response"].apply(extract_except_block)

                y_true = []
                for true_raw in y_true_raw:
                    try:
                        if isinstance(true_raw, str) and true_raw.strip().startswith('['):
                            true_list = ast.literal_eval(true_raw)
                            true_code = true_list[0] if isinstance(true_list, list) and len(true_list) > 0 else str(true_list)
                        else:
                            true_code = true_raw
                        y_true.append(true_code)
                    except:
                        y_true.append(true_raw)

                # Calculate BLEU and CodeBLEU scores
                bleu_scores = []
                codebleu_scores = []

                for true, pred in zip(y_true, y_pred):
                    if true and pred:  
                        bleu = calculate_bleu_score(true, pred)
                        bleu_scores.append(bleu)

                        try:
                            codebleu_result = calc_codebleu(
                                [[true]],   
                                [pred],    
                                lang='python'
                            )
                            codebleu = codebleu_result.get('codebleu', 0.0)
                            codebleu_scores.append(codebleu)
                        except Exception as e:
                            logger.warning(f"CodeBLEU calculation failed: {e}")

                if bleu_scores:
                    avg_bleu = sum(bleu_scores) / len(bleu_scores)
                else:
                    avg_bleu = 0.0

                if codebleu_scores:
                    avg_codebleu = sum(codebleu_scores) / len(codebleu_scores)
                else:
                    avg_codebleu = 0.0

                print(f"\nMetrics for {prompt_type} prompt in task 4:")
                print(f"Average BLEU Score:        {avg_bleu:.4f}")
                print(f"Average CodeBLEU Score:    {avg_codebleu:.4f}")

                metrics_data[task].append(
                    {
                        "task": task,
                        "style-prompt": prompt_type,
                        "model": model,
                        "metric": "BLEU",
                        "value": avg_bleu,
                    }
                )
                metrics_data[task].append(
                    {
                        "task": task,
                        "style-prompt": prompt_type,
                        "model": model,
                        "metric": "CodeBLEU",
                        "value": avg_codebleu,
                    }
                )

fieldnames = ["task", "style-prompt", "model", "metric", "value"]

for task_num, task in enumerate(tasks, 1):
    output_csv_path = f"{os.getcwd()}/llm/output/metrics_task{task_num}.csv"
    with open(output_csv_path, mode="w", newline="") as csv_file:
        writer = csv.DictWriter(csv_file, fieldnames=fieldnames)
        writer.writeheader()
        for metric in metrics_data[task]:
            writer.writerow(metric)
    print(f"\nMetrics for {task} saved to: {output_csv_path}")
