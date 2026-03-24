import pandas as pd
#import numpy as np
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix
#import seaborn as sns
#import matplotlib.pyplot as plt
import ast
import re
from difflib import SequenceMatcher
from sklearn.preprocessing import MultiLabelBinarizer
from nltk.translate.bleu_score import sentence_bleu
import os  
import csv 

# Load the dataset
output_dir = "/home/r4ph/desenv/exception-miner-multi/llm/output"  # Specify the output directory
# Only load per-task files (skip _results_all.csv and metrics.csv)
all_files = [
    f for f in os.listdir(output_dir)
    if f.endswith('.csv') and f != 'metrics.csv' and '_results_all' not in f
]

# Read and concatenate all CSV files, extracting model name from filename
df_list = []
for file in all_files:
    tmp = pd.read_csv(os.path.join(output_dir, file))
    # Extract model name: combined_{model}_task{n}_... -> group before '_task'
    match = re.match(r'combined_(.+)_task\d+', file)
    if match:
        tmp['model'] = match.group(1)
    df_list.append(tmp)
df = pd.concat(df_list, ignore_index=True)  # Concatenate all DataFrames

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
                vector[start-1:end] = [1] * (end - start + 1)
    
    return vector

# Function to parse LLM response for task2
def parse_task_response2(response):
    try:
        code_block = re.search(r'<code>(.*?)</code>', response, re.DOTALL).group(1)
        code_block = code_block.replace('\n', '')  # Remove newline characters

        return code_block.strip()
    except:
        return ""

# Function to parse LLM response for task3
def parse_task3_response(response):
    if not isinstance(response, str):
        return []
    # Check if the response has more than 20 lines
    if response.count('\n') > 20:
        return ''  # Return an empty string if there are more than 20 lines

    # Split the response by commas and strip whitespace
    exceptions = [exc.strip() for exc in response.split(',')]
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
        code_block = re.search(r'<code>(.*?)</code>', response, re.DOTALL).group(1)
        code_block = code_block.replace('\n', '')  # Remove newline characters

        return code_block.strip()
    except:
        return ""

# Function to calculate similarity between two code blocks
def code_similarity(code1, code2):
    return SequenceMatcher(None, code1, code2).ratio()

tasks = ['task1', 'task2', 'task3', 'task4']
metrics_data = []

for task in tasks:
    print(f"\nEvaluating {task}:")

    df_task = df[df['task'] == task]

    for model_name in df_task['model'].unique():
        df_model = df_task[df_task['model'] == model_name]

        for prompt_type in df_model['prompt_type'].unique():
            results = df_model[df_model['prompt_type'] == prompt_type]

            if task == 'task1':
                y_true = results['n_try_except']
                y_pred = results['llm_response'].apply(lambda x: 1 if 'yes' in x.lower() else 0)

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

                metrics_data.append({'task': task, 'style-prompt': prompt_type, 'model': model_name, 'metric': 'Accuracy', 'value': accuracy})
                metrics_data.append({'task': task, 'style-prompt': prompt_type, 'model': model_name, 'metric': 'Precision', 'value': precision})
                metrics_data.append({'task': task, 'style-prompt': prompt_type, 'model': model_name, 'metric': 'Recall', 'value': recall})
                metrics_data.append({'task': task, 'style-prompt': prompt_type, 'model': model_name, 'metric': 'F-measure', 'value': f_measure})

            elif task == 'task2':
                y_true = []
                y_pred = []
                for _, row in results.iterrows():
                    original_vector = create_try_vector(row['func_body'])
                    llm_vector = create_try_vector(parse_task_response2(row['llm_response']))

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

                metrics_data.append({'task': task, 'style-prompt': prompt_type, 'model': model_name, 'metric': 'Accuracy', 'value': accuracy})
                metrics_data.append({'task': task, 'style-prompt': prompt_type, 'model': model_name, 'metric': 'Precision', 'value': precision})
                metrics_data.append({'task': task, 'style-prompt': prompt_type, 'model': model_name, 'metric': 'Recall', 'value': recall})
                metrics_data.append({'task': task, 'style-prompt': prompt_type, 'model': model_name, 'metric': 'F-measure', 'value': f_measure})

            elif task == 'task3':
                y_true = []
                y_pred = []
                accuracies = []  # List to store individual Accuracy@k values
                k = 3  # Set the value of k
                for _, row in results.iterrows():
                    true_exceptions = row['str_except_identifiers']
                    predicted_exceptions = parse_task3_response(row['llm_response'])

                    if pd.isnull(true_exceptions):
                        true_exceptions = []
                    elif not isinstance(true_exceptions, list):
                        true_exceptions = [true_exceptions]

                    if predicted_exceptions is None or (isinstance(predicted_exceptions, float) and pd.isnull(predicted_exceptions)):
                        predicted_exceptions = []
                    elif not isinstance(predicted_exceptions, list):
                        predicted_exceptions = [predicted_exceptions]

                    y_true.append(true_exceptions)
                    y_pred.append(predicted_exceptions)

                    accuracy_at_k = 1 if any(item in true_exceptions for item in predicted_exceptions[:k]) else 0
                    accuracies.append(accuracy_at_k)

                mean_accuracy_at_k = sum(accuracies) / len(accuracies) if accuracies else 0
                print(f"\nMean Accuracy@{k} for {model_name} / {prompt_type} prompt in task 3: {mean_accuracy_at_k:.2f}")

                mlb = MultiLabelBinarizer()
                y_true_bin = mlb.fit_transform(y_true)
                y_pred_bin = mlb.transform(y_pred)

                if len(mlb.classes_) == 0:
                    macro_f1 = 0
                    weighted_f1 = 0
                    weighted_accuracy = 0
                else:
                    macro_f1 = f1_score(y_true_bin, y_pred_bin, average='macro', zero_division=0)
                    weighted_f1 = f1_score(y_true_bin, y_pred_bin, average='weighted', zero_division=0)
                    weighted_accuracy = accuracy_score(y_true_bin, y_pred_bin)

                print(f"\nDetailed Metrics for {model_name} / {prompt_type} prompt in task 3:")
                print(f"Macro F1-score:   {macro_f1:.2f}")
                print(f"Weighted F1-score:{weighted_f1:.2f}")
                print(f"Weighted Accuracy:{weighted_accuracy:.2f}")

                metrics_data.append({'task': task, 'style-prompt': prompt_type, 'model': model_name, 'metric': 'Mean Accuracy@k', 'value': mean_accuracy_at_k})
                metrics_data.append({'task': task, 'style-prompt': prompt_type, 'model': model_name, 'metric': 'Macro F1-score', 'value': macro_f1})
                metrics_data.append({'task': task, 'style-prompt': prompt_type, 'model': model_name, 'metric': 'Weighted F1-score', 'value': weighted_f1})
                metrics_data.append({'task': task, 'style-prompt': prompt_type, 'model': model_name, 'metric': 'Weighted Accuracy', 'value': weighted_accuracy})

            elif task == 'task4':
                y_true = results['str_captures_except']
                y_pred = results['llm_response'].apply(extract_except_block)

                bleu_scores = []
                exact_matches = []
                jaccard_scores = []

                from nltk.translate.bleu_score import sentence_bleu, SmoothingFunction
                smooth_fn = SmoothingFunction().method1

                for true, pred in zip(y_true, y_pred):
                    true_tokens = true.split()
                    pred_tokens = pred.split()

                    bleu_score = sentence_bleu([true_tokens], pred_tokens, smoothing_function=smooth_fn)
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
                exact_match_rate = sum(exact_matches) / len(exact_matches) if exact_matches else 0
                average_jaccard = sum(jaccard_scores) / len(jaccard_scores) if jaccard_scores else 0

                print(f"\nMetrics for {model_name} / {prompt_type} prompt in task 4:")
                print(f"Average BLEU Score:         {average_bleu:.2f}")
                print(f"Exact Match Rate:           {exact_match_rate:.2f}")
                print(f"Average Jaccard Similarity: {average_jaccard:.2f}")

                metrics_data.append({'task': task, 'style-prompt': prompt_type, 'model': model_name, 'metric': 'Average BLEU Score', 'value': average_bleu})
                metrics_data.append({'task': task, 'style-prompt': prompt_type, 'model': model_name, 'metric': 'Exact Match Rate', 'value': exact_match_rate})
                metrics_data.append({'task': task, 'style-prompt': prompt_type, 'model': model_name, 'metric': 'Average Jaccard Similarity', 'value': average_jaccard})

output_csv_path = f"{os.getcwd()}/llm/output/metrics.csv"
with open(output_csv_path, mode='w', newline='') as csv_file:
    fieldnames = ['task', 'style-prompt', 'model', 'metric', 'value']
    writer = csv.DictWriter(csv_file, fieldnames=fieldnames)

    writer.writeheader()
    for metric in metrics_data:
        writer.writerow(metric)  

