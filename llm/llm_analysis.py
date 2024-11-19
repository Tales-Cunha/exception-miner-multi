import pandas as pd
import numpy as np
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix
import seaborn as sns
import matplotlib.pyplot as plt
import ast
import re
from difflib import SequenceMatcher
from sklearn.preprocessing import MultiLabelBinarizer
from nltk.translate.bleu_score import sentence_bleu
import os  # Import os to interact with the file system

# Load the dataset
output_dir = "/home/r4ph/desenv/exception-miner-multi-tales/llm/output"  # Specify the output directory
all_files = [f for f in os.listdir(output_dir) if f.endswith('.csv')]  # List all CSV files in the directory

# Read and concatenate all CSV files into a single DataFrame
df_list = [pd.read_csv(os.path.join(output_dir, file)) for file in all_files]  # Read each file
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

# Define the tasks
tasks = ['task1', 'task2', 'task3', 'task4']

# Evaluate metrics for each task
for task in tasks:
    print(f"\nEvaluating {task}:")
    
    df_task = df[df['task'] == task]
    
    for prompt_type in df_task['prompt_type'].unique():
        results = df_task[df_task['prompt_type'] == prompt_type] 
        
        if task == 'task1':
            y_true = results['n_try_except']
            y_pred = results['llm_response'].apply(lambda x: 1 if 'yes' in x.lower() else 0)

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

        elif task == 'task2':  # task2
            y_true = []
            y_pred = []
            for _, row in results.iterrows():
                code1 = row['func_body']
                code2 = row['llm_response']
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
            
            print(f"\nMetrics for {prompt_type} prompt in task2:")
            print(f"Accuracy: {accuracy:.2f}")
            print(f"Precision: {precision:.2f}")
            print(f"Recall: {recall:.2f}")
            print(f"F-measure: {f_measure:.2f}")
            
            # Confusion Matrix
            cm = confusion_matrix(y_true, y_pred)
            print("\nConfusion Matrix:")
            print(cm)

        
        elif task == 'task3':
            y_true = []
            y_pred = []
            accuracies = []  # List to store individual accuracies
            for _, row in results.iterrows():
                true_exceptions = (row['str_except_identifiers'])  # Parse true exceptions
                predicted_exceptions = parse_task3_response(row['llm_response'])  # Parse predicted exceptions
                
                # Extend the lists for multi-label classification
                y_true.append(true_exceptions)
                y_pred.append(predicted_exceptions)

                # Calculate Accuracy@k for the current response
                k = 3  # Set the value of k
                accuracy_at_k = 1 if any(item in true_exceptions for item in predicted_exceptions[:k]) else 0
                accuracies.append(accuracy_at_k)  # Store the individual accuracy

            # Calculate the mean accuracy across all responses
            mean_accuracy_at_k = sum(accuracies) / len(accuracies) if accuracies else 0
            
            print(f"\nMean Accuracy@{k} for {prompt_type} prompt in task 3: {mean_accuracy_at_k:.2f}")
            
            # Now Extend the vectors to calculate the accuracy outside of the loop:
            # Flatten the lists for multi-label comparison
            mlb = MultiLabelBinarizer()
            y_true_bin = mlb.fit_transform(y_true)
            y_pred_bin = mlb.transform(y_pred)

            # Calculate metrics
            # accuracy = accuracy_score(y_true_bin, y_pred_bin)
            # precision = precision_score(y_true_bin, y_pred_bin, average='micro', zero_division=0)
            # recall = recall_score(y_true_bin, y_pred_bin, average='micro', zero_division=0)
            # f_measure = f1_score(y_true_bin, y_pred_bin, average='micro', zero_division=0)
            
            # print(f"\nMetrics for {prompt_type} prompt in task 3:")
            # print(f"Accuracy: {mean_accuracy_at_k:.2f}")
            # print(f"Precision: {precision:.2f}")
            # print(f"Recall: {recall:.2f}")
            # print(f"F-measure: {f_measure:.2f}")
        
        elif task == 'task4':
            y_true = results['str_captures_except']  # Assuming this is the correct column
            y_pred = results['llm_response'].apply(extract_except_block)
            
            # Calculate BLEU scores and exact matches
            bleu_scores = []
            exact_matches = []

            for true, pred in zip(y_true, y_pred):
                # Calculate BLEU score
                true_tokens = true.split()  # Tokenize the true string
                pred_tokens = pred.split()  # Tokenize the predicted string
                bleu_score = sentence_bleu([true_tokens], pred_tokens)  # Calculate BLEU score
                bleu_scores.append(bleu_score)

                # Calculate exact match
                exact_match = 1 if true == pred else 0  # 1 if they match, 0 otherwise
                exact_matches.append(exact_match)

            # Calculate average BLEU score and exact match rate
            average_bleu = sum(bleu_scores) / len(bleu_scores) if bleu_scores else 0
            exact_match_rate = sum(exact_matches) / len(exact_matches) if exact_matches else 0
            
            print(f"\nMetrics for {prompt_type} prompt in task 4:")
            print(f"Average BLEU Score: {average_bleu:.2f}")
            print(f"Exact Match Rate: {exact_match_rate:.2f}")

