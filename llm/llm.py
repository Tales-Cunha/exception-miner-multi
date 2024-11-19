import pandas as pd
import logging
import time
import requests
import json
import os

logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)

# Define the projects
projects = ["flask"]
dfs = []

# Define prompt functions for task1
def prompt_default(function, binary_answers=True):
    premise = "You will be provided with a python code snippet."
    code_text = function
    instructions = (
        f"Based on the code above, I need you to identify if this code need an exception handling mechanism."
        + f"Please return only with the word 'yes' or 'no'."
    )
    final_text = [
        premise,
        "<code>",
        code_text,
        "</code>",
        instructions
    ]   
    final_text = "\n".join(final_text)
    return final_text

def prompt_1_shot(function):
    example_code = "result = 1 / n"
    example_result = "yes\ntry:\n    result = 1 / n\nexcept ZeroDivisionError:\n    print('Division by zero is not allowed')"
    
    final_text = (
        f"Here is an example:\n<code>\n{example_code}\n</code>\n"
        f"Output: {example_result}\n\n"
        "Now, analyze the following code:\n"
        f"<code>\n{function}\n</code>\n"
        "Does this code need an exception handling mechanism? If yes, please return only the word 'yes'. If not, return only with the word 'no'."
    )
    return final_text

def prompt_few_shot(function, num_shots=2):
    examples = [
        (
            "open('file.txt', 'r')",
            "yes\ntry:\n    open('file.txt', 'r')\nexcept FileNotFoundError:\n    print('File not found')"
        ),
        (
            "value = int('not_an_int')",
            "yes\ntry:\n    value = int('not_an_int')\nexcept ValueError:\n    print('Invalid integer')"
        ),
        (
            "print('Hello, world!')",
            "no"
        )
    ]

    prompt = "Here are some examples of code snippets and their outputs:\n"
    for example_code, example_result in examples[:num_shots]:
        prompt += f"<code>\n{example_code}\n</code>\nOutput: {example_result}\n\n"
    
    prompt += (
        "Now, analyze the following code:\n"
        f"<code>\n{function}\n</code>\n"
        "Does this code need an exception handling mechanism? If yes, please return only the word 'yes'. If not, return only with the word 'no'."
    )
    return prompt

def prompt_cot(function):
    prompt = (
        "Analyze the code snippet below and think step-by-step to determine if it needs an exception handling mechanism:\n"
        f"<code>\n{function}\n</code>\n"
        "First, check if there are any operations that might raise exceptions (e.g., file handling, division, type conversion).\n"
        "Then, consider if these operations are currently protected by try-except blocks. If they are not, an exception handling mechanism may be needed.\n"
        "Finally, provide your answer only with the word 'yes if an exception handling mechanism is not required.'. If not, return only with the word 'no'."
    )
    return prompt

def prompt_task2_default(function):
    prompt = (
        "You will be provided with a Python code snippet that currently lacks exception handling.\n"
        "Your task is to add appropriate try-except blocks to the code where necessary.\n"
        "Return the modified code, keeping the format consistent. \n"
        "The output must include: the code inside the try-except block and the except block that handles the exception.\n"
        "The code must be enclosed within <code> tags.\n\n"
        f"<code>\n{function}\n</code>\n"
    )
    return prompt


def prompt_task2_1_shot(function):
    example_code = "result = 1 / n"
    example_output = (
        "<code>"
        "try:\n"
        "    result = 1 / n\n"
        "except ZeroDivisionError:\n"
        "    print('Division by zero is not allowed')\n"
        "</code>"
    )
    
    prompt = (
        "Here is an example of a Python code snippet and its corresponding exception handling:\n"
        f"<code>\n{example_code}\n</code>\n"
        f"Modified code with exception handling:\n{example_output}\n\n"
        "Now, for the following code, add the required try-except block:\n"
        f"<code>\n{function}\n</code>\n"
        "The output must include: the code inside the try-except block and the except block that handles the exception.\n"
        "The code must be enclosed within <code> tags.\n"
    )
    return prompt


def prompt_task2_few_shot(function, num_shots=2):
    examples = [
        (
            "<code>"
            "open('file.txt', 'r')\n",
            "try:\n"
            "    open('file.txt', 'r')\n"
            "except FileNotFoundError:\n"
            "    print('File not found')\n"
            "</code>"
        ),
        (
            "<code>"
            "value = int('not_a_number')\n",
            "try:\n"
            "    value = int('not_a_number')\n"
            "except ValueError:\n"
            "    print('Invalid integer')\n"
            "</code>"
        ),
        (
            "<code>"
            "result = 1 / 0\n",
            "try:\n"
            "    result = 1 / 0\n"
            "except ZeroDivisionError:\n"
            "    print('Division by zero is not allowed')\n"
            "</code>"
        ),
    ]

    prompt = "Here are examples of Python code snippets and how to add exception handling:\n"
    for example_code, example_output in examples[:num_shots]:
        prompt += f"<code>\n{example_code}\n</code>\nModified code:\n{example_output}\n\n"

    prompt += (
        "Now, add the required try-except block to the following code:\n"
        f"<code>\n{function}\n</code>\n"
        "The output must include: the code inside the try-except block and the except block that handles the exception.\n"
        "The code must be enclosed within <code> tags.\n"
    )
    return prompt


def prompt_task2_cot(function):
    prompt = (
        "Analyze the following code step-by-step to determine where exception handling is necessary:\n"
        f"<code>\n{function}\n</code>\n"
        "1. Identify the parts of the code where exceptions might occur (e.g., file handling, type conversion, division).\n"
        "2. Determine the specific exceptions that can be raised by these operations.\n"
        "3. Add a try-except block to handle the exceptions appropriately.\n"
        "4. Return the modified code with the correct exception handling.\n"
        "The output must include: the code inside the try-except block and the except block that handles the exception.\n"
        "The code must be enclosed within <code> tags.\n"
    )
    return prompt

def prompt_task3_default(function):
    prompt = (
        "You will be provided with a Python code snippet that may require exception handling.\n"
        "Your task is to identify which specific exception(s) should be caught for this code.\n"
        "Return only the name of the exception(s) that should be handled, separated by commas if multiple.\n"
        "Do not provide any explanations or additional context.\n\n"
        f"<code>\n{function}\n</code>\n"
    )
    return prompt

def prompt_task3_1_shot(function):
    example_code = "result = 1 / n"
    example_output = "ZeroDivisionError"
    
    prompt = (
        "Here is an example of a Python code snippet and the exception it should handle:\n"
        f"<code>\n{example_code}\n</code>\n"
        f"Exception to handle: {example_output}\n\n"
        "Now, for the following code, identify the exception(s) that should be handled:\n"
        f"<code>\n{function}\n</code>\n"
        "Return only the name of the exception(s), separated by commas if multiple.\n"
        "Do not provide any explanations or additional context.\n"
    )
    return prompt

def prompt_task3_few_shot(function, num_shots=2):
    examples = [
        (
            "open('file.txt', 'r')",
            "FileNotFoundError"
        ),
        (
            "value = int('not_a_number')",
            "ValueError"
        ),
        (
            "result = 1 / 0",
            "ZeroDivisionError"
        ),
    ]

    prompt = "Here are examples of Python code snippets and the exceptions they should handle:\n"
    for example_code, example_output in examples[:num_shots]:
        prompt += f"<code>\n{example_code}\n</code>\nException to handle: {example_output}\n\n"

    prompt += (
        "Now, identify the exception(s) that should be handled for the following code:\n"
        f"<code>\n{function}\n</code>\n"
        "Return only the name of the exception(s), separated by commas if multiple.\n"
        "Do not provide any explanations or additional context.\n"
    )
    return prompt

def prompt_task3_cot(function):
    prompt = (
        "Analyze the following code step-by-step to determine which exception(s) should be handled:\n"
        f"<code>\n{function}\n</code>\n"
        "1. Identify the operations in the code that might raise exceptions.\n"
        "2. For each operation, consider what type of exception it might raise.\n"
        "3. Determine the most specific exception(s) that should be caught.\n"
        "4. Return only the name(s) of the exception(s) that should be handled, separated by commas if multiple.\n"
        "Do not provide any explanations or additional context.\n"
    )
    return prompt

def prompt_task4_default(function):
    prompt = (
        "You will be provided with a Python code snippet that may require exception handling.\n"
        "Your task is to write only the exception handling block (the 'except' clause) that would be appropriate for this code.\n"
        "Return only the exception handling code block, without the 'try' part, and enclose it within <code> tags.\n\n"
        f"<code>\n{function}\n</code>\n"
    )
    return prompt

def prompt_task4_1_shot(function):
    example_code = "result = 1 / n"
    example_output = "<code>\nexcept ZeroDivisionError:\n    print('Division by zero is not allowed')\n</code>"
    
    prompt = (
        "Here is an example of a Python code snippet and its corresponding exception handling block:\n"
        f"<code>\n{example_code}\n</code>\n"
        f"Exception handling block:\n{example_output}\n\n"
        "Now, for the following code, write only the appropriate exception handling block:\n"
        f"<code>\n{function}\n</code>\n"
        "Return only the exception handling code block, without the 'try' part, and enclose it within <code> tags.\n"
    )
    return prompt

def prompt_task4_few_shot(function, num_shots=2):
    examples = [
        (
            "open('file.txt', 'r')",
            "<code>\nexcept FileNotFoundError:\n    print('File not found')\n</code>"
        ),
        (
            "value = int('not_a_number')",
            "<code>\nexcept ValueError:\n    print('Invalid integer')\n</code>"
        ),
        (
            "import os\nos.remove('/path/to/file')",
            "<code>\nexcept OSError as e:\n    if e.errno == errno.ENOENT:\n        print('File not found')\n    elif e.errno == errno.EACCES:\n        print('Permission denied')\n    else:\n        print(f'Error: {e}')\n</code>"
        ),
    ]

    prompt = "Here are examples of Python code snippets and their corresponding exception handling blocks:\n"
    for example_code, example_output in examples[:num_shots]:
        prompt += f"<code>\n{example_code}\n</code>\nException handling block:\n{example_output}\n\n"

    prompt += (
        "Now, write only the appropriate exception handling block for the following code:\n"
        f"<code>\n{function}\n</code>\n"
        "Return only the exception handling code block, without the 'try' part, and enclose it within <code> tags.\n"
    )
    return prompt

def prompt_task4_cot(function):
    prompt = (
        "Analyze the following code step-by-step to determine the appropriate exception handling block:\n"
        f"<code>\n{function}\n</code>\n"
        "1. Identify the operations in the code that might raise exceptions.\n"
        "2. Determine the specific exceptions that these operations might raise.\n"
        "3. Consider any special conditions or error messages that should be handled.\n"
        "4. Write only the exception handling block (the 'except' clause) that would be appropriate for this code.\n"
        "Return only the exception handling code block, without the 'try' part, and enclose it within <code> tags.\n"
    )
    return prompt

"""
TODO: Task 5 to evaluate if the LLM is able to create a test to exception handling code to test the exceptional behavior.
However, we need know how to evaluate if the exception test created by the developer is equivalent to the test created by the LLM.
"""

def collect_df(task):
    for project in projects:
        #filenames = glob.glob(f"../output/parser/*.csv")
        #df = pd.read_csv(f"../output/parser/{project}_stats.csv")
        df = pd.read_csv("/home/r4ph/desenv/exception-miner-multi-tales/output/parser/py/flask_stats.csv")
        df['project'] = project
        dfs.append(df)

    # for fun in df[df['n_try_except'] == 1].str_code_without_try_except.values:
    #     functions.append(fun)
    
    if task == 'task1':
        pos_samples = df[df['n_try_except'] == 1]
        neg_samples = df[df['n_try_except'] == 0].sample(n=len(pos_samples), random_state=42)
    
        # concat and shuffle the DataFrame rows
        return pd.concat([pos_samples, neg_samples], ignore_index=True).sample(frac=1)
    else:
        return df[df['n_try_except'] == 1]
    # To test:
    # return pd.concat([pos_samples.sample(n=1), neg_samples.sample(n=1)], ignore_index=True)

def call_llama(prompt, model_name):
    headers = {
        "Content-Type": "application/json"
    }

    data = {
        "model": model_name,
        "prompt": prompt,
        "stream": False
    }

    response = requests.post("http://localhost:11434/api/generate", headers=headers, data=json.dumps(data))
    return response

# Define prompt functions for tasks
TASKS = {
    'task1': {
        "style-default": prompt_default,
        "style-1-shot": prompt_1_shot,
        "style-few-shot": prompt_few_shot,
        "style-cot": prompt_cot,
    },
    'task2': {
        "style-default": prompt_task2_default,
        "style-1-shot": prompt_task2_1_shot,
        "style-few-shot": prompt_task2_few_shot,
        "style-cot": prompt_task2_cot,
    },
    'task3': {
        "style-default": prompt_task3_default,
        "style-1-shot": prompt_task3_1_shot,
        "style-few-shot": prompt_task3_few_shot,
        "style-cot": prompt_task3_cot,
    },
    'task4': {
        "style-default": prompt_task4_default,
        "style-1-shot": prompt_task4_1_shot,
        "style-few-shot": prompt_task4_few_shot,
        "style-cot": prompt_task4_cot,
    }
}

start = time.time()
model = "incept5/llama3.1-claude"
model_name = model.split("/")[-1] if "/" in model else model
project="flask"
df_result = pd.DataFrame()
count = 0

# Define a function to save the results to CSV
def save_results_to_csv(df, task, prompt_type, project, model_name):
    output_file = f"{os.getcwd()}/llm/output/{project}_{model_name}_{task}_{prompt_type}_results.csv"
    df.to_csv(output_file, index=False)
    logger.info(f"Results saved to {output_file}")

# Main processing loop
for task, prompt_functions in TASKS.items():
    print(f"Processing {task}...")

    for prompt_type, prompt_func in prompt_functions.items():
        output = []
        df = collect_df(task)
        for i, row in df.iterrows():
            count += 1
            print(f"Calling {count} of {len(df)} rows for {prompt_type} prompt of {task}")
            if row['n_try_except'] == 1:
                prompt = prompt_func(row['str_code_without_try_except'])
            else:
                prompt = prompt_func(row['func_body'])

            logger.info(f'PROMPT: {prompt}')
            response = call_llama(prompt=prompt, model_name=model)
            logger.info(f'Generated {len(response.json())} tokens in {(time.time() - start):.2f} seconds')
            logger.info('Response....' + response.json()['response'])
            output.append(response.json()['response'])

        df_style = df.copy() 
        df_style['task'] = task
        df_style['prompt_type'] = prompt_type
        df_style['llm_response'] = output

        # Save results to CSV after processing each prompt type
        df_result = pd.concat([df_result, df_style], ignore_index=True)
        save_results_to_csv(df_style, task, prompt_type, project, model_name)

# Optionally, you can also combine all results into a final CSV if needed
final_output_file = f"{os.getcwd()}/llm/output/{project}_{model_name}_results.csv"
df_result.to_csv(final_output_file, index=False)
