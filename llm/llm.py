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

# Define constants for code examples to avoid duplication
EXAMPLE_FLOAT_CONVERSION = "def can_convert_to_float(string):\n    float(string)\n    return True"
EXAMPLE_FILE_READ = "def read_file_binary(file_path: str) -> str:\n    with open(file_path, 'rb') as file:\n        return file.read().decode('utf-8')"
EXAMPLE_DIVISION = "def divide_numbers(a, b):\n    result = a / b\n    return result"
EXAMPLE_DICT_ACCESS = "def get_user_data(user_dict, key):\n    return user_dict[key]"
EXAMPLE_SIMPLE_FUNCTION = "def get_tokenizer(cls, pretrained_name=None, **kwargs):\n    kwargs.update(cls.special_tokens_map)\n    return cls.tokenizer_class.from_pretrained(pretrained_name, **kwargs)"
EXAMPLE_BACKEND_CALL = "def call(self, x):\n    return backend.nn.silu(x)"

# Define the projects
projects = ["combined"]
dfs = []

# Define prompt functions for task1
def prompt_default(function, binary_answers=True):
    premise = "You are an expert Python developer. You will be provided with a Python code snippet."
    code_text = function
    instructions = (
        "Task: Analyze whether this code snippet requires exception handling to run safely in production.\n"
        "Consider potential runtime errors like: division by zero, file not found, type conversion errors, "
        "network failures, invalid input, missing keys/indices, etc.\n\n"
        "Return exactly one word:\n"
        "- `yes` if the code needs exception handling\n"
        "- `no` if the code is safe without exception handling\n\n"
        "Important: Output only `yes` or `no` in lowercase, with no punctuation, explanation, "
        "or surrounding quotes."
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
    # Real example from the dataset: can_convert_to_float function
    example_code = EXAMPLE_FLOAT_CONVERSION
    example_result = "yes"
    
    final_text = (
        "Here is one example (input -> output):\n"
        "<code>\n"
        f"{example_code}\n"
        "</code>\n"
        f"Output: {example_result}\n\n"
        "Analysis: The float() function can raise ValueError if the string cannot be converted to float.\n\n"
        "Now analyze the following code:\n"
        "<code>\n"
        f"{function}\n"
        "</code>\n"
        "Return exactly one word: `yes` if the code needs exception handling, or `no` if it doesn't. "
        "Output only `yes` or `no` in lowercase."
    )
    return final_text

def prompt_few_shot(function, num_shots=4):
    # Real examples from the dataset
    examples = [
        (
            EXAMPLE_FLOAT_CONVERSION,
            "yes"
        ),
        (
            EXAMPLE_FILE_READ,
            "yes"
        ),
        (
            EXAMPLE_SIMPLE_FUNCTION,
            "no"
        ),
        (
            EXAMPLE_BACKEND_CALL,
            "no"
        )
    ]

    prompt = "Here are examples of Python code snippets and whether they need exception handling:\n\n"
    for i, (example_code, example_result) in enumerate(examples[:num_shots], 1):
        prompt += f"Example {i}:\n<code>\n{example_code}\n</code>\nNeeds exception handling: {example_result}\n\n"
    
    prompt += (
        "Now analyze the following code:\n"
        f"<code>\n{function}\n</code>\n"
        "Does this code need an exception handling mechanism? "
        "Return only `yes` if it needs exception handling, or `no` if it doesn't."
    )
    return prompt

def prompt_cot(function):
    prompt = (
        "Analyze the following Python code step-by-step to determine if it needs exception handling:\n"
        f"<code>\n{function}\n</code>\n\n"
        "Step 1: Identify potentially risky operations in the code:\n"
        "- File I/O operations (open, read, write)\n"
        "- Type conversions (int, float, str)\n"
        "- Division operations (/, //, %)\n"
        "- Network operations (requests, urllib)\n"
        "- Dictionary/list access with keys/indices\n"
        "- Import statements\n"
        "- JSON parsing\n\n"
        "Step 2: Assess if these operations could fail at runtime:\n"
        "- Could the operation receive invalid input?\n"
        "- Could external resources be unavailable?\n"
        "- Could the operation encounter edge cases?\n\n"
        "Step 3: Check if exception handling already exists:\n"
        "- Are there try-except blocks protecting risky operations?\n\n"
        "Step 4: Make your decision:\n"
        "If risky operations exist without protection, return `yes`.\n"
        "If the code is safe or already has exception handling, return `no`.\n\n"
        "Final answer (only `yes` or `no`):"
    )
    return prompt

def prompt_task2_default(function):
    prompt = (
        "You are an expert Python developer. You will be provided with a Python code snippet that needs exception handling.\n"
        "Your task is to add appropriate try-except blocks to make the code production-ready.\n\n"
        "Guidelines:\n"
        "- Identify operations that can raise exceptions\n"
        "- Use specific exception types (not bare except)\n"
        "- Provide meaningful error handling (logging, return values, or user feedback)\n"
        "- Maintain the original code logic and structure\n\n"
        "Return the complete modified code with exception handling enclosed within <code> tags.\n\n"
        f"<code>\n{function}\n</code>\n"
    )
    return prompt


def prompt_task2_1_shot(function):
    # Real example from dataset: can_convert_to_float without exception handling  
    example_code = EXAMPLE_FLOAT_CONVERSION
    example_output = (
        "<code>\n"
        "def can_convert_to_float(string):\n"
        "    try:\n"
        "        float(string)\n"
        "        return True\n"
        "    except ValueError:\n"
        "        return False\n"
        "</code>"
    )
    
    prompt = (
        "Here is an example of adding exception handling to Python code:\n\n"
        "Original code:\n"
        f"<code>\n{example_code}\n</code>\n\n"
        f"Modified code with exception handling:\n{example_output}\n\n"
        "Now, add appropriate exception handling to the following code:\n"
        f"<code>\n{function}\n</code>\n\n"
        "Return the complete modified code with try-except blocks enclosed within <code> tags.\n"
    )
    return prompt


def prompt_task2_few_shot(function, num_shots=3):
    examples = [
        (
            EXAMPLE_FLOAT_CONVERSION,
            "<code>\n"
            "def can_convert_to_float(string):\n"
            "    try:\n"
            "        float(string)\n"
            "        return True\n"
            "    except ValueError:\n"
            "        return False\n"
            "</code>"
        ),
        (
            EXAMPLE_FILE_READ,
            "<code>\n"
            "def read_file_binary(file_path: str) -> str:\n"
            "    try:\n"
            "        with open(file_path, 'rb') as file:\n"
            "            return file.read().decode('utf-8')\n"
            "    except OSError as e:\n"
            "        print(f'File access error: {e}')\n"
            "        return ''\n"
            "    except UnicodeDecodeError:\n"
            "        print('Error decoding file to UTF-8')\n"
            "        return ''\n"
            "</code>"
        ),
        (
            EXAMPLE_DIVISION,
            "<code>\n"
            "def divide_numbers(a, b):\n"
            "    try:\n"
            "        result = a / b\n"
            "        return result\n"
            "    except ZeroDivisionError:\n"
            "        print('Division by zero is not allowed')\n"
            "        return None\n"
            "    except TypeError:\n"
            "        print('Invalid types for division')\n"
            "        return None\n"
            "</code>"
        ),
    ]

    prompt = "Here are examples of adding exception handling to Python code:\n\n"
    for i, (example_code, example_output) in enumerate(examples[:num_shots], 1):
        prompt += f"Example {i}:\nOriginal code:\n<code>\n{example_code}\n</code>\n\nModified code:\n{example_output}\n\n"

    prompt += (
        "Now, add appropriate exception handling to the following code:\n"
        f"<code>\n{function}\n</code>\n\n"
        "Return the complete modified code with try-except blocks enclosed within <code> tags.\n"
    )
    return prompt


def prompt_task2_cot(function):
    prompt = (
        "Add exception handling to the following Python code by thinking through it step-by-step:\n\n"
        f"<code>\n{function}\n</code>\n\n"
        "Step 1: Analyze the code and identify risky operations:\n"
        "- File I/O: open(), read(), write() -> OSError, FileNotFoundError, PermissionError\n"
        "- Type conversion: int(), float(), str() -> ValueError, TypeError\n"
        "- Division: /, //, % -> ZeroDivisionError, TypeError\n"
        "- Dictionary/List access: dict[key], list[index] -> KeyError, IndexError\n"
        "- Network operations: requests, urllib -> ConnectionError, TimeoutError\n"
        "- JSON operations: json.loads() -> JSONDecodeError\n\n"
        "Step 2: Determine the appropriate exception types to catch for each operation.\n\n"
        "Step 3: Design meaningful error handling:\n"
        "- Log the error or provide user feedback\n"
        "- Return a safe default value or None\n"
        "- Re-raise if the error cannot be handled\n\n"
        "Step 4: Write the complete code with try-except blocks:\n"
        "- Wrap risky operations in try blocks\n"
        "- Use specific exception types (avoid bare except)\n"
        "- Maintain the original code structure and logic\n\n"
        "Return the complete modified code enclosed within <code> tags:\n"
    )
    return prompt

def prompt_task3_default(function):
    prompt = (
        "You are an expert Python developer. Analyze the following code snippet to identify which specific "
        "exception(s) should be caught for safe execution.\n\n"
        "Consider these common exception types:\n"
        "- ValueError: Invalid value conversion or format\n"
        "- TypeError: Wrong type for operation\n"
        "- FileNotFoundError, OSError: File/IO operations\n"
        "- ZeroDivisionError: Division by zero\n"
        "- KeyError, IndexError: Dictionary/list access\n"
        "- ImportError, ModuleNotFoundError: Import failures\n"
        "- JSONDecodeError: JSON parsing errors\n"
        "- ConnectionError, TimeoutError: Network operations\n\n"
        f"<code>\n{function}\n</code>\n\n"
        "Return only the specific exception name(s) that should be handled, separated by commas if multiple. "
        "Do not include explanations, code, or extra text.\n"
    )
    return prompt

def prompt_task3_1_shot(function):
    # Real example from dataset
    example_code = EXAMPLE_FLOAT_CONVERSION
    example_output = "ValueError"
    
    prompt = (
        "Here is an example of identifying exceptions for a Python code snippet:\n\n"
        f"<code>\n{example_code}\n</code>\n"
        f"Exception to handle: {example_output}\n\n"
        "Explanation: The float() function raises ValueError when it cannot convert the string to a float.\n\n"
        "Now, identify the exception(s) that should be handled for the following code:\n\n"
        f"<code>\n{function}\n</code>\n\n"
        "Return only the exception name(s), separated by commas if multiple. "
        "No explanations or additional text.\n"
    )
    return prompt

def prompt_task3_few_shot(function, num_shots=4):
    examples = [
        (
            EXAMPLE_FLOAT_CONVERSION,
            "ValueError"
        ),
        (
            EXAMPLE_FILE_READ,
            "OSError, UnicodeDecodeError"
        ),
        (
            EXAMPLE_DIVISION,
            "ZeroDivisionError, TypeError"
        ),
        (
            EXAMPLE_DICT_ACCESS,
            "KeyError"
        )
    ]

    prompt = "Here are examples of Python code snippets and the exceptions they should handle:\n\n"
    for i, (example_code, example_output) in enumerate(examples[:num_shots], 1):
        prompt += f"Example {i}:\n<code>\n{example_code}\n</code>\nException(s): {example_output}\n\n"

    prompt += (
        "Now, identify the exception(s) that should be handled for the following code:\n\n"
        f"<code>\n{function}\n</code>\n\n"
        "Return only the exception name(s), separated by commas if multiple. "
        "No explanations or additional text.\n"
    )
    return prompt

def prompt_task3_cot(function):
    prompt = (
        "Analyze the following Python code step-by-step to identify the specific exception(s) "
        "that should be handled:\n\n"
        f"<code>\n{function}\n</code>\n\n"
        "Step 1: Scan the code for operations that can raise exceptions:\n"
        "- File operations: open(), read(), write(), close() → OSError, FileNotFoundError, PermissionError\n"
        "- Type conversions: int(), float(), str() → ValueError, TypeError\n"
        "- Arithmetic: division (/, //, %) → ZeroDivisionError, TypeError\n"
        "- Data access: dict[key] → KeyError, list[index] → IndexError\n"
        "- Network calls: requests, urllib → ConnectionError, TimeoutError, HTTPError\n"
        "- JSON parsing: json.loads(), json.dumps() → JSONDecodeError\n"
        "- Imports: import, from...import → ImportError, ModuleNotFoundError\n"
        "- Attribute access: obj.attr → AttributeError\n\n"
        "Step 2: For each risky operation, determine the most specific exception type(s) it can raise.\n\n"
        "Step 3: Consider edge cases and input validation requirements.\n\n"
        "Step 4: List the exception name(s) that should be caught, from most specific to most general.\n\n"
        "Final answer (exception names only, separated by commas if multiple):\n"
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
        "Be specific about which exceptions to catch based on the operations in the code.\n"
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
    df = pd.read_csv("/home/talescunha/Jairo/exception-miner-multi/tmp/py_stats_combined.csv")
    df['project'] = 'combined'
    
    if task == 'task1':#700
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
project="combined"
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
