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

def prompt_default(function, binary_answers=True):
    return "\n".join([
        "Does this Python snippet require exception handling?",
        "<code>",
        function,
        "</code>",
        "Answer with exactly yes or no in lowercase. No explanation."
    ])


def prompt_1_shot(function):
    example_code = EXAMPLE_DIVISION
    example_answer = "yes"
    return "\n".join([
        "Example:",
        "<code>",
        example_code,
        "</code>",
        f"Answer: {example_answer}",
        "",
        "Does the next Python snippet require exception handling?",
        "<code>",
        function,
        "</code>",
        "Respond with exactly yes or no in lowercase. No explanation."
    ])


def prompt_few_shot(function, num_shots=2):
    examples = [
        (EXAMPLE_FILE_READ, "yes"),
        (EXAMPLE_FLOAT_CONVERSION, "yes"),
        (EXAMPLE_BACKEND_CALL, "no"),
    ]
    parts = ["Review the examples and mirror the answer style."]
    for code_text, example_answer in examples[:num_shots]:
        parts.extend(["<code>", code_text, "</code>", f"Answer: {example_answer}", ""])
    parts.extend([
        "Does the next Python snippet require exception handling?",
        "<code>",
        function,
        "</code>",
        "Reply with exactly yes or no in lowercase. No explanation."
    ])
    return "\n".join(parts).strip()



def prompt_cot(function):
    return "\n".join([
        "Assess whether the Python snippet needs exception handling.",
        "<code>",
        function,
        "</code>",
        "Think through possible failure points silently and respond with only yes or no in lowercase."
    ])


def prompt_task2_default(function):
    return "\n".join([
        "Add the minimal required try/except block to this Python snippet.",
        "Keep all unaffected lines exactly as they appear.",
        "Return only the updated code wrapped in <code> tags.",
        "<code>",
        function,
        "</code>",
    ])


def prompt_task2_1_shot(function):
    example_code = EXAMPLE_DIVISION
    example_output = "\n".join([
        "def divide_numbers(a, b):",
        "    try:",
        "        result = a / b",
        "    except ZeroDivisionError:",
        "        print('Division by zero is not allowed')",
        "        return None",
        "    return result",
    ])
    return "\n".join([
        "Example transformation:",
        "<code>",
        example_code,
        "</code>",
        "Output:",
        "<code>",
        example_output,
        "</code>",
        "",
        "Apply the same update to the snippet below.",
        "Return only the modified code wrapped in <code> tags.",
        "<code>",
        function,
        "</code>",
    ])


def prompt_task2_few_shot(function, num_shots=2):
    examples = [
        (
            EXAMPLE_FILE_READ,
            "\n".join([
                "def read_file_binary(file_path: str) -> str:",
                "    try:",
                "        with open(file_path, 'rb') as file:",
                "            return file.read().decode('utf-8')",
                "    except FileNotFoundError:",
                "        print('File not found')",
                "        return ''",
            ]),
        ),
        (
            EXAMPLE_FLOAT_CONVERSION,
            "\n".join([
                "def can_convert_to_float(string):",
                "    try:",
                "        float(string)",
                "        return True",
                "    except ValueError:",
                "        return False",
                ]),
        ),
        (
            EXAMPLE_DICT_ACCESS,
            "\n".join([
                "def get_user_data(user_dict, key):",
                "    try:",
                "        return user_dict[key]",
                "    except KeyError:",
                "        return None",
            ]),
        ),
    ]
    parts = ["Examples of minimal try/except insertion:"]
    for code_text, updated_code in examples[:num_shots]:
        parts.extend([
            "<code>",
            code_text,
            "</code>",
            "Updated:",
            "<code>",
            updated_code,
            "</code>",
            "",
        ])
    parts.extend([
        "Now edit the snippet below in the same style.",
        "Return only the modified code wrapped in <code> tags.",
        "<code>",
        function,
        "</code>",
    ])
    return "\n".join(parts).strip()


def prompt_task2_cot(function):
    return "\n".join([
        "Identify where the Python snippet could fail and insert the minimal try/except block.",
        "<code>",
        function,
        "</code>",
        "Plan silently and reply with only the updated code wrapped in <code> tags."
    ])


def prompt_task3_default(function):
    return "\n".join([
        "List the specific exception names that should be handled for this Python snippet.",
        "<code>",
        function,
        "</code>",
        "Return only the exception names, comma-separated if needed, with no extra text."
    ])


def prompt_task3_1_shot(function):
    example_code = EXAMPLE_DIVISION
    example_output = "ZeroDivisionError"
    return "\n".join([
        "Example:",
        "<code>",
        example_code,
        "</code>",
        f"Answer: {example_output}",
        "",
        "Now identify the exception names for the snippet below.",
        "<code>",
        function,
        "</code>",
        "Return only the exception names, comma-separated if needed, with no extra text."
    ])


def prompt_task3_few_shot(function, num_shots=2):
    examples = [
        (EXAMPLE_FILE_READ, "FileNotFoundError"),
        (EXAMPLE_FLOAT_CONVERSION, "ValueError"),
        (EXAMPLE_DICT_ACCESS, "KeyError"),
    ]
    parts = ["Examples mapping snippets to exception names:"]
    for code_text, exception_name in examples[:num_shots]:
        parts.extend([
            "<code>",
            code_text,
            "</code>",
            f"Answer: {exception_name}",
            "",
        ])
    parts.extend([
        "Provide the exception names for the snippet below.",
        "<code>",
        function,
        "</code>",
        "Return only the exception names, comma-separated if needed, with no extra text."
    ])
    return "\n".join(parts).strip()


def prompt_task3_cot(function):
    return "\n".join([
        "Determine the exception names that should be handled for this Python snippet.",
        "<code>",
        function,
        "</code>",
        "Think through the failure modes silently and reply with only the exception names, comma-separated if needed, no extra text."
    ])


def prompt_task4_default(function):
    return "\n".join([
        "Write only the exception handling block for the risky part of this Python snippet.",
        "Reuse the snippet's identifiers and message style to stay close to the reference solution.",
        "Return exactly one <code> block containing only the except clause with proper indentation.",
        "<code>",
        function,
        "</code>",
    ])


def prompt_task4_1_shot(function):
    example_code = EXAMPLE_DIVISION
    example_output = "\n".join([
        "except ZeroDivisionError:",
        "    print('Division by zero is not allowed')",
    ])
    return "\n".join([
        "Example:",
        "<code>",
        example_code,
        "</code>",
        "Handler:",
        "<code>",
        example_output,
        "</code>",
        "",
        "Write the handler for the snippet below.",
        "Only return exactly one <code> block containing only the except clause with proper indentation.",
        "<code>",
        function,
        "</code>",
    ])


def prompt_task4_few_shot(function, num_shots=2):
    examples = [
        (
            EXAMPLE_FILE_READ,
            "\n".join([
                "except FileNotFoundError:",
                "    print('File not found')",
            ]),
        ),
        (
            EXAMPLE_FLOAT_CONVERSION,
            "\n".join([
                "except ValueError:",
                "    return False",
            ]),
        ),
        (
            EXAMPLE_DICT_ACCESS,
            "\n".join([
                "except KeyError:",
                "    return None",
            ]),
        ),
    ]
    parts = ["Examples of snippet to handler mapping:"]
    for code_text, handler in examples[:num_shots]:
        parts.extend([
            "<code>",
            code_text,
            "</code>",
            "Handler:",
            "<code>",
            handler,
            "</code>",
            "",
        ])
    parts.extend([
        "Write the handler for the snippet below.",
        "Ensure the block mirrors the snippet's naming and keeps four-space indentation.",
        "Return exactly one <code> block containing only the except clause.",
        "<code>",
        function,
        "</code>",
    ])
    return "\n".join(parts).strip()


def prompt_task4_cot(function):
    return "\n".join([
        "Decide which exception block best matches this Python snippet.",
        "<code>",
        function,
        "</code>",
        "Plan silently so the handler aligns closely with the reference style, then reply with a single <code> block containing only the except clause and four-space indentation."
    ])

"""
TODO: Task 5 to evaluate if the LLM is able to create a test to exception handling code to test the exceptional behavior.
However, we need know how to evaluate if the exception test created by the developer is equivalent to the test created by the LLM.
"""

def collect_df(task):
    df = pd.read_csv("tmp/py_stats_combined_2.csv")
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

models = ["phi4:latest", "deepseek-r1:latest"] #codellama:latest
project="combined"
df_result = pd.DataFrame()

# Define a function to save the results to CSV
def save_results_to_csv(df, task, prompt_type, project, model_name):
    output_file = f"{os.getcwd()}/llm/output/{project}_{model_name}_{task}_{prompt_type}_results.csv"
    df.to_csv(output_file, index=False)
    logger.info(f"Results saved to {output_file}")

# Main processing loop
for model in models:
    model_name = model.split("/")[-1] if "/" in model else model
    print(f"Processing with model: {model_name}")

    for task, prompt_functions in TASKS.items():
        print(f"Processing {task}...")
        for prompt_type, prompt_func in prompt_functions.items():
            output = []
            count = 0
            df = collect_df(task)
            for i, row in df.iterrows():
                count += 1
                print(f"Calling {count} of {len(df)} rows for {prompt_type} prompt of {task}")
                if row['n_try_except'] == 1:
                    prompt = prompt_func(row['str_code_without_try_except'])
                else:
                    prompt = prompt_func(row['func_body'])

                # logger.info(f'PROMPT: {prompt}')
                start = time.time()
                response = call_llama(prompt=prompt, model_name=model)
                response_json = response.json()
                logger.info(f'Response status: {response.status_code}')
                logger.info(f'Response JSON: {response_json}')

                if 'response' in response_json:
                    logger.info(f'Generated in {(time.time() - start):.2f} seconds')
                    logger.info('Response....' + response_json['response'])
                    output.append(response_json['response'])
                else:
                    logger.error(f'Error in response: {response_json}')
                    output.append('')

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
