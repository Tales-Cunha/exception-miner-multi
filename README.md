An Empirical Study on Exception Handling Practices in Python Open-Source Projects
---
This repository includes the source code and data for our paper "An Empirical Study on Exception Handling Practices in Python Open-Source Projects".

## Requirements

- Python 3.10+

## Virtualenv (Windows)
1. Run `py -3.10 -m venv {virtualenv name}`
2. Go to Scripts repository in `{virtualenv name}/Scripts`
3. Run `activate`

## Build
To reproduce the results, follow the instructions below.

1. Back to root directory
2. Run `pip install -r requirements.txt` 

## Collecting Projects

To gather open-source projects for analysis, follow these steps:

### Using SEART-GHS (GitHub Search Engine)

1. Go to https://seart-ghs.si.usi.ch/
2. Set your search criteria:
   - **Language**: Select the desired programming language (e.g., Python, TypeScript)
   - **Min Stars**: Set minimum stars (e.g., 100 or more for quality projects)
   - **Sort By**: Select sort order (e.g., Stars, Updated date)
   - Other filters: Forks, License, Topics, etc.
3. Export the search results as CSV

### CSV Format

The exported CSV file must contain the following columns:

- `name`: The organization/user name of the repository
- `repo`: The repository name
- `source`: The source type (typically `github`)

**Example** (using `projects_py_tiny.csv`):
```csv
name,repo,source
pallets,flask,github
ytdl-org,youtube-dl,github
tensorflow,models,github
keras-team,keras,github
ansible,ansible,github
scikit-learn,scikit-learn,github
tiangolo,fastapi,github
3b1b,manim,github
psf,requests,github
```

### Available Sample Files

Pre-configured CSV files are included in the repository:
- `projects_py_tiny.csv`: Small Python projects dataset (9 projects)
- `projects_py.csv`: Full Python projects dataset
- `projects_py_40.csv`: Python projects with 40 entries
- `projects_ts.csv`: TypeScript projects dataset
- `projects_java.csv`: Java projects dataset

## Usage

1. Run the script with the command `python miner.py -in <input_path> -o <output_dir> -lang <language>`

### Parameters

- `<input_path>`: Path to the CSV file that contains the name, repo, and source. This is a required parameter.
- `<output_dir>`: Path to the output directory. If not provided, the default is `output`.
- `<language>`: Programming language of the input file(s). Available options are python, typescript. If not provided, the default is `python`.

### Example

To run the script on a CSV file named `input.csv`, output the results to a directory named `results`, and specify the language as `python`, you would use the following command:

```bash
python miner.py -in input.csv -o results -lang python
```
## LLM Experiment

The LLM experiment evaluates different large language models (LLMs) on four exception handling tasks using various prompt engineering techniques.

### Prerequisites

1. Install Ollama from https://ollama.ai
2. Start the Ollama server locally (default: http://localhost:11434)
3. Pull the required models:
   ```bash
   ollama pull phi4:latest
   ollama pull codellama:latest
   ollama pull mistral:7b
   ollama pull gemma2:9b
   ollama pull deepseek-r1:8b
   ```

### Datasets

The LLM experiment uses datasets located in `/llm/data/`:
- `combined_phi4-latest_task4_style-1-shot_results.zip`: Sample results from the experiment

The main dataset file `tmp/balanced_sample.csv` contains balanced samples of Python functions with and without exception handling.

### Prompts

The experiment uses four different prompt engineering strategies defined in `llm/llm.py`:

1. **Default Prompt**: Simple direct question to the model
2. **1-Shot Prompt**: One example provided to guide the model
3. **Few-Shot Prompt**: Multiple examples (2-3) demonstrating the task
4. **Chain-of-Thought (CoT)**: Encourages the model to think through the problem

These strategies are applied to four tasks:
- **Task 1**: Detect if code requires exception handling (binary classification)
- **Task 2**: Add minimal try/except blocks to code
- **Task 3**: Identify specific exception types to handle
- **Task 4**: Generate exception handling blocks

### Running the LLM Experiment

#### 1. Execute the LLM Analysis Script

Run the main LLM experiment:

```bash
python llm/llm.py [OPTIONS]
```

**Available Options:**
- `--tasks [task1|task2|task3|task4]`: Specify which tasks to run (default: all tasks)
- `--prompt-types [style-default|style-1-shot|style-few-shot|style-cot]`: Specify prompt styles (default: all styles)
- `--models MODEL1 MODEL2 ...`: Specify which models to test (default: phi4:latest, codellama:latest, mistral:7b, gemma2:9b, deepseek-r1:8b)
- `--limit N`: Limit the number of samples processed per task
- `--project PROJECT_NAME`: Set project label for output files (default: combined)
- `-v, --verbose`: Enable verbose logging

**Examples:**

Run all tasks with default settings:
```bash
python llm/llm.py
```

Run only task1 with phi4 model:
```bash
python llm/llm.py --tasks task1 --models phi4:latest
```

Run task2 and task3 with multiple models and few-shot prompts:
```bash
python llm/llm.py --tasks task2 task3 --prompt-types style-few-shot --models phi4:latest mistral:7b
```

Process only 50 samples per task:
```bash
python llm/llm.py --limit 50 -v
```

#### 2. Analyze the Results

After running the experiment, analyze the results using the LLM analysis script:

```bash
python llm/llm_analysis.py
```

This script will:
- Load all generated result CSV files from `llm/output/`
- Calculate evaluation metrics for each task:
  - **Task 1**: Accuracy, Precision, Recall, F-measure
  - **Task 2**: Accuracy, Precision, Recall, F-measure (using try-block vectors)
  - **Task 3**: Accuracy@k, Macro F1-score, Weighted F1-score, Weighted Accuracy
  - **Task 4**: BLEU score, CodeBLEU score
- Generate metrics CSV files: `llm/output/metrics_task1.csv` to `llm/output/metrics_task4.csv`

### Output Files

Results are saved to `llm/output/`:

- `{project}_{model}_{task}_{prompt_type}_results.csv`: Individual task results
- `{project}_{aggregate_model}_results_all.csv`: Combined results from all tasks
- `llm/output/metrics_task{1-4}.csv`: Aggregated evaluation metrics

Each result file contains:
- `func_body`: The original function code
- `llm_response`: The model's response
- `task`: Task number (task1-task4)
- `prompt_type`: Prompt strategy used
- `model`: Model name used

### Configuration

Model parameters can be configured in `llm/models/config.json`:
- `max_tokens`: Maximum tokens in response (default: 150)
- `temperature`: Temperature for response generation (default: 0.7)
- `top_p`: Top-p sampling parameter (default: 0.9)

### Notebooks

Interactive analysis notebooks are available:
- `llm/eh_llm_analysis.ipynb`: Jupyter notebook for detailed LLM experiment analysis

## Unit tests
To run the unit tests, follow the instructions below.

1. Run `python3 -m unittest`

## Coverage report  
To generate the coverage report, follow the instructions below.

1. Run `coverage run -m unittest`
2. Run `coverage report --omit *test_*,*__init__*`
