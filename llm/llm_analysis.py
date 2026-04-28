import ast
import csv
import logging
import os
import re
import shutil
import tempfile
import json
import subprocess
import warnings
import sys
from difflib import SequenceMatcher
from collections import Counter

import pandas as pd
import numpy as np
from sklearn.metrics import (
    accuracy_score,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
)
from sklearn.preprocessing import MultiLabelBinarizer
from nltk.translate.bleu_score import sentence_bleu, SmoothingFunction
from crystalbleu import sentence_bleu as cr_bleu

# Suppress warnings
warnings.filterwarnings("ignore", category=UserWarning, module="crystalbleu")

# Configuration
output_dir = "/home/tales/Mestrado/exception-miner-multi/llm/results"
metrics_output_path = os.path.join(os.getcwd(), "llm", "output", "metrics.csv")
samples_output_path = os.path.join(os.getcwd(), "llm", "output", "discussion_samples.csv")
details_output_path = os.path.join(os.getcwd(), "llm", "output", "task4_quality_details.csv")

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(message)s")
logger = logging.getLogger(__name__)

# Pylint settings
PYLINT_EXCEPTION_CODES = "W0702,W0703,W0706,W0707,W0718,E0702,E0704"
EXCEPTION_VIOLATION_CODES = {"W0702", "W0703", "W0706", "W0707", "W0718", "E0702", "E0704"}

# Cache for Pylint results to avoid redundant calls (especially for ground truth)
pylint_cache = {}

# Simplified Exception Hierarchy for RQ3
EXCEPTION_HIERARCHY = {
    "OSError": ["FileNotFoundError", "PermissionError", "IOError", "FileExistsError"],
    "ArithmeticError": ["ZeroDivisionError", "FloatingPointError", "OverflowError"],
    "LookupError": ["IndexError", "KeyError"],
    "Exception": ["AttributeError", "TypeError", "ValueError", "RuntimeError", "NameError", "ImportError"],
    "BaseException": ["Exception", "SystemExit", "KeyboardInterrupt"]
}

def is_hierarchy_match(true_exc, pred_exc):
    if true_exc == pred_exc: return True
    for parent, children in EXCEPTION_HIERARCHY.items():
        if pred_exc == parent and true_exc in children: return True
        if true_exc == parent and pred_exc in children: return True
    return False

def extract_statements(code):
    try:
        tree = ast.parse(code)
        if tree.body and isinstance(tree.body[0], ast.FunctionDef):
            return tree.body[0].body
        return tree.body
    except: return []

def create_try_vector(code):
    statements = extract_statements(code)
    vector = [0] * len(statements)
    for i, stmt in enumerate(statements):
        if isinstance(stmt, ast.Try): vector[i] = 1
    return vector

def parse_task_response2(response):
    if not isinstance(response, str) or pd.isnull(response): return ""
    markdown_match = re.search(r"```(?:python)?\n(.*?)\n```", response, re.DOTALL)
    if markdown_match: return markdown_match.group(1).strip()
    try:
        return re.search(r"<code>(.*?)</code>", response, re.DOTALL).group(1).strip()
    except:
        if "def " in response:
            match = re.search(r"(def\s+\w+.*?)(?=\ndef\s|\Z)", response, re.DOTALL)
            if match: return match.group(1).strip()
        return ""

def parse_task3_response(response):
    if not isinstance(response, str): return []
    if response.count("\n") > 20: return []
    return [exc.strip() for exc in response.replace("\n", ",").split(",") if exc.strip()]

def parse_str_except_identifiers(identifiers):
    if not isinstance(identifiers, str) or pd.isnull(identifiers): return []
    if "," in identifiers: return [exc.strip() for exc in identifiers.split(",") if exc.strip()]
    return [exc for exc in identifiers.split() if exc]

def parse_tc(tc):
    if not isinstance(tc, str) or pd.isnull(tc): return ""
    tc = tc.strip()
    if tc.startswith("[") and tc.endswith("]"):
        try:
            parsed = ast.literal_eval(tc)
            if isinstance(parsed, list):
                return "\n".join(parsed)
        except Exception:
            pass
    return tc

def extract_except_block(response):
    if not isinstance(response, str): return ""
    markdown_match = re.search(r"```(?:python)?\n(.*?)\n```", response, re.DOTALL)
    if markdown_match: return markdown_match.group(1).strip()
    xml_match = re.search(r"<code>(.*?)</code>", response, re.DOTALL)
    if xml_match: return xml_match.group(1).strip()
    return ""

def wrap_code_for_analysis(code: str) -> str:
    if not code or not isinstance(code, str): return "pass"
    try:
        ast.parse(code)
        return code
    except SyntaxError:
        pass
    code_stripped = code.strip()
    if code_stripped.startswith("except"):
        lines = code.splitlines()
        first_line = lines[0] if lines else ""
        if not first_line.startswith((" ", "\t")):
            indented_code = "\n".join("    " + line for line in lines)
            wrapped = f"def _dummy_function():\n    try:\n        pass\n{indented_code}\n"
        else:
            wrapped = f"def _dummy_function():\n    try:\n        pass\n{code}\n"
        try:
            ast.parse(wrapped)
            return wrapped
        except SyntaxError:
            pass

    indented = "\n".join("    " + line for line in code.splitlines())
    wrapped = f"def _dummy_function():\n{indented}\n"
    try:
        ast.parse(wrapped)
        return wrapped
    except SyntaxError:
        return f"def _dummy_function():\n    try:\n        pass\n    except Exception:\n        {code}\n"


def run_pylint_on_code(code: str, temp_dir: str, file_id: str) -> dict:
    code_hash = hash(code)
    if code_hash in pylint_cache:
        return pylint_cache[code_hash]

    temp_file = os.path.join(temp_dir, f"snippet_{file_id}.py")
    try:
        with open(temp_file, "w", encoding="utf-8") as f: f.write(code)
        # Fix: using sys.executable to ensure correct pylint is called
        cmd = [sys.executable, "-m", "pylint", temp_file, "--disable=all", f"--enable={PYLINT_EXCEPTION_CODES}", "--output-format=json"]
        result = subprocess.run(cmd, capture_output=True, text=True, timeout=10)
        violations = []
        output = result.stdout
        
        if output:
            try:
                # Fix: robust json parsing ignoring non-json prefixes
                if "[" in output and "]" in output:
                    output = output[output.find("["):output.rfind("]")+1]
                for msg in json.loads(output):
                    if msg.get("message-id") in EXCEPTION_VIOLATION_CODES:
                        violations.append(msg.get("message-id"))
            except json.JSONDecodeError: 
                pass
        
        res = {"violation_codes": violations}
        pylint_cache[code_hash] = res
        return res
    except Exception as e: 
        return {"violation_codes": []}

def tokenize_code(code):
    if not isinstance(code, str): return []
    return re.findall(r"\w+|[()[\]{},:.;=<>!+-/*&|]", code)

def get_ngrams(tokens, n):
    return [tuple(tokens[i:i+n]) for i in range(len(tokens)-n+1)]

# 1. Load data
logger.info(f"Loading results from {output_dir}...")
files = [f for f in os.listdir(output_dir) if f.endswith(".csv") and "_results_all" not in f and f != "metrics.csv"]
df_list = []
for file in files:
    try:
        tmp = pd.read_csv(os.path.join(output_dir, file))
        match = re.match(r"combined_(.+?)_task(\d+)", file)
        if match:
            tmp["model"] = match.group(1)
            tmp["task"] = f"task{match.group(2)}"
        df_list.append(tmp)
    except Exception as e: logger.warning(f"Could not read {file}: {e}")

if not df_list:
    logger.error("No data found!")
    exit(1)

df = pd.concat(df_list, ignore_index=True)

# Pre-calculate crystals (1- to 4-grams as tuples)
logger.info("Extracting CrystalBLEU n-grams from test set...")
all_t4 = df[df["task"] == "task4"]["llm_response"].apply(extract_except_block).tolist()
all_ngrams = []
for c in all_t4:
    toks = tokenize_code(c)
    for n in range(1, 5):
        all_ngrams.extend(get_ngrams(toks, n))

ngram_counts = Counter(all_ngrams)
# Using top 500 n-grams as commonly done with CrystalBLEU
crystals = [ngram for ngram, count in ngram_counts.most_common(500)]

metrics_data = []
task4_samples = []
task4_details = []
temp_dir = tempfile.mkdtemp(prefix="eh_")
smooth_fn = SmoothingFunction().method1

for task_name in ["task1", "task2", "task3", "task4"]:
    logger.info(f"Starting evaluation of {task_name}...")
    df_task = df[df["task"] == task_name]
    for model in df_task["model"].unique():
        df_model = df_task[df_task["model"] == model]
        for style in df_model["prompt_type"].unique():
            res = df_model[df_model["prompt_type"] == style]
            
            if task_name == "task1":
                y_true = res["n_try_except"].apply(lambda x: 1 if pd.notnull(x) and int(x) > 0 else 0)
                y_pred = res["llm_response"].apply(lambda x: 1 if isinstance(x, str) and "yes" in x.lower() else 0)
                metrics_data.extend([
                    {"task": task_name, "style-prompt": style, "model": model, "metric": "Accuracy", "value": accuracy_score(y_true, y_pred)},
                    {"task": task_name, "style-prompt": style, "model": model, "metric": "Precision", "value": precision_score(y_true, y_pred, zero_division=0)},
                    {"task": task_name, "style-prompt": style, "model": model, "metric": "Recall", "value": recall_score(y_true, y_pred, zero_division=0)},
                    {"task": task_name, "style-prompt": style, "model": model, "metric": "F1", "value": f1_score(y_true, y_pred, zero_division=0)},
                ])
            
            elif task_name == "task2":
                y_true, y_pred = [], []
                for _, row in res.iterrows():
                    vt = create_try_vector(row["func_body"])
                    vp = create_try_vector(parse_task_response2(row["llm_response"]))
                    ml = max(len(vt), len(vp))
                    vt += [0]*(ml-len(vt)); vp += [0]*(ml-len(vp))
                    y_true.extend(vt); y_pred.extend(vp)
                metrics_data.extend([
                    {"task": task_name, "style-prompt": style, "model": model, "metric": "Accuracy", "value": accuracy_score(y_true, y_pred)},
                    {"task": task_name, "style-prompt": style, "model": model, "metric": "Precision", "value": precision_score(y_true, y_pred, zero_division=0)},
                    {"task": task_name, "style-prompt": style, "model": model, "metric": "Recall", "value": recall_score(y_true, y_pred, zero_division=0)},
                    {"task": task_name, "style-prompt": style, "model": model, "metric": "F1", "value": f1_score(y_true, y_pred, zero_division=0)},
                ])

            elif task_name == "task3":
                h_acc, e_acc, acc_k = [], [], []
                y_true_all, y_pred_all = [], []
                for _, row in res.iterrows():
                    ts = parse_str_except_identifiers(row["str_except_identifiers"])
                    ps = parse_task3_response(row["llm_response"])
                    y_true_all.append(ts); y_pred_all.append(ps)
                    e_acc.append(1 if any(p in ts for p in ps) else 0)
                    h_acc.append(1 if any(is_hierarchy_match(t, p) for t in ts for p in ps) else 0)
                    acc_k.append(1 if any(item in ts for item in ps[:3]) else 0)
                
                mlb = MultiLabelBinarizer()
                y_t_bin = mlb.fit_transform(y_true_all)
                y_p_bin = mlb.transform(y_pred_all)
                
                metrics_data.extend([
                    {"task": task_name, "style-prompt": style, "model": model, "metric": "Exact Accuracy", "value": np.mean(e_acc)},
                    {"task": task_name, "style-prompt": style, "model": model, "metric": "Hierarchy Accuracy", "value": np.mean(h_acc)},
                    {"task": task_name, "style-prompt": style, "model": model, "metric": "Accuracy@3", "value": np.mean(acc_k)},
                    {"task": task_name, "style-prompt": style, "model": model, "metric": "Macro F1", "value": f1_score(y_t_bin, y_p_bin, average="macro", zero_division=0)},
                    {"task": task_name, "style-prompt": style, "model": model, "metric": "Weighted F1", "value": f1_score(y_t_bin, y_p_bin, average="weighted", zero_division=0)},
                ])

            elif task_name == "task4":
                logger.info(f"  Evaluating Task 4: Model={model}, Style={style} ({len(res)} samples)")
                bl, cbl, vl, gtvl, jacc, exact = [], [], [], [], [], []
                total = len(res)
                for idx, (r_idx, row) in enumerate(res.iterrows()):
                    if idx % 50 == 0: logger.info(f"    Progress: {idx}/{total} (Jaccard, CrystalBLEU, Static Analysis...)")
                    
                    tc_raw = row["str_captures_except"] if pd.notnull(row["str_captures_except"]) else ""
                    tc = parse_tc(tc_raw)
                    pc = extract_except_block(row["llm_response"])
                    rt, ht = tokenize_code(tc), tokenize_code(pc)
                    
                    if ht:
                        b_score = sentence_bleu([rt], ht, smoothing_function=smooth_fn)
                        cb_score = cr_bleu([rt], ht, ignoring=crystals, smoothing_function=smooth_fn)
                        bl.append(b_score)
                        cbl.append(cb_score)
                    else: 
                        b_score, cb_score = 0, 0
                        bl.append(0)
                        cbl.append(0)
                    
                    st, sp = set(rt), set(ht)
                    j_score = len(st & sp) / len(st | sp) if st | sp else 0
                    jacc.append(j_score)
                    
                    ex_score = 1 if (tc.strip() == pc.strip() or (rt == ht and rt)) else 0
                    exact.append(ex_score)
                    
                    pred_res = run_pylint_on_code(wrap_code_for_analysis(pc), temp_dir, f"{model}_{style}_{idx}_p")
                    gt_res = run_pylint_on_code(wrap_code_for_analysis(tc), temp_dir, f"{model}_{style}_{idx}_g")
                    
                    v = len(pred_res["violation_codes"])
                    gtv = len(gt_res["violation_codes"])
                    vl.append(v)
                    gtvl.append(gtv)
                    
                    task4_samples.append({"model": model, "style": style, "cbleu": cb_score, "v": v, "true": tc, "pred": pc})
                    task4_details.append({
                        "model": model,
                        "style": style,
                        "file_idx": idx,
                        "true_code": tc,
                        "pred_code": pc,
                        "exact_match": ex_score,
                        "bleu": b_score,
                        "crystalbleu": cb_score,
                        "jaccard": j_score,
                        "pred_violations_count": v,
                        "gt_violations_count": gtv,
                        "pred_violation_codes": ",".join(pred_res["violation_codes"]),
                        "gt_violation_codes": ",".join(gt_res["violation_codes"]),
                    })

                metrics_data.extend([
                    {"task": task_name, "style-prompt": style, "model": model, "metric": "BLEU", "value": np.mean(bl)},
                    {"task": task_name, "style-prompt": style, "model": model, "metric": "CrystalBLEU", "value": np.mean(cbl)},
                    {"task": task_name, "style-prompt": style, "model": model, "metric": "Jaccard", "value": np.mean(jacc)},
                    {"task": task_name, "style-prompt": style, "model": model, "metric": "Exact Match Rate", "value": np.mean(exact)},
                    {"task": task_name, "style-prompt": style, "model": model, "metric": "Avg Violations", "value": np.mean(vl)},
                    {"task": task_name, "style-prompt": style, "model": model, "metric": "GT Avg Violations", "value": np.mean(gtvl)},
                ])

pd.DataFrame(metrics_data).to_csv(metrics_output_path, index=False)

# Save task 4 quality details
if task4_details:
    pd.DataFrame(task4_details).to_csv(details_output_path, index=False)
    logger.info(f"Task 4 detailed quality CSV saved to: {details_output_path}")

if task4_samples:
    s_df = pd.DataFrame(task4_samples)
    valid = s_df[s_df["pred"].str.len() > 5]
    if not valid.empty:
        best = valid.sort_values("cbleu", ascending=False).head(5)
        worst = valid.sort_values("cbleu").head(5)
        pd.concat([best, worst]).to_csv(samples_output_path, index=False)

shutil.rmtree(temp_dir)
logger.info("Done.")