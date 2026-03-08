import glob
import logging
import os

import pandas as pd

logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)


def analyze_human_evaluation(data_dir="data_human_eval"):
    """
    Analyze human evaluation CSV files to generate summary statistics.
    For each model from the CSV's model column, calculates:
    - Average score across all evaluations
    - Count of scores <= 2 (low score)
    - Count of scores == 3 (medium score)
    - Count of scores >= 4 (high score)
    - Total number of evaluations

    Args:
        data_dir: Directory containing CSV files (default: "data_human_eval")

    Returns:
        pandas DataFrame with summary statistics for each model
        Columns: model, avg_score, score_<=2, score_==3, score_>=4, total_evaluations
    """
    model_mapping = {
        "model1": "codellama",
        "model2": "gemma",
        "model3": "mistral",
        "model4": "phi4"
    }

    csv_files = glob.glob(os.path.join(data_dir, "*.csv"))

    if not csv_files:
        logger.warning(f"No CSV files found in {data_dir}")
        return pd.DataFrame()

    all_data = []

    for csv_file in sorted(csv_files):
        filename = os.path.basename(csv_file)

        try:
            df_eval = pd.read_csv(csv_file)

            df_eval.columns = df_eval.columns.str.strip()

            if "evaluation" not in df_eval.columns:
                logger.warning(f"No 'evaluation' column found in {filename}")
                continue

            if "model" not in df_eval.columns:
                logger.warning(f"No 'model' column found in {filename}")
                continue

            df_eval["evaluation"] = pd.to_numeric(
                df_eval["evaluation"], errors="coerce"
            )

            df_eval_clean = df_eval.dropna(subset=["evaluation"])

            if df_eval_clean.empty:
                logger.warning(f"No valid evaluation scores in {filename}")
                continue

            all_data.append(df_eval_clean)

            logger.info(
                f"Processed {filename}: {len(df_eval_clean)} evaluations"
            )

        except Exception as e:
            logger.error(f"Error processing {filename}: {e}")
            continue

    if not all_data:
        logger.warning("No data collected from any CSV files")
        return pd.DataFrame()

    combined_df = pd.concat(all_data, ignore_index=True)

    summary_data = []
    for model in combined_df["model"].unique():
        model_data = combined_df[combined_df["model"] == model]

        avg_score = model_data["evaluation"].mean()
        count_low = len(model_data[model_data["evaluation"] <= 2])
        count_medium = len(model_data[model_data["evaluation"] == 3])
        count_high = len(model_data[model_data["evaluation"] >= 4])
        total_count = len(model_data)

        model_name = model_mapping.get(model, model)

        summary_data.append(
            {
                "model": model_name,
                "avg_score": round(avg_score, 2),
                "score_<=2": count_low,
                "score_==3": count_medium,
                "score_>=4": count_high,
                "total_evaluations": total_count,
            }
        )

    summary_df = pd.DataFrame(summary_data)

    if not summary_df.empty:
        summary_df = summary_df.sort_values("model").reset_index(drop=True)

    return summary_df


def save_evaluation_summary(
    data_dir="data_human_eval", output_path="output/evaluation_summary.csv"
):
    """
    Analyze human evaluation data and save summary to CSV.

    This function reads all CSV files from the data directory, calculates
    average scores and distribution statistics for each model, and saves
    the results to an output CSV file.

    Args:
        data_dir: Directory containing the evaluation CSV files
        output_path: Path where the summary CSV will be saved

    Returns:
        pandas DataFrame with the summary statistics
    """
    summary_df = analyze_human_evaluation(data_dir)

    if summary_df.empty:
        logger.warning("No data to save. Summary DataFrame is empty.")
        return pd.DataFrame()

    output_dir = os.path.dirname(output_path) or "."
    os.makedirs(output_dir, exist_ok=True)

    summary_df.to_csv(output_path, index=False)
    logger.info(f"Evaluation summary saved to: {output_path}")

    logger.info("\n" + "=" * 70)
    logger.info("HUMAN EVALUATION SUMMARY")
    logger.info("=" * 70)
    logger.info(summary_df.to_string(index=False))
    logger.info("=" * 70)
    logger.info(f"\nSummary saved to: {output_path}\n")

    return summary_df


if __name__ == "__main__":
    os.chdir(os.path.dirname(os.path.abspath(__file__)))

    # Run the analysis
    summary = save_evaluation_summary(
        "data_human_eval", "output/evaluation_summary.csv"
    )
