from pathlib import Path
import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parent

FILES = [
    "outputs/results/reduced_model_label_level_statistical_tests.csv",
    "outputs/results/reduced_model_bootstrap_mean_difference_ci.csv",
    "outputs/results/full_model_label_level_statistical_tests.csv",
    "outputs/results/full_model_bootstrap_mean_difference_ci.csv",
    "outputs/results/computational_cost_summary.csv",
    "outputs/results/reduced_per_label_baseline_vs_distilbert_comparison.csv",
    "outputs/results/full_per_label_baseline_vs_distilbert_comparison.csv",
    "outputs/results/reduced_support_metric_correlations.csv",
    "outputs/results/full_support_metric_correlations.csv",
]

OUTPUT_DIR = PROJECT_ROOT / "outputs" / "Summary Tables"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

TXT_OUTPUT = OUTPUT_DIR / "results_raw_output.txt"
EXCEL_OUTPUT = OUTPUT_DIR / "results_tables.xlsx"


def clean_dataframe(df: pd.DataFrame) -> pd.DataFrame:
    """Prepare result tables for appendix export by rounding numeric values."""
    df = df.copy()

    for col in df.columns:
        if pd.api.types.is_numeric_dtype(df[col]):
            df[col] = df[col].round(4)

    return df


def safe_sheet_name(name: str) -> str:
    """Convert a file name into a valid Excel worksheet name."""
    bad_chars = ["\\", "/", "*", "?", ":", "[", "]"]

    for char in bad_chars:
        name = name.replace(char, "_")

    return name[:31]


def main():
    """Export selected result CSV files into a text appendix log and Excel workbook."""
    with open(TXT_OUTPUT, "w", encoding="utf-8") as txt_file:
        txt_file.write(" RAW OUTPUT\n")
        txt_file.write("=" * 80 + "\n\n")

        with pd.ExcelWriter(EXCEL_OUTPUT, engine="openpyxl") as writer:
            for file_path in FILES:
                full_path = PROJECT_ROOT / file_path

                txt_file.write("\n" + "=" * 80 + "\n")
                txt_file.write(f"FILE: {file_path}\n")
                txt_file.write("=" * 80 + "\n\n")

                if not full_path.exists():
                    txt_file.write("STATUS: MISSING FILE\n\n")
                    continue

                try:
                    df = pd.read_csv(full_path)
                    df = clean_dataframe(df)

                    txt_file.write("STATUS: FOUND\n")
                    txt_file.write(f"ROWS: {df.shape[0]}\n")
                    txt_file.write(f"COLUMNS: {df.shape[1]}\n\n")

                    txt_file.write("COLUMN NAMES:\n")
                    txt_file.write(", ".join(df.columns.astype(str)) + "\n\n")

                   

                    sheet_name = safe_sheet_name(Path(file_path).stem)
                    df.to_excel(writer, sheet_name=sheet_name, index=False)

                except Exception as e:
                    txt_file.write("STATUS: ERROR READING FILE\n")
                    txt_file.write(f"ERROR: {e}\n\n")

    print("Done.")
    print(f"Text output saved to: {TXT_OUTPUT}")
    print(f"Excel output saved to: {EXCEL_OUTPUT}")


if __name__ == "__main__":
    main()