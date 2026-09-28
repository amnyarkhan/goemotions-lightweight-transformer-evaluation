import os
import pandas as pd
import numpy as np


# Define input and output directories for raw and processed GoEmotions data.
RAW_SPLIT_DIR = "data/raw/splits"
PROCESSED_DIR = "data/processed"

# Store the path to the file containing the ordered list of emotion labels.
EMOTION_FILE = os.path.join(RAW_SPLIT_DIR, "emotions.txt")

# Map each dataset split to its corresponding raw TSV file.
SPLIT_FILES = {
    "train": os.path.join(RAW_SPLIT_DIR, "train.tsv"),
    "val": os.path.join(RAW_SPLIT_DIR, "dev.tsv"),
    "test": os.path.join(RAW_SPLIT_DIR, "test.tsv"),
}


def load_emotions():
    # Load emotion labels in the same order used by the original GoEmotions split files.
    with open(EMOTION_FILE, "r", encoding="utf-8") as f:
        emotions = [line.strip() for line in f if line.strip()]

    return emotions


def parse_label_string(label_string, num_labels):
    # Convert comma-separated label IDs into a multi-hot binary label vector.
    labels = np.zeros(num_labels, dtype=int)

    for label_id in str(label_string).split(","):
        label_id = label_id.strip()

        if label_id == "":
            continue

        labels[int(label_id)] = 1

    return labels


def load_split(path, split_name, emotions):
    # Read one GoEmotions split file and assign standard column names.
    df = pd.read_csv(
        path,
        sep="\t",
        header=None,
        names=["text", "label_ids", "comment_id"],
        quoting=3
    )

    # Convert all raw label ID strings into a binary multi-label matrix.
    label_matrix = np.vstack(
        df["label_ids"].apply(lambda x: parse_label_string(x, len(emotions)))
    )

    # Create a labelled DataFrame where each emotion has its own binary column.
    label_df = pd.DataFrame(label_matrix, columns=emotions)

    # Combine comment metadata, text, and binary emotion labels into one processed table.
    final_df = pd.concat(
        [
            df[["comment_id", "text"]],
            label_df
        ],
        axis=1
    )

    # Add the split name to preserve train, validation, and test provenance.
    final_df["split"] = split_name

    return final_df


def main():
    # Ensure the processed data directory exists before writing output files.
    os.makedirs(PROCESSED_DIR, exist_ok=True)

    # Load the full ordered emotion label list once for all splits.
    emotions = load_emotions()

    processed_splits = []

    for split_name, path in SPLIT_FILES.items():
        # Process each official split using the same label transformation pipeline.
        print(f"Loading {split_name} from {path}")
        split_df = load_split(path, split_name, emotions)
        processed_splits.append(split_df)

        # Save the processed split as a CSV file for later modelling stages.
        output_path = os.path.join(PROCESSED_DIR, f"{split_name}.csv")
        split_df.to_csv(output_path, index=False)

        print(f"Saved {output_path} with shape {split_df.shape}")

    # Combine all processed splits into a single file for EDA and dataset inspection.
    all_data = pd.concat(processed_splits, axis=0).reset_index(drop=True)
    all_output_path = os.path.join(PROCESSED_DIR, "all_splits.csv")
    all_data.to_csv(all_output_path, index=False)

    # Print a concise processing summary for verification.
    print(f"Saved combined data to {all_output_path}")
    print("Emotion labels:")
    print(emotions)
    print("Combined shape:", all_data.shape)


if __name__ == "__main__":
    # Run the preprocessing pipeline when the script is executed directly.
    main()