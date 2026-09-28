import os
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns


PROCESSED_DIR = "data/processed"
FIGURE_DIR = "outputs/figures"
RESULTS_DIR = "outputs/results"

os.makedirs(FIGURE_DIR, exist_ok=True)
os.makedirs(RESULTS_DIR, exist_ok=True)


def get_label_columns(df):
    exclude = {"comment_id", "text", "split"}
    return [col for col in df.columns if col not in exclude]


def plot_label_frequencies(df, label_cols):
    """Generate label-frequency summary and visualise class imbalance."""
    label_counts = df[label_cols].sum().sort_values(ascending=False)

    label_counts.to_csv(os.path.join(RESULTS_DIR, "label_frequencies.csv"))

    plt.figure(figsize=(12, 7))
    label_counts.plot(kind="bar")
    plt.title("GoEmotions Label Frequency Distribution")
    plt.xlabel("Emotion Label")
    plt.ylabel("Number of Examples")
    plt.tight_layout()
    plt.savefig(os.path.join(FIGURE_DIR, "label_frequency_distribution.png"), dpi=300)
    plt.close()

    return label_counts


def plot_labels_per_comment(df, label_cols):
    """Summarise how many emotion labels are assigned to each comment."""
    labels_per_comment = df[label_cols].sum(axis=1)

    summary = labels_per_comment.describe()
    summary.to_csv(os.path.join(RESULTS_DIR, "labels_per_comment_summary.csv"))

    plt.figure(figsize=(8, 5))
    labels_per_comment.value_counts().sort_index().plot(kind="bar")
    plt.title("Number of Emotion Labels per Comment")
    plt.xlabel("Number of Labels")
    plt.ylabel("Number of Comments")
    plt.tight_layout()
    plt.savefig(os.path.join(FIGURE_DIR, "labels_per_comment.png"), dpi=300)
    plt.close()

    return summary


def plot_text_length(df):
    """Analyse comment length to support tokenisation and sequence-length decisions."""
    text_lengths = df["text"].astype(str).apply(lambda x: len(x.split()))

    summary = text_lengths.describe()
    summary.to_csv(os.path.join(RESULTS_DIR, "text_length_summary.csv"))

    plt.figure(figsize=(8, 5))
    text_lengths.hist(bins=50)
    plt.title("Comment Length Distribution")
    plt.xlabel("Number of Words")
    plt.ylabel("Number of Comments")
    plt.tight_layout()
    plt.savefig(os.path.join(FIGURE_DIR, "comment_length_distribution.png"), dpi=300)
    plt.close()

    return summary


def plot_cooccurrence(df, label_cols):
    """Create a co-occurrence matrix to inspect relationships between emotion labels."""
    cooccurrence = df[label_cols].T.dot(df[label_cols])
    cooccurrence.to_csv(os.path.join(RESULTS_DIR, "label_cooccurrence_matrix.csv"))

    plt.figure(figsize=(14, 12))
    sns.heatmap(cooccurrence, cmap="Blues")
    plt.title("Emotion Label Co-occurrence Matrix")
    plt.tight_layout()
    plt.savefig(os.path.join(FIGURE_DIR, "label_cooccurrence_matrix.png"), dpi=300)
    plt.close()

    return cooccurrence


def main():
    df = pd.read_csv(os.path.join(PROCESSED_DIR, "train.csv"))
    label_cols = get_label_columns(df)

    print("Training data shape:", df.shape)
    print("Number of labels:", len(label_cols))

    label_counts = plot_label_frequencies(df, label_cols)
    labels_summary = plot_labels_per_comment(df, label_cols)
    length_summary = plot_text_length(df)
    plot_cooccurrence(df, label_cols)

    print("\nTop labels:")
    print(label_counts.head(10))

    print("\nRare labels:")
    print(label_counts.tail(10))

    print("\nLabels per comment summary:")
    print(labels_summary)

    print("\nText length summary:")
    print(length_summary)

    neutral_count = df["neutral"].sum()
    print("\nNeutral examples:", neutral_count)
    print("Neutral percentage:", neutral_count / len(df) * 100)


if __name__ == "__main__":
    main()