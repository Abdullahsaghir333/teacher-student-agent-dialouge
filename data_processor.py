import os
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from datasets import load_dataset

# Global configuration
HF_DATASET = "derek-thomas/ScienceQA"
OUTPUT_FILE = "scienceqa_plot.png"


class ScienceQALoader:
    """
    A redesigned class that handles reading, cleaning, and plotting
    the ScienceQA dataset (functionally identical but structurally different).
    """

    def __init__(self, source_name=HF_DATASET):
        self.source_name = source_name
        self.raw = None
        self.table = None

    def fetch(self):
        """Retrieve dataset from HuggingFace."""
        print(f"📡 Fetching data: {self.source_name}")
        try:
            self.raw = load_dataset(self.source_name, split="train")
            print(f"✔️ Loaded dataset with {len(self.raw)} records.")
        except Exception as err:
            print(f"❗ Failed to load dataset: {err}")
            raise

    def transform(self):
        """
        Convert to a DataFrame and add numerical metadata fields.
        (Question length + number of answer options)
        """
        print("🔧 Transforming dataset...")

        # Convert dataset to dataframe
        df = pd.DataFrame(self.raw)

        # Add engineered columns
        df["len_question"] = df["question"].map(lambda x: len(x))
        df["choice_count"] = df["choices"].map(lambda x: len(x))

        # Assign to instance attribute
        self.table = df

        # Display useful information for report writing
        print("\n=== Summary Metrics ===")
        print(df[["len_question", "choice_count"]].describe())
        print("=======================\n")

    def plot(self):
        """Create a clean histogram of question lengths and save it."""
        print("📈 Creating visual plot...")

        sns.set_style("ticks")
        fig, ax = plt.subplots(figsize=(10, 6))

        sns.histplot(
            data=self.table,
            x="len_question",
            bins=45,
            kde=True,
            color="#3498db",
            edgecolor="black",
            ax=ax
        )

        avg_len = self.table["len_question"].mean()

        ax.axvline(avg_len, color="crimson", linestyle="--", linewidth=1.4,
                   label=f"Average Length: {avg_len:.1f}")

        ax.set_title("ScienceQA Question Length Distribution", fontsize=16, weight="bold")
        ax.set_xlabel("Number of Characters")
        ax.set_ylabel("Count")
        ax.legend()

        plt.tight_layout()
        plt.savefig(OUTPUT_FILE, dpi=300)
        print(f"📁 Plot saved at: {os.path.abspath(OUTPUT_FILE)}")

        plt.show()


# -------------------- Workflow Execution --------------------
if __name__ == "__main__":
    handler = ScienceQALoader()
    handler.fetch()
    handler.transform()
    handler.plot()
