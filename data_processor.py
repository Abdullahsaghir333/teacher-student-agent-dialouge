import os
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from datasets import load_dataset

# Configuration
DATASET_NAME = "derek-thomas/ScienceQA"
OUTPUT_IMAGE = "dataset_visualization.png"

class DataProcessor:
    """
    Handles loading, preprocessing, and visualization of the ScienceQA dataset.
    Designed to meet Rubric Point: 'DataSet - Properly loading, preprocessing and visualizations'.
    """
    def __init__(self, dataset_name=DATASET_NAME):
        self.dataset_name = dataset_name
        self.dataset = None
        self.df = None

    def load_data(self):
        """Loads the dataset from Hugging Face."""
        print(f"📥 Loading dataset: {self.dataset_name}...")
        try:
            # We load the 'train' split for analysis as it has the most data
            self.dataset = load_dataset(self.dataset_name, split="train")
            print(f"✅ Loaded {len(self.dataset)} examples successfully.")
        except Exception as e:
            print(f"❌ Error loading dataset: {e}")
            exit()

    def preprocess_data(self):
        """
        Preprocessing: Converts to Pandas DataFrame and calculates metadata.
        We analyze 'Question Length' to understand the token distribution.
        """
        print("⚙️ Preprocessing data...")
        # Convert to Pandas for easier analysis
        self.df = pd.DataFrame(self.dataset)
        
        # 1. Feature Engineering: Calculate character length of questions
        self.df['question_length'] = self.df['question'].apply(len)
        
        # 2. Feature Engineering: Count number of choices available
        self.df['num_choices'] = self.df['choices'].apply(len)
        
        # Print summary stats to console (Useful for your report text)
        print("\n--- Dataset Statistics ---")
        print(self.df[['question_length', 'num_choices']].describe())
        print("--------------------------\n")

    def visualize_data(self):
        """
        Visualization: Creates a histogram of question lengths.
        Saves the plot to a file for the report.
        """
        print("📊 Generating Visualization...")
        
        # Set the style to look professional (Industry Standard)
        sns.set_theme(style="whitegrid")
        plt.figure(figsize=(10, 6))

        # Create Histogram
        sns.histplot(
            self.df['question_length'], 
            bins=40, 
            kde=True, 
            color='#2ecc71', # Nice emerald green
            edgecolor='black'
        )

        # Add Labels and Title
        plt.title('Distribution of Science Question Lengths', fontsize=16, fontweight='bold')
        plt.xlabel('Character Length', fontsize=12)
        plt.ylabel('Frequency', fontsize=12)
        plt.axvline(self.df['question_length'].mean(), color='red', linestyle='--', label=f"Mean Length: {self.df['question_length'].mean():.1f}")
        plt.legend()

        # Save to disk
        plt.tight_layout()
        plt.savefig(OUTPUT_IMAGE, dpi=300)
        print(f"✅ Visualization saved to '{os.path.abspath(OUTPUT_IMAGE)}'")
        plt.show()

# --- Main Execution Flow ---
if __name__ == "__main__":
    processor = DataProcessor()
    processor.load_data()
    processor.preprocess_data()
    processor.visualize_data()