import pandas as pd
import os

def load_and_explore():
    data_dir = "../../dataset/train"
    
    print("Loading datasets...")
    # Read files with tab separator
    df_s1 = pd.read_csv(os.path.join(data_dir, "train_source1.tsv"), sep="\t")
    df_s2 = pd.read_csv(os.path.join(data_dir, "train_source2.tsv"), sep="\t")
    df_s3 = pd.read_csv(os.path.join(data_dir, "train_source3.tsv"), sep="\t")
    df_gt = pd.read_csv(os.path.join(data_dir, "train_ground_truth.tsv"), sep="\t")
    
    print(f"Source 1 shape: {df_s1.shape}")
    print(f"Source 2 shape: {df_s2.shape}")
    print(f"Source 3 shape: {df_s3.shape}")
    print(f"Ground Truth shape: {df_gt.shape}")
    
    print("\n--- Source 1 Head ---")
    print(df_s1.head(3))
    
    print("\n--- Source 2 Head ---")
    print(df_s2.head(3))
    
    print("\n--- Ground Truth Head ---")
    print(df_gt.head(3))
    
    print("\n--- Missing Values ---")
    print("Source 1 nulls:\n", df_s1.isnull().sum())
    print("\nSource 2 nulls:\n", df_s2.isnull().sum())
    
    print("\n--- Country Distribution Source 1 ---")
    print(df_s1['country'].value_counts())

if __name__ == "__main__":
    load_and_explore()
