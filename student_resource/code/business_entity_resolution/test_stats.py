import pandas as pd
df_gt = pd.read_csv("../../dataset/train/train_ground_truth.tsv", sep="\t")
matches = df_gt['matched_entity_ids'].dropna()
print(f"Total S1 entities: {len(df_gt)}")
print(f"Entities with matches: {len(matches)}")
