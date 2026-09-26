import pandas as pd
import numpy as np
import os
from sklearn.feature_extraction.text import TfidfVectorizer

def test_tfidf():
    data_dir = "../../dataset/train"
    print("Loading data...")
    df_s1 = pd.read_csv(os.path.join(data_dir, "train_source1.tsv"), sep="\t")
    df_s2 = pd.read_csv(os.path.join(data_dir, "train_source2.tsv"), sep="\t")
    df_s3 = pd.read_csv(os.path.join(data_dir, "train_source3.tsv"), sep="\t")
    df_gt = pd.read_csv(os.path.join(data_dir, "train_ground_truth.tsv"), sep="\t")
    
    # Just take 10k from S1 to test recall quickly
    df_s1 = df_s1.head(10000)
    s1_ids = set(df_s1['entity_id'])
    
    gt_dict = {}
    for _, row in df_gt.iterrows():
        if row['source1_entity_id'] in s1_ids:
            if not pd.isna(row['matched_entity_ids']):
                gt_dict[row['source1_entity_id']] = set(row['matched_entity_ids'].split(','))
            else:
                gt_dict[row['source1_entity_id']] = set()
                
    total_true_matches = sum(len(v) for v in gt_dict.values())
    
    print("Preparing text...")
    def prep(df):
        return (df['business_name'].fillna("").astype(str) + " " + df['business_address'].fillna("").astype(str) + " " + df['country'].fillna("").astype(str)).str.lower()
        
    s1_text = prep(df_s1)
    s23_df = pd.concat([df_s2, df_s3], ignore_index=True)
    s23_text = prep(s23_df)
    s23_ids = s23_df['entity_id'].values
    
    print("Vectorizing...")
    vec = TfidfVectorizer(analyzer='word', ngram_range=(1, 2))
    s23_tfidf = vec.fit_transform(s23_text)
    s1_tfidf = vec.transform(s1_text)
    
    print("Computing dot product...")
    K = 10
    found = 0
    
    chunk_size = 1000
    for i in range(0, s1_tfidf.shape[0], chunk_size):
        end = min(i + chunk_size, s1_tfidf.shape[0])
        chunk = s1_tfidf[i:end]
        sims = chunk.dot(s23_tfidf.T)
        
        for row_idx in range(sims.shape[0]):
            s1_id = df_s1.iloc[i + row_idx]['entity_id']
            if len(gt_dict[s1_id]) == 0: continue
            
            row = sims.getrow(row_idx)
            if len(row.data) == 0: continue
                
            top_k_idx = row.indices[np.argsort(-row.data)[:K]]
            candidate_ids = set(s23_ids[top_k_idx])
            
            true_ids = gt_dict[s1_id]
            found += len(true_ids.intersection(candidate_ids))
            
    print(f"Total true matches: {total_true_matches}")
    print(f"Found matches: {found}")
    if total_true_matches > 0:
        print(f"Recall: {found/total_true_matches:.4f}")

if __name__ == '__main__':
    test_tfidf()
