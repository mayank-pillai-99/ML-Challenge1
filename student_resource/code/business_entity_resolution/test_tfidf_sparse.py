import pandas as pd
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
import os
import gc

def run_tfidf():
    print("Loading data...")
    data_dir = "../../dataset/train"
    df_s1 = pd.read_csv(os.path.join(data_dir, "train_source1.tsv"), sep="\t").head(2000)
    df_s2 = pd.read_csv(os.path.join(data_dir, "train_source2.tsv"), sep="\t").head(1000000) # 1M for test
    
    df_s1['text'] = df_s1['business_name'].astype(str) + " " + df_s1['country'].astype(str)
    df_s2['text'] = df_s2['business_name'].astype(str) + " " + df_s2['country'].astype(str)
    
    print("Vectorizing...")
    vec = TfidfVectorizer(analyzer='char_wb', ngram_range=(3,3), max_features=100000)
    s2_matrix = vec.fit_transform(df_s2['text'])
    s1_matrix = vec.transform(df_s1['text'])
    
    s2_matrix_T = s2_matrix.T.tocsr()
    del s2_matrix
    gc.collect()
    
    print("Chunked Dot Product...")
    chunk_size = 500
    for i in range(0, s1_matrix.shape[0], chunk_size):
        chunk = s1_matrix[i:i+chunk_size]
        sim = chunk.dot(s2_matrix_T)
        print(f"Chunk {i} done. Sparsity: {sim.nnz}")
        
        # Test extraction for first row
        row = sim.getrow(0)
        if len(row.data) > 0:
            top_k = min(5, len(row.data))
            top_idx = np.argpartition(row.data, -top_k)[-top_k:]
            best_match_indices = row.indices[top_idx]
            
            # Sort them properly
            sorted_idx = np.argsort(row.data[top_idx])[::-1]
            best_match_indices = best_match_indices[sorted_idx]
            
            print(f"Top matches for S1[0]: {best_match_indices}")
        break

run_tfidf()
