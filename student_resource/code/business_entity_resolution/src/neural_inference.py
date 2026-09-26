import pandas as pd
import os
import gc
import torch
from sentence_transformers import SentenceTransformer, CrossEncoder, util
import numpy as np
from collections import defaultdict

def run_neural_pipeline():
    data_dir = "../../dataset/test"
    out_dir = "../../output"
    os.makedirs(out_dir, exist_ok=True)
    
    print("==================================================")
    print(" V6: NEURAL SEMANTIC SEARCH (DUAL-ENCODER + FAISS) ")
    print("==================================================")
    
    print("\n1. Loading 11 Million Test Datasets...")
    df_s1 = pd.read_csv(os.path.join(data_dir, "test_source1.tsv"), sep="\t")
    df_s2 = pd.read_csv(os.path.join(data_dir, "test_source2.tsv"), sep="\t")
    df_s3 = pd.read_csv(os.path.join(data_dir, "test_source3.tsv"), sep="\t")
    
    df_s1['text'] = df_s1['business_name'].fillna("") + " " + df_s1['business_address'].fillna("") + " " + df_s1['country'].fillna("")
    
    df_s23 = pd.concat([df_s2, df_s3], ignore_index=True)
    del df_s2, df_s3
    gc.collect()
    df_s23['text'] = df_s23['business_name'].fillna("") + " " + df_s23['business_address'].fillna("") + " " + df_s23['country'].fillna("")
    
    s23_ids = df_s23['entity_id'].values
    s1_dict = dict(zip(df_s1['entity_id'], df_s1['text'].str.lower()))
    s23_dict = dict(zip(df_s23['entity_id'], df_s23['text'].str.lower()))
    
    cand_file = os.path.join(out_dir, "neural_candidate_pairs.tsv")
    s1_candidates = {}
    
    device = 'mps' if torch.backends.mps.is_available() else 'cpu'
    
    if os.path.exists(cand_file):
        print(f"\n[SKIP] Found existing {cand_file}! Loading to save 4 hours...")
        df_cand = pd.read_csv(cand_file, sep="\t")
        df_cand['candidate_entity_ids'] = df_cand['candidate_entity_ids'].fillna("")
        for s1_id, c_str in zip(df_cand['source1_entity_id'], df_cand['candidate_entity_ids']):
            s1_candidates[s1_id] = c_str.split(",") if c_str else []
        del df_cand
    else:
        print("\n2. STAGE 1: Generating Semantic Vectors (Bi-Encoder)...")
        print("Loading all-MiniLM-L6-v2 (Fast & Highly Semantic)...")
        bi_encoder = SentenceTransformer('all-MiniLM-L6-v2', device=device)
        
        print("\nEncoding 10 Million Source 2/3 Businesses...")
        print("(This will take ~2-3 hours and run perfectly on your Mac GPU)")
        s23_texts = df_s23['text'].tolist()
        
        # We process in large batches to keep RAM safe
        corpus_embeddings = bi_encoder.encode(s23_texts, batch_size=1024, show_progress_bar=True, convert_to_tensor=True)
        del s23_texts, df_s23
        gc.collect()
        
        print("\nEncoding 1.7 Million Source 1 Businesses...")
        s1_texts = df_s1['text'].tolist()
        queries_embeddings = bi_encoder.encode(s1_texts, batch_size=1024, show_progress_bar=True, convert_to_tensor=True)
        del s1_texts
        gc.collect()
        
        print("\n3. STAGE 2: PyTorch KNN Semantic Search...")
        print("Finding the 5 closest mathematical vectors for every business...")
        
        candidate_tsv_rows = []
        chunk_size = 50000 
        
        for i in range(0, len(queries_embeddings), chunk_size):
            chunk = queries_embeddings[i:i+chunk_size]
            # PyTorch accelerated Cosine Similarity Nearest Neighbors
            hits = util.semantic_search(chunk, corpus_embeddings, top_k=5)
            
            for j, hit_list in enumerate(hits):
                s1_idx = i + j
                s1_id = df_s1.iloc[s1_idx]['entity_id']
                
                best_matches = [s23_ids[hit['corpus_id']] for hit in hit_list]
                s1_candidates[s1_id] = best_matches
                candidate_tsv_rows.append({'source1_entity_id': s1_id, 'candidate_entity_ids': ",".join(best_matches)})
                
            print(f"Nearest Neighbor Search: {min(i+chunk_size, len(queries_embeddings))} / {len(queries_embeddings)} completed...")
            
        pd.DataFrame(candidate_tsv_rows).to_csv(cand_file, sep="\t", index=False)
        print("Neural Candidate Generation Complete!")
        
        del corpus_embeddings, queries_embeddings, candidate_tsv_rows, bi_encoder
        gc.collect()
        
    print("\n4. STAGE 3: Final CrossEncoder Scoring...")
    model_path = os.path.abspath('business_entity_model')
    print(f"Loading Fine-Tuned CrossEncoder: {model_path}")
    cross_model = CrossEncoder(model_path, device=device)
    
    THRESHOLD = 0.95
    out_file = os.path.join(out_dir, "neural_matching_results.tsv")
    
    s1_ids = list(df_s1['entity_id'])
    pd.DataFrame(columns=['source1_entity_id', 'matched_entity_ids']).to_csv(out_file, sep="\t", index=False)
        
    checkpoint_size = 50000
    for i in range(0, len(s1_ids), checkpoint_size):
        chunk_s1_ids = s1_ids[i:i+checkpoint_size]
        all_pairs = []
        pair_meta = []
        
        for s1_id in chunk_s1_ids:
            candidates = s1_candidates.get(s1_id, [])
            s1_text = s1_dict[s1_id]
            for c in candidates:
                all_pairs.append([s1_text, s23_dict[c]])
                pair_meta.append((s1_id, c))
                
        print(f"Scoring Neural Candidates chunk {i} to {i+len(chunk_s1_ids)} (Pairs: {len(all_pairs)})...")
        
        scores = []
        if all_pairs:
            scores = cross_model.predict(all_pairs, batch_size=256, show_progress_bar=False)
            
        match_dict = defaultdict(list)
        for (s1_id, c), score in zip(pair_meta, scores):
            if score > THRESHOLD:
                match_dict[s1_id].append(c)
                
        chunk_results = []
        for s1_id in chunk_s1_ids:
            matches = match_dict.get(s1_id, [])
            chunk_results.append({'source1_entity_id': s1_id, 'matched_entity_ids': ",".join(matches)})
            
        pd.DataFrame(chunk_results).to_csv(out_file, sep="\t", mode='a', header=False, index=False)
        
    print("\n==================================================")
    print(" V6 NEURAL PIPELINE FINISHED COMPLETELY! ")
    print("==================================================")

if __name__ == "__main__":
    run_neural_pipeline()
