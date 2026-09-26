import pandas as pd
import os
import gc
import torch
from sentence_transformers import CrossEncoder
from collections import defaultdict

def run_scoring():
    data_dir = "../../dataset/test"
    out_dir = "../../output"
    
    print("1. Loading Test Datasets and Dictionaries...")
    df_s1 = pd.read_csv(os.path.join(data_dir, "test_source1.tsv"), sep="\t")
    df_s2 = pd.read_csv(os.path.join(data_dir, "test_source2.tsv"), sep="\t")
    df_s3 = pd.read_csv(os.path.join(data_dir, "test_source3.tsv"), sep="\t")
    
    df_s1['text'] = df_s1['business_name'].fillna("") + " " + df_s1['business_address'].fillna("") + " " + df_s1['country'].fillna("")
    s1_dict = dict(zip(df_s1['entity_id'], df_s1['text'].str.lower()))
    
    s23_dict = {}
    for df in [df_s2, df_s3]:
        for id_val, name, addr, country in zip(df['entity_id'], df['business_name'], df['business_address'], df['country']):
            s23_dict[id_val] = f"{name} {addr} {country}".lower()
            
    del df_s2, df_s3
    gc.collect()
    
    print("2. Loading Candidate Pairs (Limiting to Top 5 for Speed)...")
    cand_file = os.path.join(out_dir, "candidate_pairs.tsv")
    s1_candidates = {}
    
    df_cand = pd.read_csv(cand_file, sep="\t")
    df_cand['candidate_entity_ids'] = df_cand['candidate_entity_ids'].fillna("")
    for s1_id, c_str in zip(df_cand['source1_entity_id'], df_cand['candidate_entity_ids']):
        cands = c_str.split(",") if c_str else []
        # LIMIT TO 5 CANDIDATES MAX to keep PyTorch fast!
        s1_candidates[s1_id] = cands[:5]
    del df_cand
    gc.collect()

    print("3. Loading ML Model for Inference...")
    device = 'mps' if torch.backends.mps.is_available() else 'cpu'
    model_path = os.path.abspath('business_entity_model')
    model = CrossEncoder(model_path, device=device)
    
    print("4. Scoring Candidates in Checkpoints...")
    THRESHOLD = 0.95
    out_file = os.path.join(out_dir, "matching_results.tsv")
    
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
                
        print(f"Scoring checkpoint chunk {i} to {i+len(chunk_s1_ids)} (Pairs: {len(all_pairs)})...")
        
        scores = []
        if all_pairs:
            scores = model.predict(all_pairs, batch_size=256, show_progress_bar=False)
            
        match_dict = defaultdict(list)
        for (s1_id, c), score in zip(pair_meta, scores):
            if score > THRESHOLD:
                match_dict[s1_id].append(c)
                
        chunk_results = []
        for s1_id in chunk_s1_ids:
            matches = match_dict.get(s1_id, [])
            chunk_results.append({'source1_entity_id': s1_id, 'matched_entity_ids': ",".join(matches)})
            
        pd.DataFrame(chunk_results).to_csv(out_file, sep="\t", mode='a', header=False, index=False)
        
    print("matching_results.tsv fully generated in output/ directory.")

if __name__ == "__main__":
    run_scoring()
