import pandas as pd
import torch
import os
import random
from sentence_transformers import CrossEncoder
from blocking import get_blocking_keys, extract_words

def run_diagnostic():
    data_dir = "../../dataset/train"
    print("1. Loading Train Data for Diagnostic...")
    df_s1 = pd.read_csv(os.path.join(data_dir, "train_source1.tsv"), sep="\t").head(5000)
    df_s2 = pd.read_csv(os.path.join(data_dir, "train_source2.tsv"), sep="\t")
    df_s3 = pd.read_csv(os.path.join(data_dir, "train_source3.tsv"), sep="\t")
    df_gt = pd.read_csv(os.path.join(data_dir, "train_ground_truth.tsv"), sep="\t")
    
    gt_dict = {}
    for _, row in df_gt.iterrows():
        if pd.isna(row['matched_entity_ids']):
            gt_dict[row['source1_entity_id']] = set()
        else:
            gt_dict[row['source1_entity_id']] = set(row['matched_entity_ids'].split(','))
            
    df_s1['full_text'] = df_s1['business_name'].astype(str) + " " + df_s1['business_address'].astype(str) + " " + df_s1['country'].astype(str)
    s1_dict = dict(zip(df_s1['entity_id'], df_s1['full_text']))
    
    s23_dict = {}
    for df in [df_s2, df_s3]:
        for id_val, name, addr, country in zip(df['entity_id'], df['business_name'], df['business_address'], df['country']):
            s23_dict[id_val] = f"{name} {addr} {country}"
            
    print("2. Generating Inverted Index Candidates...")
    from collections import defaultdict, Counter
    candidate_buckets = defaultdict(list)
    for df in [df_s2, df_s3]:
        for id_val, name, addr, country in zip(df['entity_id'], df['business_name'], df['business_address'], df['country']):
            keys = get_blocking_keys(name, addr, country)
            for key in keys: candidate_buckets[key].append(id_val)
            
    # Keep buckets up to 50,000 to retain important words, but avoid massive stopword buckets
    keys_to_delete = [k for k, v in candidate_buckets.items() if len(v) > 50000]
    for k in keys_to_delete: del candidate_buckets[k]
    
    def jaccard(set1, set2):
        if not set1 or not set2: return 0.0
        return len(set1.intersection(set2)) / float(len(set1.union(set2)))
        
    s1_candidates = {}
    for i, (id_val, name, addr, country) in enumerate(zip(df_s1['entity_id'], df_s1['business_name'], df_s1['business_address'], df_s1['country'])):
        keys = get_blocking_keys(name, addr, country)
        
        cand_list = []
        for key in keys: cand_list.extend(candidate_buckets.get(key, []))
        
        # Get the top 200 candidates that share the most keys with S1
        top_200_cands = [item for item, count in Counter(cand_list).most_common(200)]
        
        s1_words = set(extract_words(name) + extract_words(addr))
        scored_matches = []
        for match_id in top_200_cands:
            s23_words = set(s23_dict[match_id].lower().split())
            score = jaccard(s1_words, s23_words)
            scored_matches.append((score, match_id))
        scored_matches.sort(reverse=True, key=lambda x: x[0])
        s1_candidates[id_val] = [m[1] for m in scored_matches[:10]]
        
    print("3. Scoring with Model...")
    device = 'mps' if torch.backends.mps.is_available() else 'cpu'
    model = CrossEncoder('./business_entity_model', device=device)
    
    all_pairs = []
    pair_meta = []
    for s1_id in df_s1['entity_id']:
        for c in s1_candidates.get(s1_id, []):
            all_pairs.append([s1_dict[s1_id], s23_dict[c]])
            pair_meta.append((s1_id, c))
            
    scores = model.predict(all_pairs, batch_size=256, show_progress_bar=False)
    
    for thresh in [0.1, 0.5, 0.9, 0.95, 0.98, 0.99, 0.999]:
        match_dict = defaultdict(list)
        for (s1_id, c), score in zip(pair_meta, scores):
            if score > thresh:
                match_dict[s1_id].append(c)
                
        tp = 0
        fp = 0
        fn = 0
        
        for s1_id in df_s1['entity_id']:
            true_matches = gt_dict.get(s1_id, set())
            pred_matches = set(match_dict.get(s1_id, []))
            
            tp += len(true_matches.intersection(pred_matches))
            fp += len(pred_matches - true_matches)
            fn += len(true_matches - pred_matches)
            
        precision = tp / (tp + fp) if (tp + fp) > 0 else 0
        recall = tp / (tp + fn) if (tp + fn) > 0 else 0
        f05 = (1.25 * precision * recall) / (0.25 * precision + recall) if (precision + recall) > 0 else 0
        
        print(f"Threshold: {thresh} | P: {precision:.4f} | R: {recall:.4f} | F0.5: {f05:.4f}")

if __name__ == '__main__':
    run_diagnostic()
