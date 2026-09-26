import pandas as pd
import re
import os
import gc
import torch
from sentence_transformers import CrossEncoder
from collections import defaultdict

def soundex(name):
    if not name: return ""
    name = str(name).upper()
    name = re.sub(r'[^A-Z]', '', name)
    if not name: return ""
    
    sdx = name[0]
    dictionary = {"BFPV": "1", "CGJKQSXZ": "2", "DT": "3", "L": "4", "MN": "5", "R": "6", "AEIOUHWY": "."}
    for char in name[1:]:
        for key in dictionary.keys():
            if char in key:
                code = dictionary[key]
                if code != '.' and code != sdx[-1]:
                    sdx += code
    sdx = sdx.replace(".", "")
    return sdx[:4].ljust(4, "0")

def extract_words(text):
    text = str(text).lower()
    return re.findall(r'\b[a-z0-9]+\b', text)

def extract_zipcode(address):
    address = str(address)
    match = re.search(r'\b\d{5,6}\b', address)
    if match: return match.group(0)
    return "no_zip"

def get_blocking_keys(name, address, country):
    country = str(country).lower()
    name_words = extract_words(name)
    zipcode = extract_zipcode(address)
    stopwords = {'the', 'a', 'an', 'and', 'of', 'in', 'to', 'for', 'with', 'inc', 'co', 'ltd', 'corp', 'llc', 'company', 'pvt', 'private'}
    
    words = [w for w in name_words if w not in stopwords]
    keys = []
    
    if len(words) >= 1:
        w1_sdx = soundex(words[0])
        if zipcode != "no_zip":
            keys.append(f"{country}_{w1_sdx}_{zipcode}")
            
        if len(words) >= 2:
            w2_sdx = soundex(words[1])
            keys.append(f"{country}_{w1_sdx}_{w2_sdx}")
            
        keys.append(f"{country}_{w1_sdx}")
        
    return list(set(keys))

def run_inference():
    data_dir = "../../dataset/test"
    out_dir = "../../output"
    
    print("1. Loading Test Datasets...")
    df_s1 = pd.read_csv(os.path.join(data_dir, "test_source1.tsv"), sep="\t")
    df_s2 = pd.read_csv(os.path.join(data_dir, "test_source2.tsv"), sep="\t")
    df_s3 = pd.read_csv(os.path.join(data_dir, "test_source3.tsv"), sep="\t")
    
    print("Building Lookup Dictionaries...")
    df_s1['text'] = df_s1['business_name'].fillna("") + " " + df_s1['business_address'].fillna("") + " " + df_s1['country'].fillna("")
    s1_dict = dict(zip(df_s1['entity_id'], df_s1['text'].str.lower()))
    
    s23_dict = {}
    for df in [df_s2, df_s3]:
        for id_val, name, addr, country in zip(df['entity_id'], df['business_name'], df['business_address'], df['country']):
            s23_dict[id_val] = f"{name} {addr} {country}".lower()
            
    print("2. Test Candidate Generation (Soundex Phonetic Blocker)...")
    cand_file = os.path.join(out_dir, "candidate_pairs.tsv")
    s1_candidates = {}
    
    if os.path.exists(cand_file):
        print(f"Found existing candidate pairs! Loading {cand_file} to save time...")
        df_cand = pd.read_csv(cand_file, sep="\t")
        df_cand['candidate_entity_ids'] = df_cand['candidate_entity_ids'].fillna("")
        for s1_id, c_str in zip(df_cand['source1_entity_id'], df_cand['candidate_entity_ids']):
            s1_candidates[s1_id] = c_str.split(",") if c_str else []
        del df_cand
    else:
        print("Building Phonetic Buckets...")
        candidate_buckets = defaultdict(list)
        for df in [df_s2, df_s3]:
            for id_val, name, addr, country in zip(df['entity_id'], df['business_name'], df['business_address'], df['country']):
                keys = get_blocking_keys(name, addr, country)
                for key in keys:
                    candidate_buckets[key].append(id_val)
                    
        print("Pruning mega-buckets...")
        keys_to_delete = [k for k, v in candidate_buckets.items() if len(v) > 20000]
        for k in keys_to_delete: del candidate_buckets[k]
        
        def jaccard(set1, set2):
            if not set1 or not set2: return 0.0
            return len(set1.intersection(set2)) / float(len(set1.union(set2)))
            
        candidate_tsv_rows = []
        from collections import Counter
        
        for i, (id_val, name, addr, country) in enumerate(zip(df_s1['entity_id'], df_s1['business_name'], df_s1['business_address'], df_s1['country'])):
            keys = get_blocking_keys(name, addr, country)
            
            cand_list = []
            for key in keys: cand_list.extend(candidate_buckets.get(key, []))
            
            top_200_cands = [item for item, count in Counter(cand_list).most_common(200)]
            
            s1_words = set(extract_words(name) + extract_words(addr))
            
            scored_matches = []
            for match_id in top_200_cands:
                s23_words = set(s23_dict[match_id].split())
                score = jaccard(s1_words, s23_words)
                scored_matches.append((score, match_id))
                
            scored_matches.sort(reverse=True, key=lambda x: x[0])
            best_matches = [m[1] for m in scored_matches[:10]]
            
            s1_candidates[id_val] = best_matches
            candidate_tsv_rows.append({'source1_entity_id': id_val, 'candidate_entity_ids': ",".join(best_matches)})
            
            if i % 100000 == 0:
                print(f"Processed {i}/{len(df_s1)} entities...")
                
        pd.DataFrame(candidate_tsv_rows).to_csv(cand_file, sep="\t", index=False)
        print("candidate_pairs.tsv generated.")
        del candidate_buckets, candidate_tsv_rows
        
    del df_s2, df_s3
    gc.collect()

    print("3. Loading ML Model for Inference...")
    device = 'mps' if torch.backends.mps.is_available() else 'cpu'
    model_path = os.path.abspath('business_entity_model')
    model = CrossEncoder(model_path, device=device)
    
    print("4. Scoring Candidates in Checkpoints...")
    THRESHOLD = 0.95
    out_file = os.path.join(out_dir, "matching_results.tsv")
    
    s1_ids = list(df_s1['entity_id'])
    start_idx = 0
    
    if os.path.exists(out_file):
        try:
            existing_df = pd.read_csv(out_file, sep="\t")
            start_idx = len(existing_df)
            print(f"Found existing checkpoint! Resuming from entity index {start_idx} / {len(s1_ids)}...")
        except pd.errors.EmptyDataError:
            pd.DataFrame(columns=['source1_entity_id', 'matched_entity_ids']).to_csv(out_file, sep="\t", index=False)
    else:
        pd.DataFrame(columns=['source1_entity_id', 'matched_entity_ids']).to_csv(out_file, sep="\t", index=False)
        
    checkpoint_size = 50000
    
    for i in range(start_idx, len(s1_ids), checkpoint_size):
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
    run_inference()
