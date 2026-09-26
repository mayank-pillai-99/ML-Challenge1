import pandas as pd
import re
import os
import gc
from collections import defaultdict

def extract_words(text):
    text = str(text).lower()
    return re.findall(r'\b[a-z0-9]+\b', text)

def extract_zipcode(address):
    # US zipcodes (5 digits) or Indian PIN codes (6 digits)
    address = str(address)
    match = re.search(r'\b\d{5,6}\b', address)
    if match:
        return match.group(0)
    return "no_zip"

def get_blocking_keys(name, address, country):
    country = str(country).lower()
    name_words = extract_words(name)
    addr_words = extract_words(address)
    
    stopwords = {'the', 'a', 'an', 'and', 'of', 'in', 'to', 'for', 'with', 'inc', 'co', 'ltd', 'corp', 'llc', 'company', 'pvt', 'private'}
    filtered_name_words = [w for w in name_words if w not in stopwords and len(w) > 2]
    
    keys = []
    
    # PASS 1: First meaningful word of name + country
    if filtered_name_words:
        keys.append(f"p1_{country}_{filtered_name_words[0]}")
    elif name_words:
        keys.append(f"p1_{country}_{name_words[0]}")
        
    # PASS 2: Zip code + country (Very strong signal if present)
    zipcode = extract_zipcode(address)
    if zipcode != "no_zip":
        keys.append(f"p2_{country}_{zipcode}")
        
    # PASS 3: First 3 chars of name + First 3 chars of address
    name_prefix = "".join(name_words)[:3] if name_words else "noname"
    addr_prefix = "".join(addr_words)[:3] if addr_words else "noaddr"
    if name_prefix != "noname" and addr_prefix != "noaddr":
        keys.append(f"p3_{country}_{name_prefix}_{addr_prefix}")
        
    # PASS 4: Last meaningful word of name + country (catches word order flips)
    if len(filtered_name_words) > 1:
        keys.append(f"p4_{country}_{filtered_name_words[-1]}")
        
    return keys

def run_multipass_blocker():
    data_dir = "../../dataset/train"
    out_dir = "../../output"
    os.makedirs(out_dir, exist_ok=True)
    
    print("1. Loading datasets...")
    df_s1 = pd.read_csv(os.path.join(data_dir, "train_source1.tsv"), sep="\t")
    df_s2 = pd.read_csv(os.path.join(data_dir, "train_source2.tsv"), sep="\t")
    df_s3 = pd.read_csv(os.path.join(data_dir, "train_source3.tsv"), sep="\t")
    
    print("2. Generating blocking keys and grouping Candidate Data (S2 & S3)...")
    candidate_buckets = defaultdict(list)
    
    def process_candidates(df):
        for id_val, name, addr, country in zip(df['entity_id'], df['business_name'], df['business_address'], df['country']):
            keys = get_blocking_keys(name, addr, country)
            for key in keys:
                candidate_buckets[key].append(id_val)
                
    process_candidates(df_s2)
    del df_s2
    gc.collect()

    process_candidates(df_s3)
    del df_s3
    gc.collect()

    print(f"Created {len(candidate_buckets)} unique buckets across 4 passes.")

    print("Pruning mega-buckets to speed up search...")
    keys_to_delete = [k for k, v in candidate_buckets.items() if len(v) > 500]
    for k in keys_to_delete:
        del candidate_buckets[k]
    print(f"Deleted {len(keys_to_delete)} extremely generic buckets.")

    print("3. Finding candidates for Source 1...")
    results_s1 = []
    results_candidates = []
    
    for id_val, name, addr, country in zip(df_s1['entity_id'], df_s1['business_name'], df_s1['business_address'], df_s1['country']):
        keys = get_blocking_keys(name, addr, country)
        
        # Use a set to take the UNION of all candidates found across passes (removes duplicates)
        matches = set()
        for key in keys:
            matches.update(candidate_buckets.get(key, []))
            
        # We cap at 100 to avoid runaway buckets (e.g. if a zip code has 5000 businesses)
        match_list = list(matches)[:100]
            
        results_s1.append(id_val)
        results_candidates.append(",".join(match_list))

    print("4. Saving candidate pairs to output/...")
    output_df = pd.DataFrame({
        'source1_entity_id': results_s1,
        'candidate_entity_ids': results_candidates
    })
    
    output_path = os.path.join(out_dir, "candidate_pairs.tsv")
    output_df.to_csv(output_path, sep="\t", index=False)
    
    print(f"Done! Candidate pairs saved to {output_path}")

if __name__ == "__main__":
    run_multipass_blocker()
