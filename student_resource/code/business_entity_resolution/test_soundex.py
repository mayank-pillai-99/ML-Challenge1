import pandas as pd
import os
import re

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

def run_test():
    data_dir = "../../dataset/train"
    df_s1 = pd.read_csv(os.path.join(data_dir, "train_source1.tsv"), sep="\t").head(5000)
    df_s2 = pd.read_csv(os.path.join(data_dir, "train_source2.tsv"), sep="\t")
    df_s3 = pd.read_csv(os.path.join(data_dir, "train_source3.tsv"), sep="\t")
    df_gt = pd.read_csv(os.path.join(data_dir, "train_ground_truth.tsv"), sep="\t")
    
    gt_dict = {}
    for _, row in df_gt.iterrows():
        gt_dict[row['source1_entity_id']] = set(row['matched_entity_ids'].split(',')) if not pd.isna(row['matched_entity_ids']) else set()
        
    from collections import defaultdict
    buckets = defaultdict(list)
    for df in [df_s2, df_s3]:
        for id_val, name, addr, country in zip(df['entity_id'], df['business_name'], df['business_address'], df['country']):
            keys = get_blocking_keys(name, addr, country)
            for key in keys:
                buckets[key].append(id_val)
                
    for k in [k for k, v in buckets.items() if len(v) > 20000]:
        del buckets[k]
        
    tp = 0
    fn = 0
    for id_val, name, addr, country in zip(df_s1['entity_id'], df_s1['business_name'], df_s1['business_address'], df_s1['country']):
        keys = get_blocking_keys(name, addr, country)
        cand_list = set()
        for key in keys:
            cand_list.update(buckets.get(key, []))
            
        true_matches = gt_dict.get(id_val, set())
        tp += len(true_matches.intersection(cand_list))
        fn += len(true_matches - cand_list)
        
    print(f"Recall: {tp / (tp+fn) if tp+fn>0 else 0}")
    
run_test()
