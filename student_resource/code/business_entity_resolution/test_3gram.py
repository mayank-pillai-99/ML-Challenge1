import pandas as pd
import os
from blocking import extract_words

def run_3gram():
    data_dir = "../../dataset/train"
    df_s1 = pd.read_csv(os.path.join(data_dir, "train_source1.tsv"), sep="\t").head(5000)
    df_s2 = pd.read_csv(os.path.join(data_dir, "train_source2.tsv"), sep="\t")
    df_s3 = pd.read_csv(os.path.join(data_dir, "train_source3.tsv"), sep="\t")
    df_gt = pd.read_csv(os.path.join(data_dir, "train_ground_truth.tsv"), sep="\t")
    
    gt_dict = {}
    for _, row in df_gt.iterrows():
        gt_dict[row['source1_entity_id']] = set(row['matched_entity_ids'].split(',')) if not pd.isna(row['matched_entity_ids']) else set()
        
    def get_3grams(text):
        words = extract_words(text)
        trigrams = []
        for w in words:
            if len(w) >= 3:
                for i in range(len(w)-2):
                    trigrams.append(w[i:i+3])
        return trigrams

    print("Building 3-gram index...")
    from collections import defaultdict, Counter
    buckets = defaultdict(list)
    for df in [df_s2, df_s3]:
        for id_val, name in zip(df['entity_id'], df['business_name']):
            for tg in get_3grams(name):
                buckets[tg].append(id_val)
                
    for k in [k for k, v in buckets.items() if len(v) > 100000]:
        del buckets[k]
        
    print("Testing recall...")
    tp = 0
    fn = 0
    for id_val, name in zip(df_s1['entity_id'], df_s1['business_name']):
        cand_list = []
        for tg in get_3grams(name):
            cand_list.extend(buckets.get(tg, []))
            
        top_cands = set([item for item, count in Counter(cand_list).most_common(200)])
        true_matches = gt_dict.get(id_val, set())
        
        tp += len(true_matches.intersection(top_cands))
        fn += len(true_matches - top_cands)
        
    print(f"Recall: {tp / (tp+fn) if tp+fn>0 else 0}")
    
run_3gram()
