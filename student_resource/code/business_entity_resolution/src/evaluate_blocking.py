import pandas as pd
import os

def evaluate_recall():
    data_dir = "../../dataset/train"
    out_dir = "../../output"
    
    print("Loading Ground Truth...")
    df_gt = pd.read_csv(os.path.join(data_dir, "train_ground_truth.tsv"), sep="\t")
    
    print("Loading Generated Candidates...")
    df_candidates = pd.read_csv(os.path.join(out_dir, "candidate_pairs.tsv"), sep="\t")
    
    # Create dictionaries for fast lookup
    gt_dict = dict(zip(df_gt['source1_entity_id'], df_gt['matched_entity_ids']))
    cand_dict = dict(zip(df_candidates['source1_entity_id'], df_candidates['candidate_entity_ids']))
    
    total_true_matches = 0
    total_found_matches = 0
    
    print("Calculating Recall...")
    for s1_id, true_matches_str in gt_dict.items():
        if pd.isna(true_matches_str) or not true_matches_str:
            continue
            
        true_matches = set(true_matches_str.split(','))
        total_true_matches += len(true_matches)
        
        cand_matches_str = cand_dict.get(s1_id, "")
        if pd.isna(cand_matches_str) or not cand_matches_str:
            cand_matches = set()
        else:
            cand_matches = set(cand_matches_str.split(','))
            
        found_matches = true_matches.intersection(cand_matches)
        total_found_matches += len(found_matches)
        
    recall = total_found_matches / total_true_matches if total_true_matches > 0 else 0
    
    print("--- BLOCKING ENGINE RECALL ---")
    print(f"Total True Match Pairs:  {total_true_matches}")
    print(f"Matches Found by Bucket: {total_found_matches}")
    print(f"Recall: {recall:.4f} ({recall*100:.2f}%)")

if __name__ == "__main__":
    evaluate_recall()
