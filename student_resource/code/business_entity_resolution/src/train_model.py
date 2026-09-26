import pandas as pd
import random
import os
import torch
from sentence_transformers import CrossEncoder, InputExample
from torch.utils.data import DataLoader
import math

def prepare_data():
    data_dir = "../../dataset/train"
    out_dir = "../../output"
    
    print("Loading data (This takes a moment)...")
    df_s1 = pd.read_csv(os.path.join(data_dir, "train_source1.tsv"), sep="\t")
    df_s2 = pd.read_csv(os.path.join(data_dir, "train_source2.tsv"), sep="\t")
    df_s3 = pd.read_csv(os.path.join(data_dir, "train_source3.tsv"), sep="\t")
    df_gt = pd.read_csv(os.path.join(data_dir, "train_ground_truth.tsv"), sep="\t")
    df_cand = pd.read_csv(os.path.join(out_dir, "candidate_pairs.tsv"), sep="\t")
    
    print("Creating lookup dictionaries...")
    df_s1['full_text'] = df_s1['business_name'].astype(str) + " " + df_s1['business_address'].astype(str) + " " + df_s1['country'].astype(str)
    s1_dict = dict(zip(df_s1['entity_id'], df_s1['full_text']))
    del df_s1
    
    df_s2['full_text'] = df_s2['business_name'].astype(str) + " " + df_s2['business_address'].astype(str) + " " + df_s2['country'].astype(str)
    s23_dict = dict(zip(df_s2['entity_id'], df_s2['full_text']))
    del df_s2
    
    df_s3['full_text'] = df_s3['business_name'].astype(str) + " " + df_s3['business_address'].astype(str) + " " + df_s3['country'].astype(str)
    s23_dict.update(dict(zip(df_s3['entity_id'], df_s3['full_text'])))
    del df_s3
        
    gt_dict = {}
    for _, row in df_gt.iterrows():
        if pd.isna(row['matched_entity_ids']):
            gt_dict[row['source1_entity_id']] = set()
        else:
            gt_dict[row['source1_entity_id']] = set(row['matched_entity_ids'].split(','))
            
    print("Building labeled dataset...")
    labeled_pairs = []
    
    # Sample a small training set to prove the pipeline works fast on M3 Mac
    pos_limit = 5000
    neg_limit = 10000
    pos_count = 0
    neg_count = 0
    
    df_cand_shuffled = df_cand.sample(frac=1.0, random_state=42)
    
    for _, row in df_cand_shuffled.iterrows():
        s1_id = row['source1_entity_id']
        cands_str = row['candidate_entity_ids']
        
        if pos_count >= pos_limit and neg_count >= neg_limit:
            break
            
        if pd.isna(cands_str) or not cands_str:
            continue
            
        true_matches = gt_dict.get(s1_id, set())
        cand_ids = str(cands_str).split(',')
        
        for cid in cand_ids:
            if cid not in s23_dict: continue
            
            s1_text = str(s1_dict[s1_id]).lower()
            s23_text = str(s23_dict[cid]).lower()
            
            if cid in true_matches:
                if pos_count < pos_limit:
                    labeled_pairs.append(InputExample(texts=[s1_text, s23_text], label=1.0))
                    pos_count += 1
            else:
                if neg_count < neg_limit:
                    labeled_pairs.append(InputExample(texts=[s1_text, s23_text], label=0.0))
                    neg_count += 1
                    
    print(f"Generated {pos_count} positives and {neg_count} negatives for training.")
    return labeled_pairs

def train_and_eval():
    labeled_pairs = prepare_data()
    
    random.seed(42)
    random.shuffle(labeled_pairs)
    
    train_size = int(0.8 * len(labeled_pairs))
    train_examples = labeled_pairs[:train_size]
    val_examples = labeled_pairs[train_size:]
    
    model_name = 'cross-encoder/ms-marco-MiniLM-L-6-v2'
    print(f"Loading pre-trained transformer: {model_name}...")
    
    # Use Mac's Neural Engine / GPU
    device = 'mps' if torch.backends.mps.is_available() else 'cpu'
    print(f"Hardware Acceleration active. Using device: {device}")
    
    model = CrossEncoder(model_name, num_labels=1, device=device)
    train_dataloader = DataLoader(train_examples, shuffle=True, batch_size=16)
    
    print("Training model... (1 Epoch for demonstration)")
    model.fit(
        train_dataloader=train_dataloader,
        epochs=1,
        warmup_steps=100,
        output_path='./business_entity_model'
    )
    
    print("Saving the fine-tuned model...")
    model.save('./business_entity_model')
    
    print("Evaluating on Validation Set to find optimal F0.5 threshold...")
    val_texts = [ex.texts for ex in val_examples]
    val_labels = [ex.label for ex in val_examples]
    
    preds = model.predict(val_texts)
    
    best_f05 = 0
    best_threshold = 0.5
    
    for thresh in [x/100.0 for x in range(10, 96, 5)]:
        y_pred = [1 if p > thresh else 0 for p in preds]
        
        tp = sum((p == 1 and l == 1.0) for p, l in zip(y_pred, val_labels))
        fp = sum((p == 1 and l == 0.0) for p, l in zip(y_pred, val_labels))
        fn = sum((p == 0 and l == 1.0) for p, l in zip(y_pred, val_labels))
        
        precision = tp / (tp + fp) if (tp + fp) > 0 else 0
        recall = tp / (tp + fn) if (tp + fn) > 0 else 0
        
        if precision + recall > 0:
            f05 = (1.25 * precision * recall) / (0.25 * precision + recall)
        else:
            f05 = 0
            
        if f05 > best_f05:
            best_f05 = f05
            best_threshold = thresh
            
    print(f"--- VALIDATION RESULTS ---")
    print(f"Optimal Threshold found at: {best_threshold}")
    print(f"Validation F0.5 Score: {best_f05:.4f}")
    
if __name__ == "__main__":
    train_and_eval()
