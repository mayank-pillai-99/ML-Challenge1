import pandas as pd
import torch
import os
import random
from sentence_transformers import CrossEncoder

def test_t():
    print("Loading gt...")
    df_gt = pd.read_csv("../../dataset/train/train_ground_truth.tsv", sep="\t")
    gt_dict = {}
    for _, row in df_gt.iterrows():
        gt_dict[row['source1_entity_id']] = set(row['matched_entity_ids'].split(',')) if not pd.isna(row['matched_entity_ids']) else set()
        
    print("Loading candidates...")
    cand_file = "../../output/candidate_pairs.tsv"
    # Actually we don't have train candidates for inverted index generated yet. 
