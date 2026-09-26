import pandas as pd
import os

print("Fixing candidate_pairs.tsv...")
out_dir = "../../output"
test_s1 = pd.read_csv("../../dataset/test/test_source1.tsv", sep="\t")
cand_df = pd.read_csv(os.path.join(out_dir, "candidate_pairs.tsv"), sep="\t", dtype=str)

all_ids = test_s1[['entity_id']].rename(columns={'entity_id': 'source1_entity_id'})
fixed_cand = pd.merge(all_ids, cand_df, on='source1_entity_id', how='left')
fixed_cand['candidate_entity_ids'] = fixed_cand['candidate_entity_ids'].fillna("")

fixed_cand.to_csv(os.path.join(out_dir, "candidate_pairs.tsv"), sep="\t", index=False)
print("Done! Validating again...")
