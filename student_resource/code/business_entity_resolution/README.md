# Business Entity Resolution Pipeline

## Overview
This pipeline uses a Multi-Pass Bucket Hashing blocker for high-recall candidate generation, followed by a fine-tuned HuggingFace Cross-Encoder (`ms-marco-MiniLM-L-6-v2`) to accurately classify candidate pairs.

## How to Reproduce End-to-End

1. **Install Requirements:**
   ```bash
   pip install -r requirements.txt
   ```

2. **Train the Model:**
   Generates training candidate pairs, builds the labeled dataset, and fine-tunes the CrossEncoder.
   ```bash
   python3 src/train_model.py
   ```
   *Note: This will save the fine-tuned model to `./business_entity_model`.*

3. **Run Inference:**
   Runs blocking on the test set (or uses cached candidate pairs if already generated) and scores them in checkpoints using the fine-tuned model.
   ```bash
   python3 src/inference.py
   ```
   *Note: This generates `candidate_pairs.tsv` and `matching_results.tsv` in the `../../output/` directory.*

4. **Validate Output:**
   Run the validator script from the root `student_resource` directory:
   ```bash
   python3 utils/validate_submission.py --matching output/matching_results.tsv --candidate output/candidate_pairs.tsv --test-dir dataset/test
   ```
