```python
import json
import os
from tqdm import tqdm
import argparse

# --- 1. Configure paths (updated for Disruption task) ---

# Path to the model recall results file
RESULTS_FILE_PATH = "results/queries_task1_disruptive.json"

# Path to the query file
QUERIES_FILE_PATH = "results/queries_task1_disruptive.json"

# Directory containing Ground Truth files
GROUND_TRUTH_DIR = "results/disruption"


# --- 2. Main evaluation logic ---
def main(recall_k):
    """
    Main function for evaluating disruptive recall (Recall@K).
    """
    # Check if the core files and directory exist
    if not os.path.exists(QUERIES_FILE_PATH):
        print(f"Error: Query file not found: {QUERIES_FILE_PATH}")
        return
    if not os.path.exists(RESULTS_FILE_PATH):
        print(f"Error: Results file not found: {RESULTS_FILE_PATH}")
        return
    if not os.path.isdir(GROUND_TRUTH_DIR):
        print(f"Error: Ground Truth directory not found: {GROUND_TRUTH_DIR}")
        return

    print(f"--- Starting evaluation of disruptive recall quality (Recall@{recall_k}) ---")

    # Load queries and recall results
    try:
        with open(QUERIES_FILE_PATH, 'r', encoding='utf-8') as f:
            queries_data = json.load(f)
        with open(RESULTS_FILE_PATH, 'r', encoding='utf-8') as f:
            results_data = json.load(f)
    except Exception as e:
        print(f"Error: Failed to read core query or result file: {e}")
        return

    # Initialize statistics variables
    all_query_recalls = []
    total_queries_evaluated = 0

    # Iterate through all queries
    for field_name, query_text in tqdm(queries_data.items(), desc="Evaluating queries"):
        
        # 1. Locate and load the Ground Truth file
        gt_file_path = os.path.join(GROUND_TRUTH_DIR, f"{field_name}.json")
        
        if not os.path.exists(gt_file_path):
            print(f"\nWarning: Ground Truth file for field '{field_name}' not found, skipping. Path: {gt_file_path}")
            continue

        try:
            with open(gt_file_path, 'r', encoding='utf-8') as f:
                gt_data = json.load(f)
        except json.JSONDecodeError:
            print(f"\nWarning: JSON parsing failed, file may be corrupted, skipping. File: {gt_file_path}")
            continue 
        except Exception as e:
            print(f"\nWarning: Unknown error occurred while reading Ground Truth file, skipping. Error: {e}")
            continue

        # 2. Extract Top-K Ground Truth IDs (assuming GT file is sorted in descending order of disruption)
        top_k_gt_papers = gt_data[:recall_k]
        ground_truth_ids_set = {paper['id'] for paper in top_k_gt_papers if 'id' in paper}

        # 3. Get recall results and compute hits
        recalled_papers = results_data.get(query_text)
        if recalled_papers is None:
            print(f"\nWarning: Recall list for query '{query_text[:50]}...' not found in results file, skipping.")
            continue
        
        hits = 0
        for paper in recalled_papers:
            if paper.get("id") in ground_truth_ids_set:
                hits += 1
        
        # 4. Compute Recall@K for the current query
        recall_for_query = hits / recall_k if recall_k > 0 else 0.0
        all_query_recalls.append(recall_for_query)
        total_queries_evaluated += 1

    # --- 3. Compute and print the final evaluation report ---
    if total_queries_evaluated > 0:
        # Compute macro-average Recall
        macro_average_recall = sum(all_query_recalls) / len(all_query_recalls)

        print(f"\n--- Final Evaluation Report (Disruption Recall@{recall_k}) ---")
        print(f"Total queries successfully evaluated: {total_queries_evaluated}")
        print(f"Evaluation criteria: Whether the recalled papers hit the Top {recall_k} most disruptive papers in each field")
        print("-" * 50)
        print(f"Macro-Average Recall@{recall_k}: {macro_average_recall:.2%}")
        print("-" * 50)

    else:
        print("No queries were successfully evaluated. Please check whether the file paths and contents match.")

if __name__ == "__main__":
    # --- Command-line argument parsing ---
    parser = argparse.ArgumentParser(description="Evaluate disruption recall rate Recall@K.")
    parser.add_argument(
        '--k',
        type=int,
        default=50,
        help="Top-K value used to compute Recall (default: 50)"
    )
    args = parser.parse_args()
    
    # Pass the parsed k value to the main function
    main(args.k)
```
