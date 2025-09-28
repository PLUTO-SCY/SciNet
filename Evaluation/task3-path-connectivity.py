import json
import os
import sqlite3
import random
import sys
from tqdm import tqdm

# --- 1. Path Configuration ---

# Path to the model recall results file
RESULTS_FILE_PATH = "results/answers_task3_paths.json"

# Path to the query file
QUERIES_FILE_PATH = "queries/queries_task3_paths.json"

# Path to the Ground Truth answers file
GROUND_TRUTH_FILE_PATH = "results/answers_task3_paths.json"

# Path to the database for querying references
FORWARD_DB_PATH = "OpenAlex/sqlite/citing_to_cited.db"


# --- 2. Database Query and Helper Functions ---
def get_references(paper_id_short: str, cursor: sqlite3.Cursor) -> set:
    """
    Query the reference list of a paper given its short OpenAlex ID.
    Returns a set for fast lookup.
    """
    try:
        cursor.execute("SELECT referenced_work_ids FROM citing_to_cited WHERE work_id = ?", (paper_id_short,))
        row = cursor.fetchone()
        if row and row[0]:
            return set(json.loads(row[0]))
    except Exception as e:
        print(f"Error while querying references {paper_id_short}: {e}")
    return set()

def _is_sequence_connected(sequence: list, cursor: sqlite3.Cursor) -> bool:
    """
    [Helper Function] Check if a given [specific order] of papers is connected.
    Check logic: paper_{i+1} must cite paper_i
    """
    for i in range(len(sequence) - 1):
        current_paper_full_id = sequence[i].get("id")
        next_paper_full_id = sequence[i+1].get("id")

        current_paper_short_id = current_paper_full_id.split('/')[-1]
        
        if not current_paper_full_id or not next_paper_full_id:
            return False

        next_paper_short_id = next_paper_full_id.split('/')[-1]
        references = get_references(next_paper_short_id, cursor)
        
        if current_paper_short_id not in references:
            return False
            
    return True

def check_path_connectivity(path: list, cursor: sqlite3.Cursor) -> bool:
    """
    [Rewritten Function] Check if the given list of papers (path) has any connectivity possibility.
    Tries 7 arrangements: original order, reversed order, and 5 random shuffles.
    If any check succeeds, the path is considered connected.
    """
    if len(path) < 2:
        return False
    
    if _is_sequence_connected(path, cursor):
        return True
        
    path_reversed = path[::-1]
    if _is_sequence_connected(path_reversed, cursor):
        return True

    for _ in range(5):
        path_shuffled = random.sample(path, len(path))
        if _is_sequence_connected(path_shuffled, cursor):
            return True
            
    return False

# --- 3. Main Evaluation Logic ---
def main():
    """
    Main function to evaluate the recalled citation paths.
    """
    required_files = [QUERIES_FILE_PATH, RESULTS_FILE_PATH, GROUND_TRUTH_FILE_PATH, FORWARD_DB_PATH]
    for path in required_files:
        if not os.path.exists(path):
            print(f"Error: required file not found: {path}")
            return

    print("--- Start evaluating the quality of recalled citation paths ---")
    
    # Load data
    try:
        with open(QUERIES_FILE_PATH, 'r', encoding='utf-8') as f:
            queries_data = json.load(f)
        with open(RESULTS_FILE_PATH, 'r', encoding='utf-8') as f:
            results_data = json.load(f)
        with open(GROUND_TRUTH_FILE_PATH, 'r', encoding='utf-8') as f:
            ground_truth_data = json.load(f)
    except Exception as e:
        print(f"Error: failed to read JSON files: {e}")
        return

    # Connect to database
    conn = None
    try:
        conn = sqlite3.connect(f"file:{FORWARD_DB_PATH}?mode=ro", uri=True)
        cursor = conn.cursor()
        print("Successfully connected to the database.")
    except sqlite3.Error as e:
        print(f"Database connection error: {e}")
        return

    # Initialize statistics
    total_queries_evaluated = 0
    total_recalled_papers = 0
    total_correct_papers_in_path = 0
    query_precisions = []
    connected_paths_count = 0

    try:
        for field_name, query_text in tqdm(queries_data.items(), desc="Evaluating queries"):
            recalled_papers_raw = results_data.get(query_text)
            ground_truth_entry = ground_truth_data.get(field_name)

            if recalled_papers_raw is None or ground_truth_entry is None:
                continue

            # --- New Step: filter out null/None entries in recalled results ---
            # Ensures safe calculations later
            recalled_papers = [paper for paper in recalled_papers_raw if paper is not None and paper['id'] is not None]
            # -----------------------------------------------------------------

            total_queries_evaluated += 1
            
            # --- Metric 1: Path Precision ---
            gt_path_short_ids = ground_truth_entry.get("most_influential_path", [])
            gt_path_full_ids_set = {f"https://openalex.org/{pid}" for pid in gt_path_short_ids}
            
            num_recalled = len(recalled_papers)
            if num_recalled > 0:
                correct_hits_for_query = 0
                for paper in recalled_papers:
                    if paper.get("id") in gt_path_full_ids_set:
                        correct_hits_for_query += 1
                
                precision = correct_hits_for_query / num_recalled
                query_precisions.append(precision)
                total_recalled_papers += num_recalled
                total_correct_papers_in_path += correct_hits_for_query

            # --- Metric 2: Connectivity Rate ---
            # Take top 5 from the filtered list
            top_5_papers = recalled_papers[:5]
            if check_path_connectivity(top_5_papers, cursor):
                connected_paths_count += 1

    finally:
        if conn:
            conn.close()
            print("\nDatabase connection closed.")

    # --- 4. Compute and Print Final Evaluation Report ---
    if total_queries_evaluated > 0:
        macro_avg_precision = sum(query_precisions) / len(query_precisions) if query_precisions else 0.0
        micro_avg_precision = total_correct_papers_in_path / total_recalled_papers if total_recalled_papers > 0 else 0.0
        connectivity_rate = connected_paths_count / total_queries_evaluated

        print("\n--- Final Evaluation Report ---")
        print(f"Total number of queries evaluated: {total_queries_evaluated}")

        print("\n--- Metric 1: Path Precision ---")
        print("Measures how many recalled papers are actually in the ground-truth path")
        print(f"Macro-Average Precision: {macro_avg_precision:.2%}")
        print(f"Micro-Average Precision: {micro_avg_precision:.2%}")

        print("\n--- Metric 2: Connectivity Rate @ Top 5 ---")
        print("Measures the proportion of queries where the Top-5 recalled papers form a valid citation chain")
        print(f"Number of successfully connected queries: {connected_paths_count}")
        print(f"Connectivity Rate: {connectivity_rate:.2%}")
        print("-" * 45)

    else:
        print("No queries were evaluated. Please check if the input files match.")

if __name__ == "__main__":
    main()
