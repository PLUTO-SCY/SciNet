```python
import json
import sqlite3
import os
from tqdm import tqdm

# --- 1. Configure paths ---

# Path to the new model recall results file
RESULTS_FILE_PATH = "results/answers_task2_cooccur_v2.json"

# Path to the query file
QUERIES_FILE_PATH = "results/queries_task2_cooccur_v2.json"

# Path to the database file (used to query citing papers)
REVERSE_DB_PATH = "OpenAlex/sqlite/cited_to_citing.db"


# --- 2. Database query function ---
def get_citations(paper_id_short: str, cursor: sqlite3.Cursor) -> list[str]:
    """
    Query all papers that cite a given paper based on its OpenAlex short ID.
    
    Args:
        paper_id_short: Short paper ID, e.g., 'W12345'.
        cursor: Database cursor object.
        
    Returns:
        A list of citing paper IDs (in full format).
    """
    try:
        cursor.execute("SELECT citing_work_ids FROM cited_to_citing WHERE referenced_work_id = ?", (paper_id_short,))
        row = cursor.fetchone()
        if row and row[0]:
            return json.loads(row[0])
    except Exception as e:
        print(f"Error querying citations for {paper_id_short}: {e}")
    return []


# --- 3. Main evaluation logic ---
def main():
    """
    Main function to evaluate recall results (Co-citation logic).
    """
    # Check if required files exist
    if not all(os.path.exists(p) for p in [QUERIES_FILE_PATH, RESULTS_FILE_PATH, REVERSE_DB_PATH]):
        print("Error: One or more required files (queries, results, or database) not found. Please check path configuration.")
        return

    print("--- Starting evaluation of recall quality (Co-citation Logic) ---")
    print(f"Results file: {RESULTS_FILE_PATH}")


    # Load queries and recall results
    try:
        with open(QUERIES_FILE_PATH, 'r', encoding='utf-8') as f:
            queries_data = json.load(f)
        with open(RESULTS_FILE_PATH, 'r', encoding='utf-8') as f:
            results_data = json.load(f)
        print(f"Successfully loaded {len(queries_data)} queries and {len(results_data)} recall results.")
    except (json.JSONDecodeError, FileNotFoundError) as e:
        print(f"Error: Failed to read JSON file: {e}")
        return

    # Connect to the database
    conn = None
    try:
        conn = sqlite3.connect(f"file:{REVERSE_DB_PATH}?mode=ro", uri=True)
        cursor = conn.cursor()
        print(f"Successfully connected to database: {REVERSE_DB_PATH}")
    except sqlite3.Error as e:
        print(f"Database connection error: {e}")
        return

    # Initialize statistics variables
    total_queries_evaluated = 0
    total_retrieved_papers = 0
    total_correct_papers = 0
    query_accuracies = []
    
    # Iterate through all queries
    for source_paper_id_full, query_text in tqdm(queries_data.items(), desc="Evaluating co-citation queries"):
        source_paper_id_short = source_paper_id_full.split('/')[-1]

        retrieved_papers = results_data.get(query_text)
        if retrieved_papers is None:
            # print(f"\nWarning: Recall list for query '{query_text[:50]}...' not found in results file.")
            continue
        
        # For efficiency, first get the set of citing papers for the source paper A
        source_citations_set = set(get_citations(source_paper_id_short, cursor))

        # If the recall list is empty or the source paper has no citations, accuracy is 0
        if not retrieved_papers or not source_citations_set:
            if retrieved_papers:
                query_accuracies.append(0.0)
                total_retrieved_papers += len(retrieved_papers)
            total_queries_evaluated += 1
            continue

        correct_count_for_query = 0
        num_retrieved_for_query = len(retrieved_papers)

        for paper in retrieved_papers:
            retrieved_paper_id_full = paper.get("id")
            if not retrieved_paper_id_full:
                continue

            retrieved_paper_id_short = retrieved_paper_id_full.split('/')[-1]
            
            # Get the list of citing papers for the retrieved paper B
            retrieved_citations = get_citations(retrieved_paper_id_short, cursor)
            
            # Check if the two citation lists have a non-empty intersection
            # isdisjoint() checks if two sets have no common elements, faster than manual intersection
            if not source_citations_set.isdisjoint(retrieved_citations):
                correct_count_for_query += 1

        # Compute accuracy for the current query
        accuracy = correct_count_for_query / num_retrieved_for_query
        query_accuracies.append(accuracy)

        # Update global statistics
        total_queries_evaluated += 1
        total_retrieved_papers += num_retrieved_for_query
        total_correct_papers += correct_count_for_query

    # Close the database connection
    if conn:
        conn.close()
        print("\nDatabase connection closed.")

    # --- 4. Compute and print final evaluation report ---
    if total_queries_evaluated > 0:
        # Micro-average accuracy
        micro_avg_accuracy = total_correct_papers / total_retrieved_papers if total_retrieved_papers > 0 else 0.0
        # Macro-average accuracy
        macro_avg_accuracy = sum(query_accuracies) / len(query_accuracies) if query_accuracies else 0.0

        print("\n--- Final Evaluation Report (Co-citation) ---")
        print(f"Total queries evaluated: {total_queries_evaluated}")
        print(f"Total retrieved papers: {total_retrieved_papers}")
        print(f"Total correct recalls (co-cited): {total_correct_papers}")
        print("-" * 35)
        print(f"Macro-Average Accuracy: {macro_avg_accuracy:.2%}")
        print(f"Micro-Average Accuracy: {micro_avg_accuracy:.2%}")
        print("-" * 35)

    else:
        print("No queries were evaluated. Please check input files and paths.")

if __name__ == "__main__":
    main()
```
