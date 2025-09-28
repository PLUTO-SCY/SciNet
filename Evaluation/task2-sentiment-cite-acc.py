```python
import json
import sqlite3
import os
from tqdm import tqdm
import sys

# --- 1. Path Configuration ---

# Path to the model retrieval results file
RESULTS_FILE_PATH = "results/answers_task2_sentiment.json"

# Path to the query file
QUERIES_FILE_PATH = "queries/queries_task2_sentiment.json"

# Path to the database file (used to query references)
FORWARD_DB_PATH = "OpenAlex/sqlite/citing_to_cited.db"


# --- 2. Database Query Function ---
def get_references(paper_id: str, cursor: sqlite3.Cursor) -> list[str]:
    """
    Given an OpenAlex paper ID (short format, e.g., 'W12345'), query its reference list.
    
    Args:
        paper_id: Short paper ID.
        cursor: SQLite cursor object.
        
    Returns:
        A list of all referenced paper IDs (full format).
    """
    try:
        cursor.execute("SELECT referenced_work_ids FROM citing_to_cited WHERE work_id = ?", (paper_id,))
        row = cursor.fetchone()
        # If a record is found and the content is not empty, parse the JSON
        if row and row[0]:
            return json.loads(row[0])
    except Exception as e:
        print(f"Error querying references for {paper_id}: {e}")
    # If no record is found or an error occurs, return an empty list
    return []


# --- 3. Main Evaluation Logic ---
def main():
    """
    Main function to evaluate retrieval results.
    """
    # Check whether required files exist
    if not all(os.path.exists(p) for p in [QUERIES_FILE_PATH, RESULTS_FILE_PATH, FORWARD_DB_PATH]):
        print("Error: One or more required files (queries, results, or database) were not found. Please check the path configuration.")
        return

    print("--- Start evaluating retrieval quality ---")

    # Load queries and retrieval results
    try:
        with open(QUERIES_FILE_PATH, 'r', encoding='utf-8') as f:
            queries_data = json.load(f)
        with open(RESULTS_FILE_PATH, 'r', encoding='utf-8') as f:
            results_data = json.load(f)
        print(f"Successfully loaded {len(queries_data)} queries and {len(results_data)} retrieval results.")
    except (json.JSONDecodeError, FileNotFoundError) as e:
        print(f"Error: Failed to read JSON file: {e}")
        return

    # Connect to database
    conn = None
    try:
        conn = sqlite3.connect(f"file:{FORWARD_DB_PATH}?mode=ro", uri=True)
        cursor = conn.cursor()
        print(f"Successfully connected to database: {FORWARD_DB_PATH}")
    except sqlite3.Error as e:
        print(f"Database connection error: {e}")
        return

    # Initialize statistics
    total_queries_evaluated = 0
    total_retrieved_papers = 0
    total_correct_papers = 0
    query_accuracies = []
    
    # Use tqdm to display progress bar
    for source_paper_id_full, query_text in tqdm(queries_data.items(), desc="Evaluating queries"):
        # Extract short ID from full URL, e.g., 'W3092732263'
        source_paper_id_short = source_paper_id_full.split('/')[-1]

        # Find the corresponding retrieval list from results by query text
        retrieved_papers = results_data.get(query_text)

        if retrieved_papers is None:
            print(f"\nWarning: No retrieval list found in results file for query '{query_text[:50]}...'.")
            continue
            
        if not retrieved_papers: # If the retrieval list is empty
            accuracy = 0.0
            query_accuracies.append(accuracy)
            print(f"\nQuery '{query_text[:50]}...' returned an empty list. Accuracy: {accuracy:.2%}")
            total_queries_evaluated += 1
            continue

        correct_count_for_query = 0
        num_retrieved_for_query = len(retrieved_papers)

        for paper in retrieved_papers:
            retrieved_paper_id_full = paper.get("id")
            if not retrieved_paper_id_full:
                continue

            # Get the short ID of the retrieved paper to query the database
            retrieved_paper_id_short = retrieved_paper_id_full.split('/')[-1]
            
            # Query its reference list
            references = get_references(retrieved_paper_id_short, cursor)
            
            # Check if the source paper is in the reference list
            if source_paper_id_short in references:
                correct_count_for_query += 1

        # Calculate accuracy for current query
        accuracy = correct_count_for_query / num_retrieved_for_query if num_retrieved_for_query > 0 else 0.0
        query_accuracies.append(accuracy)

        # Update global statistics
        total_queries_evaluated += 1
        total_retrieved_papers += num_retrieved_for_query
        total_correct_papers += correct_count_for_query

        # Uncomment to print detailed results for each query
        # print(f"\n[Result] Source ID: {source_paper_id_short}")
        # print(f"  - Retrieved: {num_retrieved_for_query}")
        # print(f"  - Correct: {correct_count_for_query}")
        # print(f"  - Accuracy: {accuracy:.2%}")

    # Close database connection
    if conn:
        conn.close()
        print("\nDatabase connection closed.")

    # --- 4. Compute and Print Final Evaluation Report ---
    if total_queries_evaluated > 0:
        # Micro-Average Accuracy
        micro_avg_accuracy = total_correct_papers / total_retrieved_papers if total_retrieved_papers > 0 else 0.0
        # Macro-Average Accuracy
        macro_avg_accuracy = sum(query_accuracies) / len(query_accuracies) if query_accuracies else 0.0

        print("\n--- Final Evaluation Report ---")
        print(f"Total queries evaluated: {total_queries_evaluated}")
        print(f"Total retrieved papers: {total_retrieved_papers}")
        print(f"Total correct papers: {total_correct_papers}")
        print("-" * 25)
        print(f"Macro-Average Accuracy: {macro_avg_accuracy:.4f} ({macro_avg_accuracy:.2%})")
        print(f"Micro-Average Accuracy: {micro_avg_accuracy:.4f} ({micro_avg_accuracy:.2%})")
        print("----------------------\n")
        print("Notes:")
        print(" - Macro-Average Accuracy: First compute accuracy for each query, then average across all queries. Each query is treated equally.")
        print(" - Micro-Average Accuracy: Treats all retrieval results as one big set and computes the overall correct ratio. Queries with more retrieved results have more influence.")

    else:
        print("No queries were evaluated. Please check the input files and paths.")


if __name__ == "__main__":
    main()
```
